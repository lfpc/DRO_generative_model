"""Find the active subspace of the surrogate, then walk its directions with the simulator.

The active subspace of a response f(u) is the eigen-decomposition of

    C = E_u [ grad f(u) grad f(u)^T ],

averaged over designs u drawn from the tolerance box. Its leading eigenvectors are the
directions along which f actually moves; the rest of the 43-dimensional design space is
inactive, and the eigenvalue gap says how many directions are really needed. Here f is
log H, the log of the expected hit count, so a direction is "active" when it changes the
hits by a large FACTOR rather than by a large absolute number -- the hits span decades
across the box, and without the log the estimate is dominated by whichever designs happen
to leak most.

Nothing about that is a statement about the shield until the simulator agrees, so every
direction the method returns is then walked exactly as grad_test.py walks the gradient:
the surrogate's prediction with its Bernoulli spread against the simulator on the same
muons, at the same designs.

Both halves run on the SAME full muon sample -- --n_muons 0 means every muon the file
holds, and with --muons gen it draws that many from the flow. What makes the subspace
affordable at that size is the structure of the DeepONet: with s = p(1-p),

    grad H(u) = J(u)^T T(u),   J = db/du,   T(u) = sum_x w_x s_x(u) t_x

and T is the only muon-dependent piece. It is a p-vector per design, so ONE sweep over the
muons accumulates it for every design at once, and the gradients then come from a single
backward through the branch net -- no muon work per design and no subsampling. The script
makes two sweeps: one for the subspace, one for the walks it selects.

Two more things come out of the same sweep:

  activity scores    a_i = sum_k nu_k W_ik^2, the standard per-parameter importance of the
                     method: how much of the response the parameter carries once the
                     directions are taken into account, which is not the same ranking as
                     moving it on its own
  sufficient summary H against the first active variable y = w_1.u. If the cloud collapses
                     onto a curve, the response really is a function of that one direction

The eigenvalues and the subspace itself are bootstrapped over the sampled designs, so the
report says how many directions survive resampling rather than asserting a number.

Designs are box samples scaled by a random factor in [0, 1], which spreads their radii
from the nominal design out to the box face; sampling uniformly in the box instead would
put every one of them at ||u|| ~ sqrt(D/3) = 3.8, far outside the range the walks cover.

    python active_subspace_test.py --no_sim
    python active_subspace_test.py --k 3 --n_points 5
    python active_subspace_test.py --muons data --n_repeats 3
"""
import argparse
import os
import time

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import load_surrogate
from problems import ShipMuonShieldCuda


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter,
                   argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument('--model', default='outputs/lcso_model.pt')
parser.add_argument('--muons', default='gen', choices=['gen', 'data'],
                    help="'gen' samples the generative flow (unweighted); 'data' reads the "
                         "real file, whose per-muon weights every sum then carries")
parser.add_argument('--temperature', type=float, default=1.0, help='gen: flow temperature')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_muons', type=int, default=0,
                    help='Muons for the whole analysis -- the subspace and the walks both. '
                         '0 means as many as the muon file holds, for --muons gen too')
parser.add_argument('--n_designs', type=int, default=256,
                    help='Designs the gradients are averaged over: the Monte Carlo sample '
                         'of the active-subspace estimator. The top-k subspace needs '
                         'O(k log D) of them, so a few hundred is plenty; they all ride '
                         'along in one sweep, but each costs a dot product per muon')
parser.add_argument('--k', type=int, default=3, help='Active directions to walk')
parser.add_argument('--t_max', type=float, default=1.0,
                    help='Half-length of the walks, in normalized units')
parser.add_argument('--n_points', type=int, default=5,
                    help='Points per walk (forced odd so the nominal design is one, and it '
                         'is evaluated once for all the directions)')
parser.add_argument('--n_repeats', type=int, default=1,
                    help='Simulate each design this many times; the spread over repeats is '
                         'the transport noise')
parser.add_argument('--no_sim', dest='sim', action='store_false',
                    help='Skip the simulator. The walks are still evaluated with the '
                         'surrogate over the full muon sample')
parser.add_argument('--block', type=int, default=int(2e7), help='Muons held in memory')
parser.add_argument('--chunk', type=int, default=2 ** 19, help='Muon chunk for the surrogate')
parser.add_argument('--out_dir', default='outputs')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

N_BOOT = 200
DEVICE = torch.device(args.device)
FIGS = os.path.join(args.out_dir, 'figs')
os.makedirs(FIGS, exist_ok=True)
COLS = ['px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight']
GEN = args.muons == 'gen'
OK, BAD, GREY = '#0072B2', '#D55E00', '0.6'

shield = ShipMuonShieldCuda(n_samples=1, uniform_fields=True)
phi_0 = shield.initial_phi
model, cfg = load_surrogate(args.model, phi_0)
model.eval().to(DEVICE)
for q in model.parameters():
    q.requires_grad_(False)          # only the designs carry gradients here
D = model.dim
labels = [f'M{m}:{shield.idx_mag[i].split("[")[0]}' for m, i in shield.params_idx.tolist()]
if GEN:
    shield._load_flow()              # before any seeding; see grad_test.py


def read_muons(lo, hi):
    """Muons [lo, hi) as (raw, weights); weights is None when there are none.

    Both sweeps call this with the same offsets, so they see the same muons: the file
    slices are deterministic, and the generated blocks are seeded from their offset.
    """
    if GEN:
        torch.manual_seed(args.seed + lo)
        return shield.sample_gen(n_samples=hi - lo, temperature=args.temperature), None
    with h5py.File(shield.muons_file, 'r') as f:
        x = np.stack([np.asarray(f[k][lo:hi], dtype=np.float32) for k in COLS], axis=1)
    x[:, 3], x[:, 4], x[:, 5] = 0.0, 0.0, -1.0
    x = torch.from_numpy(x)
    return x, x[:, 7]


n_file = None
if not GEN or args.n_muons <= 0:            # the only reason to touch the file
    with h5py.File(shield.muons_file, 'r') as f:
        n_file = f['px'].shape[0]
if GEN:
    n = args.n_muons if args.n_muons > 0 else n_file
    print(f'muons: {n:,} from the generative flow at temperature {args.temperature:g} '
          f'(unweighted)')
else:
    n = n_file if args.n_muons <= 0 else min(args.n_muons, n_file)
    print(f'muons: {n:,} of the {n_file:,} in {shield.muons_file} (weighted)')
print(f'model: {os.path.basename(args.model)} {cfg["model_type"]}, p = {cfg["p"]}, D = {D}')

# =============================================================================
# Sweep 1: H and the frozen-trunk vector T of every design, over all the muons
# =============================================================================
# Row 0 is the nominal design; the rest is the Monte Carlo sample the subspace averages
# over. The trunk is evaluated once per muon chunk and shared by every design.
torch.manual_seed(args.seed + 1)
U = torch.cat([torch.zeros(1, D),
               model.sample_phi(args.n_designs) * torch.rand(args.n_designs, 1)])
MD = U.shape[0]
with torch.no_grad():
    B = model.branch_net(U.to(DEVICE))                                # (MD, p)
P = B.shape[1]
H_acc = torch.zeros(MD, dtype=torch.float64, device=DEVICE)
T_acc = torch.zeros(MD, P, dtype=torch.float64, device=DEVICE)
DES = max(1, 2 ** 24 // args.chunk)          # designs held against one muon chunk
print(f'sweep 1: {MD} designs x {n:,} muons = {MD * n:.3g} design-muon evaluations')
t0 = time.time()
for lo in range(0, n, args.block):
    raw, w_mu = read_muons(lo, min(lo + args.block, n))
    with torch.no_grad():
        for c in range(0, raw.shape[0], args.chunk):
            xb = raw[c:c + args.chunk, :7].to(DEVICE)
            t = model.trunk_net(model.normalize_muons(xb))             # (chunk, p)
            wb = None if w_mu is None else w_mu[c:c + args.chunk].to(DEVICE)
            for j in range(0, MD, DES):
                pr = torch.sigmoid(B[j:j + DES] @ t.T + model.bias)    # (m, chunk)
                sw = pr * (1 - pr)
                if wb is not None:
                    pr, sw = pr * wb, sw * wb
                H_acc[j:j + DES] += pr.double().sum(1)
                T_acc[j:j + DES] += (sw @ t).double()
    print(f'  sweep 1: {min(lo + args.block, n):,}/{n:,} muons ({time.time() - t0:.0f}s)')
    del raw, w_mu

# grad_j = J(u_j)^T T_j = d/du [b(u).T_j] with T_j frozen, and the rows are independent,
# so one backward through the branch net gives every design's gradient at once.
Ud = U.to(DEVICE).detach().requires_grad_(True)
(model.branch_net(Ud) * T_acc.float()).sum().backward()
G_all = Ud.grad.detach().cpu().numpy()
H_all = H_acc.cpu().numpy()
H_0, g_0 = H_all[0], G_all[0]
U_mc, H_mc, G_mc = U[1:].numpy(), H_all[1:], G_all[1:]
print(f'nominal design: {H_0:,.2f} {"" if GEN else "weighted "}hits')
if H_0 <= 0:
    raise SystemExit('the surrogate predicts no hits at the nominal design')

# Free consistency check on the gradient: the nearest design's rise, by difference,
# against what the gradient at the nominal design predicts along the same direction.
j = int(np.argmin(np.linalg.norm(U_mc, axis=1)))
r = float(np.linalg.norm(U_mc[j]))
print(f'  check: the nearest design sits at ||u|| = {r:.3f}, where the difference gives a '
      f'directional derivative of {(H_mc[j] - H_0) / max(r, 1e-30):.4g} against '
      f'{g_0 @ (U_mc[j] / max(r, 1e-30)):.4g} from the gradient')

# =============================================================================
# The active subspace
# =============================================================================
G = G_mc / np.maximum(H_mc, 1e-30)[:, None]            # grad log H
C = G.T @ G / len(G)
nu, W = np.linalg.eigh(C)
nu, W = np.clip(nu[::-1], 0, None).copy(), W[:, ::-1].copy()
print(f'\nsubspace from {len(G)} design gradients; ||u|| from '
      f'{np.linalg.norm(U_mc, axis=1).min():.2f} to '
      f'{np.linalg.norm(U_mc, axis=1).max():.2f}')

# Bootstrap over the designs: eigenvalues, and the distance between the true k-dimensional
# subspace and the resampled one, which is what says how many directions are resolved.
rng = np.random.default_rng(args.seed + 2)
K_MAX = min(6, D)
nu_boot = np.zeros((N_BOOT, D))
sub_dist = np.zeros((N_BOOT, K_MAX))
for b in range(N_BOOT):
    Gb = G[rng.integers(0, len(G), len(G))]
    nb, Wb = np.linalg.eigh(Gb.T @ Gb / len(Gb))
    nb, Wb = nb[::-1], Wb[:, ::-1]
    nu_boot[b] = nb
    for k in range(K_MAX):
        A, Bk = W[:, :k + 1], Wb[:, :k + 1]
        sub_dist[b, k] = np.linalg.norm(A @ A.T - Bk @ Bk.T, 2)
nu_lo, nu_hi = np.percentile(nu_boot, [5, 95], axis=0)
dist = sub_dist.mean(0)
act = (nu[None, :K_MAX] * W[:, :K_MAX] ** 2).sum(1)     # activity scores
gap = nu[:-1] / np.maximum(nu[1:], 1e-300)

print(f'\n{"k":>3s} {"eigenvalue":>12s} {"90% CI":>26s} {"share":>8s} {"gap to k+1":>11s} '
      f'{"subspace dist":>14s}')
for k in range(K_MAX):
    print(f'{k + 1:3d} {nu[k]:12.4g} [{nu_lo[k]:11.4g},{nu_hi[k]:11.4g}] '
          f'{nu[k] / max(nu.sum(), 1e-300):8.1%} {gap[k]:11.2f} {dist[k]:14.3f}')
n_res = int(np.sum(dist < 0.3))
print(f'  the subspace distance is the mean over {N_BOOT} bootstrap resamples of '
      f'||W_k W_k^T - W_k* W_k*^T||_2:\n  0 means the k-dimensional subspace is pinned '
      f'down, 1 means it is not. {n_res} of the first {K_MAX} are below 0.3.')
print(f'  the largest gap is at k = {int(np.argmax(gap[:K_MAX])) + 1} '
      f'({gap[:K_MAX].max():.1f}x), which is where the method says to cut the subspace.')

print(f'\nactivity scores (a_i = sum_k nu_k W_ik^2 over the top {K_MAX}):')
order = np.argsort(-act)
for i in order[:12]:
    print(f'  {labels[i]:22s} {act[i] / max(act.sum(), 1e-300):6.1%}')

g0_hat = g_0 / max(np.linalg.norm(g_0), 1e-300)
print(f'\nalignment with the gradient at the nominal design: '
      + ', '.join(f'w{k + 1} {abs(g0_hat @ W[:, k]):.2f}' for k in range(min(3, D))))

# =============================================================================
# The directions to walk
# =============================================================================
# Eigenvectors have no sign; point each one the way that raises the hits, so +t is the
# bad direction in every panel.
K = max(1, min(args.k, D))
dirs = []
for k in range(K):
    w = W[:, k].copy()
    proj = float(g_0 @ w)
    if proj == 0:
        proj = float(G.mean(0) @ w)
    dirs.append(w if proj >= 0 else -w)

n_pts = args.n_points + (args.n_points + 1) % 2
ts = np.linspace(-args.t_max, args.t_max, n_pts)
U_list = [np.zeros(D)]              # index 0: the nominal design, walked once for all
idx_of = []
for w in dirs:
    row = []
    for t in ts:
        if abs(t) < 1e-12:
            row.append(0)
        else:
            row.append(len(U_list))
            U_list.append(t * w)
    idx_of.append(row)
U_all = np.clip(np.stack(U_list), -1, 1)
PHI = model.denormalize_phi(torch.from_numpy(U_all).float().to(DEVICE)).cpu()
n_des, R = len(U_list), max(1, args.n_repeats)

print(f'\nsweep 2: {K} directions, {n_des} designs over {n:,} muons'
      + (f' x {R} repeats' if args.sim else ' (surrogate only)'))
for k, w in enumerate(dirs):
    lead = ', '.join(f'{labels[i]}{w[i]:+.2f}' for i in np.argsort(-np.abs(w))[:4])
    print(f'  w{k + 1}: {lead}')

# =============================================================================
# Sweep 2: walk them with the surrogate and the simulator, on the same muons
# =============================================================================
pred = np.zeros(n_des)          # sum_x w p         expected hits
pvar = np.zeros(n_des)          # sum_x w^2 p(1-p)  spread of one simulator draw
sim = np.zeros((R, n_des))      # sum_x w hit       simulated hits, per repeat
svar = np.zeros((R, n_des))     # sum_x w^2 hit     its counting error
PHI_D = PHI.to(DEVICE)
t0 = time.time()
for lo in range(0, n, args.block):
    raw, w_mu = read_muons(lo, min(lo + args.block, n))
    with torch.no_grad():
        for c in range(0, raw.shape[0], args.chunk):
            xb = raw[c:c + args.chunk, :7].to(DEVICE)
            pr = model.predict_proba(PHI_D, xb.unsqueeze(0))          # (n_des, chunk)
            q = pr * (1 - pr)
            if w_mu is not None:
                wb = w_mu[c:c + args.chunk].to(DEVICE)
                pr, q = pr * wb, q * wb ** 2
            pred += pr.sum(1).double().cpu().numpy()
            pvar += q.sum(1).double().cpu().numpy()
    if args.sim:
        wd = None if w_mu is None else w_mu.double()
        for i in range(n_des):
            for rr in range(R):
                hits = shield(PHI[i], raw).double()   # a new transport seed every call
                sim[rr, i] += float(hits.sum() if wd is None else (hits * wd).sum())
                svar[rr, i] += float(hits.sum() if wd is None else (hits * wd ** 2).sum())
        del wd
    print(f'  sweep 2: {min(lo + args.block, n):,}/{n:,} muons ({time.time() - t0:.0f}s)')
    del raw, w_mu

pstd = np.sqrt(pvar)
smean = sim.mean(0)
serr = sim.std(0, ddof=1) if R > 1 else np.sqrt(svar).mean(0)

print(f'\n{"dir":>5s} {"t":>7s} {"surrogate":>15s} {"+- std":>12s}'
      + (f' {"simulator":>15s} {"+- err":>12s} {"ratio":>7s}' if args.sim else ''))
for k in range(K):
    for j, t in enumerate(ts):
        i = idx_of[k][j]
        line = (f'{("w" + str(k + 1)) if j == 0 else "":>5s} {t:+7.3f} {pred[i]:15,.1f} '
                f'{pstd[i]:12,.1f}')
        if args.sim:
            rat = f'{pred[i] / smean[i]:7.3g}' if smean[i] > 0 else f'{"-":>7s}'
            line += f' {smean[i]:15,.1f} {serr[i]:12,.1f} {rat}'
        print(line)
print(f'\nH/H_0 at the ends of each walk (surrogate'
      + (', simulator)' if args.sim else ')'))
for k in range(K):
    lo_i, hi_i = idx_of[k][0], idx_of[k][-1]
    s = (f'  w{k + 1}: {pred[lo_i] / max(pred[0], 1e-30):.3g} at t=-{args.t_max:g}, '
         f'{pred[hi_i] / max(pred[0], 1e-30):.3g} at t=+{args.t_max:g}')
    if args.sim and smean[0] > 0:
        s += f'   simulator {smean[lo_i] / smean[0]:.3g}, {smean[hi_i] / smean[0]:.3g}'
    elif args.sim:
        s += '   simulator: no hits at the nominal design, nothing to compare'
    print(s)

# =============================================================================
# Figures
# =============================================================================
SRC = (f'generative muons, T = {args.temperature:g}' if GEN else 'real muons, weighted')
HITS = 'hits' if GEN else 'weighted hits'

# --- 1. spectrum and how well it is resolved --------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
ax[0].errorbar(np.arange(1, D + 1), np.maximum(nu, 1e-300),
               yerr=[np.maximum(nu - nu_lo, 0), np.maximum(nu_hi - nu, 0)],
               fmt='o-', ms=4, color=OK, lw=1.5, capsize=2)
ax[0].set(xlabel='index', ylabel='eigenvalue of $C$', yscale='log',
          title='Active-subspace spectrum (90% bootstrap)')
ax[1].plot(np.arange(1, K_MAX + 1), dist, 's-', ms=5, color=BAD)
ax[1].axhline(0.3, color=GREY, lw=1, ls=':')
ax[1].set(xlabel='subspace dimension $k$', ylabel='bootstrap subspace distance',
          ylim=(0, 1), title='How well the subspace is pinned down')
fig.savefig(os.path.join(FIGS, 'active_subspace_spectrum.png'), dpi=150)
plt.close(fig)

# --- 2. sufficient summary: is the response a function of one direction? ----
y1 = U_mc @ dirs[0]
y2 = U_mc @ (dirs[1] if K > 1 else dirs[0])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
ax[0].scatter(y1, H_mc, s=8, color=OK, alpha=0.6, edgecolors='none')
ax[0].axhline(H_0, color=GREY, lw=1, ls='--')
ax[0].set(xlabel='$y_1 = w_1^T u$', ylabel=f'surrogate {HITS}', yscale='log',
          title='Sufficient summary plot')
sc = ax[1].scatter(y1, y2, c=np.log10(np.maximum(H_mc, 1e-30)), s=8, cmap='viridis')
fig.colorbar(sc, ax=ax[1], label='$\\log_{10}$ hits')
ax[1].set(xlabel='$y_1$', ylabel='$y_2$', title='The first two active variables')
fig.suptitle(f'{len(U_mc)} designs, {n:,} muons, {SRC}')
fig.savefig(os.path.join(FIGS, 'active_subspace_summary.png'), dpi=150)
plt.close(fig)

# --- 3. what the directions are made of -------------------------------------
M = np.stack(dirs)
fig, ax = plt.subplots(2, 1, figsize=(max(9, D * 0.28), 3.2 + 0.5 * K),
                       constrained_layout=True,
                       gridspec_kw={'height_ratios': [1, max(1, K) / 2]})
o = np.argsort(-act)
ax[0].bar(np.arange(D), act[o] / max(act.sum(), 1e-300), color=OK)
ax[0].set_xticks(np.arange(D), [labels[i] for i in o], rotation=90, fontsize=7)
ax[0].set(ylabel='activity score', title='Per-parameter activity and the directions')
lim = np.abs(M).max()
im = ax[1].imshow(M[:, o], cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
ax[1].set_xticks(np.arange(D), [labels[i] for i in o], rotation=90, fontsize=7)
ax[1].set_yticks(np.arange(K), [f'$w_{k + 1}$' for k in range(K)])
fig.colorbar(im, ax=ax[1], orientation='horizontal', pad=0.35, label='component')
fig.savefig(os.path.join(FIGS, 'active_subspace_directions.png'), dpi=150)
plt.close(fig)

# --- 4. the walks, as in grad_test.py ---------------------------------------
nc = min(3, K)
nr = int(np.ceil(K / nc))
fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.9 * nr), constrained_layout=True,
                         squeeze=False)
for a in axes.ravel()[K:]:
    a.axis('off')
for k, a in enumerate(axes.ravel()[:K]):
    i = np.array(idx_of[k])
    a.plot(ts, pred[i], color=OK, lw=2, marker='o', ms=4, label='surrogate')
    a.fill_between(ts, np.maximum(pred[i] - pstd[i], 0), pred[i] + pstd[i], color=OK,
                   alpha=0.25, lw=0, label='surrogate $\\pm$ std')
    if args.sim:
        a.errorbar(ts, smean[i], yerr=serr[i], fmt='s', ms=6, color=BAD, capsize=3,
                   lw=1.5, label=f'simulator (mean of {R})' if R > 1 else 'simulator')
        if R > 1:
            a.plot(np.repeat(ts[None], R, 0), sim[:, i], '.', ms=3, color=BAD, alpha=0.35,
                   zorder=0, label='_')
    a.axvline(0, color=GREY, lw=1, ls='--')
    a.set_xlabel(f'step along $w_{k + 1}$ (normalized units)')
    a.set_ylabel(HITS)
    a.set_title(f'$w_{k + 1}$: $\\nu = {nu[k]:.3g}$, {nu[k] / max(nu.sum(), 1e-300):.0%} '
                f'of the spectrum', fontsize=10)
    curves = [pred[i]] + ([smean[i]] if args.sim else [])
    if all(np.all(c > 0) for c in curves):
        a.set_yscale('log')
    if k == 0:
        a.legend(fontsize=8)
fig.suptitle(f'Active directions: surrogate vs. simulator ({SRC}, {n:,} muons)')
fig.savefig(os.path.join(FIGS, 'active_subspace_walks.png'), dpi=150)
plt.close(fig)

np.savez(os.path.join(args.out_dir, 'active_subspace.npz'),
         labels=np.array(labels), phi_0=phi_0.numpy(), muons=args.muons,
         temperature=args.temperature, n_muons=n, n_designs=len(U_mc), H_0=H_0,
         designs_u=U_mc, designs_H=H_mc, designs_grad=G_mc,
         C=C, eigvals=nu, eigvecs=W, eigvals_lo=nu_lo, eigvals_hi=nu_hi,
         subspace_dist=dist, activity=act, grad_0=g_0,
         dirs=M, t=ts, dir_idx=np.array(idx_of), walk_u=U_all, walk_phi=PHI.numpy(),
         pred=pred, pred_std=pstd, sim=sim, sim_mean=smean, sim_err=serr)
print(f'\nsaved {FIGS}/active_subspace_{{spectrum,summary,directions,walks}}.png and '
      f'{args.out_dir}/active_subspace.npz')
