"""Active subspace of the surrogate's log hits, then walk its top directions with the simulator.

C = E_u[grad log H grad log H^T] over designs u in the training box; its leading
eigenvectors are walked like grad_test.py walks the gradient, surrogate vs. simulator on the
same muons. Eigenvalues and subspaces are bootstrapped over the designs.

    python active_subspace_test.py --no_sim
    python active_subspace_test.py --k 3 --n_points 5
    python active_subspace_test.py --muons data --n_repeats 3
"""
import argparse
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import load_surrogate
from muons_io import Muons, count_muons, evaluate_designs, param_labels
from problems import ShipMuonShieldCuda


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter,
                   argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument('--model', default='outputs/lcso_model.pt', help='surrogate checkpoint')
parser.add_argument('--muons', default='gen', choices=['gen', 'data'],
                    help="gen: generative flow (unweighted); data: real file (weighted)")
parser.add_argument('--temperature', type=float, default=1.0, help='gen: flow temperature')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--n_muons', type=int, default=0, help='0 = size of the muon file')
parser.add_argument('--n_designs', type=int, default=256,
                    help='designs the gradient outer product is averaged over')
parser.add_argument('--k', type=int, default=3, help='active directions to walk')
parser.add_argument('--t_max', type=float, default=1.0, help='walk half-length, normalized')
parser.add_argument('--n_points', type=int, default=5, help='points per walk (made odd)')
parser.add_argument('--n_repeats', type=int, default=1, help='simulator runs per design')
parser.add_argument('--no_sim', dest='sim', action='store_false', help='surrogate only')
parser.add_argument('--block', type=int, default=int(2e7), help='muons held in memory')
parser.add_argument('--chunk', type=int, default=2 ** 19, help='muon chunk for the surrogate')
parser.add_argument('--out_dir', default='outputs', help='output directory')
parser.add_argument('--device', default='cuda', help='torch device')
args = parser.parse_args()

N_BOOT = 200
DEVICE = torch.device(args.device)
FIGS = os.path.join(args.out_dir, 'figs')
os.makedirs(FIGS, exist_ok=True)
GEN = args.muons == 'gen'
OK, BAD, GREY = '#0072B2', '#D55E00', '0.6'

shield = ShipMuonShieldCuda(n_samples=1, uniform_fields=True)
phi_0 = shield.initial_phi
model, cfg = load_surrogate(args.model, phi_0)
model.eval().to(DEVICE)
for q in model.parameters():
    q.requires_grad_(False)          # only the designs carry gradients here
D = model.dim
labels = param_labels(shield)
muons = Muons(shield, args.muons, args.seed, args.temperature)
n = count_muons(shield, args.n_muons, GEN)
print(f'muons: {n:,} from {muons.describe()}')
print(f'model: {os.path.basename(args.model)} {cfg["model_type"]}, p = {cfg["p"]}, D = {D}')

# =============================================================================
# Sweep 1: H and the frozen-trunk vector T of every design, over all the muons
# =============================================================================
# Row 0 is the nominal design; the rest is the Monte Carlo sample the subspace averages
# over. The trunk is evaluated once per muon chunk and shared by every design.
torch.manual_seed(args.seed + 1)
U_box = model.sample_phi(args.n_designs, seed=args.seed + 1)
U = torch.cat([torch.zeros(1, D), U_box * torch.rand(args.n_designs, 1)])
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
    hi = min(lo + args.block, n)
    raw, w_mu = muons.read(lo, hi)
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
    print(f'  sweep 1: {hi:,}/{n:,} muons ({time.time() - t0:.0f}s)')
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

# Gradient check: finite difference to the nearest design vs. the nominal gradient.
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

# Bootstrap over the designs: eigenvalue CIs, and the distance between the k-dim subspace
# and its resampled version (how many directions are actually resolved).
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
# Sign each eigenvector so that +t raises the hits.
K = min(args.k, D)
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
n_des, R = len(U_list), args.n_repeats

print(f'\nsweep 2: {K} directions, {n_des} designs over {n:,} muons'
      + (f' x {R} repeats' if args.sim else ' (surrogate only)'))
for k, w in enumerate(dirs):
    lead = ', '.join(f'{labels[i]}{w[i]:+.2f}' for i in np.argsort(-np.abs(w))[:4])
    print(f'  w{k + 1}: {lead}')

# =============================================================================
# Sweep 2: walk them with the surrogate and the simulator, on the same muons
# =============================================================================
pred, pvar, sim, svar, _ = evaluate_designs(model, shield, muons, PHI, n, args.block,
                                            args.chunk, R, args.sim, DEVICE)
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
                       gridspec_kw={'height_ratios': [1, K / 2]})
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
