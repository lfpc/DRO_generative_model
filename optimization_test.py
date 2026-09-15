"""Take one optimization step off the nominal design and check the gain with the simulator.

Two steps are computed from the same derivatives, both aimed at MINIMIZING the expected
number of hits:

  steepest descent   p along -grad H
  trust-region Newton  p from argmin over ||p|| <= Delta of  g.p + 0.5 p^T B p, solved
                       exactly (More-Sorensen). B is indefinite here, so this is not
                       Newton's method with a safeguard -- the negative-curvature
                       directions are what the subproblem exploits.

--step decides how far to go along each:

  armijo   backtrack t = t0, t0/2, t0/4, ... and accept the first step with a sufficient
           decrease, H(u+p) <= H(u) + c1 g.p
  trust    shrink the radius Delta = t0, t0/2, ... and accept the largest one whose actual
           reduction is at least eta of the reduction the quadratic model predicted

--eval decides what "H(u+p)" means in those tests: the surrogate, or the simulator itself.
Either way both are reported, so the run always says what the surrogate promised and what
the simulator delivered.

The trust region is the region the model was trained in. Designs are normalized so that
the training box is |u_i| <= 1, and an L2 ball of radius 1 is the largest ball inside it,
so t0 = 1 means "as far as the training data reaches, in any direction" and every step
with ||p|| <= 1 is inside it by construction.

The derivatives come from one sweep over the muons. The muon side of the DeepONet does not
depend on the design, so with s = p(1-p) the accumulators

    T = sum_x w_x s_x t_x,     A = sum_x w_x s_x (1 - 2 p_x) t_x t_x^T

give grad = J^T T and hess = J^T A J + d2(b.T)/du2 exactly, with J = db/du and the branch
Hessian taken by autograd on the branch net alone -- 43 inputs, no muons, so the Hessian
costs nothing extra. A second sweep evaluates every candidate step, surrogate and
simulator, on the same muons.

    python optimization_test.py
    python optimization_test.py --step trust --eval simulator
    python optimization_test.py --muons data --n_repeats 3
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
                    help='Muons for the derivatives and the steps alike (0 = as many as '
                         'the muon file holds, for --muons gen too)')
parser.add_argument('--step', default='armijo', choices=['armijo', 'trust'],
                    help='armijo: backtrack until the decrease is sufficient. trust: '
                         'shrink the radius until the model is trusted, i.e. the actual '
                         'reduction reaches eta of the predicted one')
parser.add_argument('--eval', default='surrogate', choices=['surrogate', 'simulator'],
                    help='What the step test measures H with. The other one is reported '
                         'either way')
parser.add_argument('--t0', type=float, default=1.0,
                    help='First step length / trust radius, in normalized units. 1 is the '
                         'largest ball inside the training box')
parser.add_argument('--n_back', type=int, default=6,
                    help='Rungs in the backtracking ladder. They are all evaluated in one '
                         'sweep, so this costs muon work but no extra passes')
parser.add_argument('--n_repeats', type=int, default=1,
                    help='Simulate each candidate this many times; the spread over repeats '
                         'is the transport noise')
parser.add_argument('--no_sim', dest='sim', action='store_false',
                    help='Skip the simulator entirely (forces --eval surrogate)')
parser.add_argument('--block', type=int, default=int(2e7), help='Muons held in memory')
parser.add_argument('--chunk', type=int, default=2 ** 19, help='Muon chunk for the surrogate')
parser.add_argument('--out_dir', default='outputs')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

BETA, C1, ETA = 0.5, 1e-4, 0.25     # backtracking factor, Armijo constant, trust threshold
if not args.sim:
    args.eval = 'surrogate'
torch.backends.cuda.matmul.allow_tf32 = False       # see robustness.py

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
if not GEN or args.n_muons <= 0:
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
print(f'step rule: {args.step}, tested on the {args.eval}, from t0 = {args.t0:g}')

# =============================================================================
# Sweep 1: the derivatives at the nominal design
# =============================================================================
u_0 = torch.zeros(D, device=DEVICE)
b_0 = model.branch_net(u_0.unsqueeze(0)).squeeze(0).detach()
P = b_0.numel()
T_acc = torch.zeros(P, dtype=torch.float64, device=DEVICE)
A_acc = torch.zeros(P, P, dtype=torch.float64, device=DEVICE)
H_0, var_0, sumw = 0.0, 0.0, 0.0
t0 = time.time()
for lo in range(0, n, args.block):
    raw, w_mu = read_muons(lo, min(lo + args.block, n))
    with torch.no_grad():
        for c in range(0, raw.shape[0], args.chunk):
            xb = raw[c:c + args.chunk, :7].to(DEVICE)
            t = model.trunk_net(model.normalize_muons(xb))
            pr = torch.sigmoid(t @ b_0 + model.bias)
            s = pr * (1 - pr)
            wb = None if w_mu is None else w_mu[c:c + args.chunk].to(DEVICE)
            ws = s if wb is None else s * wb
            H_0 += float((pr if wb is None else pr * wb).double().sum())
            var_0 += float((s if wb is None else s * wb ** 2).double().sum())
            sumw += float(xb.shape[0] if wb is None else wb.double().sum())
            T_acc += (ws @ t).double()
            A_acc += ((t * (ws * (1 - 2 * pr)).unsqueeze(1)).T @ t).double()
    print(f'  sweep 1: {min(lo + args.block, n):,}/{n:,} muons ({time.time() - t0:.0f}s)')
    del raw, w_mu

J = torch.autograd.functional.jacobian(
    lambda u: model.branch_net(u.unsqueeze(0)).squeeze(0), u_0).detach()
Hb = torch.autograd.functional.hessian(
    lambda u: model.branch_net(u.unsqueeze(0)).squeeze(0) @ T_acc.float(), u_0).detach()
Jd = J.double()
g = (Jd.T @ T_acc).cpu().numpy()                          # dH/du
B = (Jd.T @ A_acc @ Jd + Hb.double()).cpu().numpy()
B = 0.5 * (B + B.T)
if H_0 <= 0:
    raise SystemExit('the surrogate predicts no hits at the nominal design')

ev = np.linalg.eigvalsh(B)
g_hat = g / max(np.linalg.norm(g), 1e-300)
print(f'\nH_0 = {H_0:,.4g} {"" if GEN else "weighted "}hits over {n:,} muons '
      f'(rate {H_0 / max(sumw, 1e-30):.3e}), spread of one simulator draw '
      f'{np.sqrt(var_0):,.4g}')
print(f'|grad| = {np.linalg.norm(g):.4g} ({np.linalg.norm(g) / H_0:+.3f} of H_0 per '
      f'normalized unit); Hessian eigenvalues from {ev[0]:+.4g} to {ev[-1]:+.4g}, '
      f'curvature along -grad {g_hat @ B @ g_hat:+.4g}')
print('  the steepest-descent direction is led by: ' + ', '.join(
    f'{labels[i]}{-g_hat[i]:+.2f}' for i in np.argsort(-np.abs(g))[:5]))

# =============================================================================
# The candidate steps
# =============================================================================
def trs_min(gv, Bm, delta):
    """argmin of g.p + 0.5 p^T B p over ||p|| <= delta, exactly (More-Sorensen).

    The solution satisfies (B + lam I) p = -g with lam >= max(0, -min eig B), and
    ||p(lam)|| falls monotonically in lam, so one bisection finds it. D = 43, so the
    eigendecomposition this is written in terms of is free.
    """
    e, Q = np.linalg.eigh(Bm)
    gh = Q.T @ gv
    if e[0] > 0:                                  # convex: the free minimum may be inside
        p = -Q @ (gh / e)
        if np.linalg.norm(p) <= delta:
            return p, 0.0, 'interior'
    lam_lo = max(0.0, -e[0])

    def p_of(lam):
        return -Q @ (gh / (e + lam))

    lam = lam_lo + 1e-12 * max(1.0, abs(lam_lo))
    if np.linalg.norm(p_of(lam)) < delta:
        # Hard case: g has no component on the smallest eigenvector, so no lam reaches the
        # boundary. Solve in the rest of the space and step out along that direction.
        keep = e > e[0] + 1e-12 * max(1.0, abs(e[0]))
        p = -Q[:, keep] @ (gh[keep] / (e[keep] + lam_lo))
        alpha = np.sqrt(max(delta ** 2 - p @ p, 0.0))
        return p - alpha * Q[:, 0] * np.sign(gh[0] or 1.0), lam_lo, 'boundary (hard case)'
    hi = lam + np.linalg.norm(gv) / max(delta, 1e-30)
    while np.linalg.norm(p_of(hi)) > delta:
        hi = lam + 2 * (hi - lam)
    for _ in range(200):
        mid = 0.5 * (lam + hi)
        lam, hi = (mid, hi) if np.linalg.norm(p_of(mid)) > delta else (lam, mid)
    return p_of(0.5 * (lam + hi)), 0.5 * (lam + hi), 'boundary'


def model_drop(p):
    """The decrease the quadratic model predicts: m(0) - m(p)."""
    return float(-(g @ p + 0.5 * p @ B @ p))


radii = args.t0 * BETA ** np.arange(max(1, args.n_back))    # longest first: backtracking
sd = [(-r * g_hat, r, 'descent') for r in radii]
tr = []
for r in radii:
    p, lam, kind = trs_min(g, B, r)
    tr.append((p, r, kind))
METHODS = [{'name': 'steepest descent', 'short': 'grad', 'rungs': sd},
           {'name': 'trust-region Newton', 'short': 'newton', 'rungs': tr}]

# Every candidate of every method goes into one batch, so the whole line search is a
# single sweep over the muons.
U_list = [np.zeros(D)]
for m in METHODS:
    m['idx'] = []
    for p, r, kind in m['rungs']:
        m['idx'].append(len(U_list))
        U_list.append(p)
U_all = np.stack(U_list)
inside = np.abs(U_all).max(1) <= 1 + 1e-9
if not inside.all():
    print(f'note: {int((~inside).sum())} candidate steps leave the training box; they are '
          f'kept as they are and flagged in the table')
PHI = model.denormalize_phi(torch.from_numpy(U_all).float().to(DEVICE)).cpu()
n_des, R = len(U_list), max(1, args.n_repeats)
print(f'\nsweep 2: {n_des} designs ({len(METHODS)} methods x {len(radii)} rungs, plus the '
      f'nominal) over {n:,} muons' + (f' x {R} repeats' if args.sim else ' (surrogate '
      f'only)'))

# =============================================================================
# Sweep 2: evaluate every candidate, surrogate and simulator, on the same muons
# =============================================================================
pred = np.zeros(n_des)
pvar = np.zeros(n_des)
sim = np.zeros((R, n_des))
svar = np.zeros((R, n_des))
PHI_D = PHI.to(DEVICE)
t0 = time.time()
for lo in range(0, n, args.block):
    raw, w_mu = read_muons(lo, min(lo + args.block, n))
    with torch.no_grad():
        for c in range(0, raw.shape[0], args.chunk):
            xb = raw[c:c + args.chunk, :7].to(DEVICE)
            pr = model.predict_proba(PHI_D, xb.unsqueeze(0))
            q = pr * (1 - pr)
            if w_mu is not None:
                wb = w_mu[c:c + args.chunk].to(DEVICE)
                pr, q = pr * wb, q * wb ** 2
            pred += pr.sum(1).double().cpu().numpy()
            pvar += q.sum(1).double().cpu().numpy()
    if args.sim:
        wd = None if w_mu is None else w_mu.double()
        for i in range(n_des):
            for r in range(R):
                hits = shield(PHI[i], raw).double()   # a new transport seed every call
                sim[r, i] += float(hits.sum() if wd is None else (hits * wd).sum())
                svar[r, i] += float(hits.sum() if wd is None else (hits * wd ** 2).sum())
        del wd
    print(f'  sweep 2: {min(lo + args.block, n):,}/{n:,} muons ({time.time() - t0:.0f}s)')
    del raw, w_mu

pstd = np.sqrt(pvar)
smean = sim.mean(0)
serr = sim.std(0, ddof=1) if R > 1 else np.sqrt(svar).mean(0)
H_sur, H_sim = pred, smean
H_ref = H_sim if args.eval == 'simulator' else H_sur      # what the step test believes

# The smallest step is a free check on the derivatives: its measured drop should match
# what the model predicts, to the accuracy of the third-order term.
p_small = METHODS[0]['rungs'][-1][0]
i_small = METHODS[0]['idx'][-1]
print(f'\ncheck: the shortest descent step, ||p|| = {np.linalg.norm(p_small):.3g}, drops '
      f'the surrogate by {H_sur[0] - H_sur[i_small]:.4g} against {model_drop(p_small):.4g} '
      f'from the model')

# =============================================================================
# Apply the step rule
# =============================================================================
def accept(m):
    """First rung of the ladder the rule accepts, with the reason."""
    for j, (p, r, kind) in enumerate(m['rungs']):
        i = m['idx'][j]
        drop = H_ref[0] - H_ref[i]                            # actual decrease
        if args.step == 'armijo':
            need = -C1 * (g @ p)                              # sufficient decrease
            if drop >= need:
                return j, f'Armijo satisfied: dropped {drop:.4g} >= {need:.4g}'
        else:
            pm = model_drop(p)
            rho = drop / pm if pm > 0 else -np.inf
            if rho >= ETA:
                return j, f'rho = {rho:.3f} >= {ETA}'
    return None, 'no rung passed the test'


print('\n' + '=' * 79)
for m in METHODS:
    print(f'{m["name"]}')
    print(f'{"||p||":>8s} {"kind":>18s} {"model drop":>12s} {"surrogate":>14s} '
          f'{"drop":>10s}' + (f' {"simulator":>14s} {"drop":>10s} {"rho":>7s}'
                              if args.sim else '') + '   box')
    for j, (p, r, kind) in enumerate(m['rungs']):
        i = m['idx'][j]
        pm = model_drop(p)
        line = (f'{np.linalg.norm(p):8.3f} {kind:>18s} {pm:12.4g} {H_sur[i]:14.6g} '
                f'{H_sur[0] - H_sur[i]:10.4g}')
        if args.sim:
            rho = (H_sim[0] - H_sim[i]) / pm if pm > 0 else np.nan
            line += (f' {H_sim[i]:14.6g} {H_sim[0] - H_sim[i]:10.4g} {rho:7.2f}')
        print(line + ('   in' if np.abs(U_all[i]).max() <= 1 + 1e-9 else '   OUT'))
    j, why = accept(m)
    m['accepted'] = j
    if j is None:
        print(f'  -> no step accepted ({why}); the model does not predict this response '
              f'anywhere on the ladder')
        continue
    i = m['idx'][j]
    m['i'] = i
    print(f'  -> accepted ||p|| = {np.linalg.norm(m["rungs"][j][0]):.3f} ({why})')
    print(f'     surrogate {H_sur[0]:.6g} -> {H_sur[i]:.6g}, gain '
          f'{1 - H_sur[i] / H_sur[0]:+.1%}')
    if args.sim:
        e = np.hypot(serr[i], serr[0])
        print(f'     simulator {H_sim[0]:.6g} -> {H_sim[i]:.6g} +- {e:.4g}, gain '
              f'{1 - H_sim[i] / max(H_sim[0], 1e-30):+.1%}'
              + (f' ({(H_sim[0] - H_sim[i]) / e:+.1f} sigma)' if e > 0 else ''))
    print(f'     largest moves: ' + ', '.join(
        f'{labels[k]} {m["rungs"][j][0][k]:+.3f}'
        for k in np.argsort(-np.abs(m['rungs'][j][0]))[:5]))

# --- the headline ----------------------------------------------------------
print('\n' + '=' * 79)
print(f'GAIN over the nominal design ({args.n_back} rungs, rule = {args.step}, '
      f'tested on the {args.eval})')
best, best_m = None, None
for m in METHODS:
    if m['accepted'] is None:
        print(f'  {m["short"]:7s} no step accepted')
        continue
    i = m['i']
    ref = H_ref[i]
    print(f'  {m["short"]:7s} surrogate {1 - H_sur[i] / H_sur[0]:+7.1%}'
          + (f'   simulator {1 - H_sim[i] / max(H_sim[0], 1e-30):+7.1%}' if args.sim
             else '')
          + f'   ||p|| = {np.linalg.norm(m["rungs"][m["accepted"]][0]):.3f}')
    if best is None or ref < best:
        best, best_m = ref, m
if best_m is not None:
    i = best_m['i']
    phi_best = PHI[i].numpy()
    np.save(os.path.join(args.out_dir, 'optimization_test_best_phi.npy'), phi_best)
    print(f'  best: {best_m["name"]}, saved its design to '
          f'{args.out_dir}/optimization_test_best_phi.npy')
else:
    print('  nothing to save: no step was accepted')

# =============================================================================
# Figure: the line search, as grad_test plots its walk
# =============================================================================
fig, axes = plt.subplots(1, len(METHODS), figsize=(5.2 * len(METHODS), 4.2),
                         constrained_layout=True, squeeze=False)
for a, m in zip(axes[0], METHODS):
    idx = np.array(m['idx'])
    x = np.array([np.linalg.norm(p) for p, _, _ in m['rungs']])
    a.axhline(H_sur[0], color=OK, lw=1, ls='--', label='surrogate at $\\phi_0$')
    a.plot(x, H_sur[idx], 'o-', color=OK, lw=2, ms=5, label='surrogate')
    a.fill_between(x, np.maximum(H_sur[idx] - pstd[idx], 0), H_sur[idx] + pstd[idx],
                   color=OK, alpha=0.25, lw=0)
    a.plot(x, [H_sur[0] - model_drop(p) for p, _, _ in m['rungs']], ':', color='0.4',
           lw=1.5, label='quadratic model')
    if args.sim:
        a.axhline(H_sim[0], color=BAD, lw=1, ls='--', label='simulator at $\\phi_0$')
        a.errorbar(x, H_sim[idx], yerr=serr[idx], fmt='s', ms=6, color=BAD, capsize=3,
                   lw=1.5, label=f'simulator (mean of {R})' if R > 1 else 'simulator')
    if m['accepted'] is not None:
        a.axvline(x[m['accepted']], color=GREY, lw=1.5, label='accepted')
    a.set(xscale='log', xlabel='$\\|p\\|$ (normalized units)',
          ylabel='hits' if GEN else 'weighted hits', title=m['name'])
    curves = [H_sur[idx]] + ([H_sim[idx]] if args.sim else [])
    if all(np.all(c > 0) for c in curves):
        a.set_yscale('log')
    a.legend(fontsize=8)
fig.suptitle(f'One step off the nominal design ({args.step}, tested on the {args.eval}, '
             f'{n:,} muons)')
fig.savefig(os.path.join(FIGS, 'optimization_test.png'), dpi=150)
plt.close(fig)

np.savez(os.path.join(args.out_dir, 'optimization_test.npz'),
         labels=np.array(labels), phi_0=phi_0.numpy(), muons=args.muons, n_muons=n,
         step=args.step, eval=args.eval, H_0=H_0, var_0=var_0, grad=g, hess=B,
         radii=radii, steps=U_all, phis=PHI.numpy(),
         method=np.array([m['short'] for m in METHODS]),
         idx=np.array([m['idx'] for m in METHODS]),
         accepted=np.array([-1 if m['accepted'] is None else m['accepted']
                            for m in METHODS]),
         pred=pred, pred_std=pstd, sim=sim, sim_mean=smean, sim_err=serr)
print(f'\nsaved {FIGS}/optimization_test.png and {args.out_dir}/optimization_test.npz')
