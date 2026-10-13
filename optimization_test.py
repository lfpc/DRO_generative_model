"""Take one optimization step off the nominal design and check the gain with the simulator.

Two steps from the same derivatives, both minimizing the expected hits H:
  steepest descent     p along -grad H
  trust-region Newton  p = argmin_{||p|| <= Delta} g.p + 0.5 p^T B p, solved exactly
                       (More-Sorensen; B is indefinite here)
--step picks how far: 'armijo' halves t until H(u+p) <= H(u) + c1 g.p; 'trust' halves the
radius until the actual drop is >= eta of the predicted one. --eval picks whether H in those
tests is the surrogate or the simulator; both are always reported. Designs are normalized so
the training box is |u_i| <= 1, so t0 = 1 is the largest ball inside it.

    python optimization_test.py
    python optimization_test.py --step trust --eval simulator
    python optimization_test.py --muons data --n_repeats 3
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
parser.add_argument('--step', default='armijo', choices=['armijo', 'trust'], help='step rule')
parser.add_argument('--eval', default='surrogate', choices=['surrogate', 'simulator'],
                    help='what the step test measures H with')
parser.add_argument('--t0', type=float, default=1.0,
                    help='first step length / trust radius, normalized units')
parser.add_argument('--n_back', type=int, default=6, help='rungs in the backtracking ladder')
parser.add_argument('--n_repeats', type=int, default=1, help='simulator runs per candidate')
parser.add_argument('--no_sim', dest='sim', action='store_false',
                    help='skip the simulator (forces --eval surrogate)')
parser.add_argument('--block', type=int, default=int(2e7), help='muons held in memory')
parser.add_argument('--chunk', type=int, default=2 ** 19, help='muon chunk for the surrogate')
parser.add_argument('--out_dir', default='outputs', help='output directory')
parser.add_argument('--device', default='cuda', help='torch device')
args = parser.parse_args()

BETA, C1, ETA = 0.5, 1e-4, 0.25     # backtracking factor, Armijo constant, trust threshold
if not args.sim:
    args.eval = 'surrogate'
# TF32 would form the p x p accumulators at 10 mantissa bits: too few for the curvature.
torch.backends.cuda.matmul.allow_tf32 = False

DEVICE = torch.device(args.device)
FIGS = os.path.join(args.out_dir, 'figs')
os.makedirs(FIGS, exist_ok=True)
GEN = args.muons == 'gen'
OK, BAD, GREY = '#0072B2', '#D55E00', '0.6'

shield = ShipMuonShieldCuda(n_samples=1, uniform_fields=True)
phi_0 = shield.initial_phi
model, cfg = load_surrogate(args.model, phi_0)
model.eval().to(DEVICE)
D = model.dim
labels = param_labels(shield)
muons = Muons(shield, args.muons, args.seed, args.temperature)
n = count_muons(shield, args.n_muons, GEN)
print(f'muons: {n:,} from {muons.describe()}')
print(f'model: {os.path.basename(args.model)} {cfg["model_type"]}, p = {cfg["p"]}, D = {D}')
print(f'step rule: {args.step}, tested on the {args.eval}, from t0 = {args.t0:g}')

# =============================================================================
# Sweep 1: the derivatives at the nominal design
# =============================================================================
# The trunk t_x does not depend on the design, so with s = p(1-p) the accumulators
#   T = sum_x w s t_x,   A = sum_x w s (1 - 2p) t_x t_x^T
# give grad = J^T T and hess = J^T A J + d2(b.T)/du2, with J = db/du of the branch net.
u_0 = torch.zeros(D, device=DEVICE)
b_0 = model.branch_net(u_0.unsqueeze(0)).squeeze(0).detach()
P = b_0.numel()
T_acc = torch.zeros(P, dtype=torch.float64, device=DEVICE)
A_acc = torch.zeros(P, P, dtype=torch.float64, device=DEVICE)
H_0, var_0, sumw = 0.0, 0.0, 0.0
t0 = time.time()
for lo in range(0, n, args.block):
    hi = min(lo + args.block, n)
    raw, w_mu = muons.read(lo, hi)
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
    print(f'  sweep 1: {hi:,}/{n:,} muons ({time.time() - t0:.0f}s)')
    del raw, w_mu

J = torch.autograd.functional.jacobian(
    lambda u: model.branch_net(u.unsqueeze(0)).squeeze(0), u_0).detach()
Hb = torch.autograd.functional.hessian(
    lambda u: model.branch_net(u.unsqueeze(0)).squeeze(0) @ T_acc.float(), u_0).detach()
Jd = J.double()
g = (Jd.T @ T_acc).cpu().numpy()                          # dH/du
B = (Jd.T @ A_acc @ Jd + Hb.double()).cpu().numpy()
B = 0.5 * (B + B.T)

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
    """argmin of g.p + 0.5 p^T B p over ||p|| <= delta (More-Sorensen): (B + lam I) p = -g
    with lam >= max(0, -min eig B), and ||p(lam)|| is monotone in lam, so bisect on lam."""
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


radii = args.t0 * BETA ** np.arange(args.n_back)    # longest first: backtracking
sd = [(-r * g_hat, r, 'descent') for r in radii]
tr = []
for r in radii:
    p, _, kind = trs_min(g, B, r)
    tr.append((p, r, kind))
METHODS = [{'name': 'steepest descent', 'short': 'grad', 'rungs': sd},
           {'name': 'trust-region Newton', 'short': 'newton', 'rungs': tr}]

# All candidates of both methods go into one batch: a single sweep over the muons.
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
n_des, R = len(U_list), args.n_repeats
print(f'\nsweep 2: {n_des} designs ({len(METHODS)} methods x {len(radii)} rungs, plus the '
      f'nominal) over {n:,} muons' + (f' x {R} repeats' if args.sim else ' (surrogate '
      f'only)'))

# =============================================================================
# Sweep 2: evaluate every candidate, surrogate and simulator, on the same muons
# =============================================================================
pred, pvar, sim, svar, _ = evaluate_designs(model, shield, muons, PHI, n, args.block,
                                            args.chunk, R, args.sim, DEVICE)
pstd = np.sqrt(pvar)
smean = sim.mean(0)
serr = sim.std(0, ddof=1) if R > 1 else np.sqrt(svar).mean(0)
H_sur, H_sim = pred, smean
H_ref = H_sim if args.eval == 'simulator' else H_sur      # what the step test believes

# Derivative check: the shortest step's drop should match the quadratic model.
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
