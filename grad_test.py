"""Walk the surrogate's descent direction at the nominal design and compare with the simulator.

The gradient of the expected hits H = sum_x w p is taken in normalized design units; the
surrogate (with its Bernoulli spread) and the simulator are then evaluated along the line
on the same muons.

    python grad_test.py
    python grad_test.py --muons data --n_repeats 3
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
parser.add_argument('--seed', type=int, default=0, help='gen: base seed of the muon draw')
parser.add_argument('--pz_min', type=float, default=ShipMuonShieldCuda.PZ_MIN,
                    help='gen: reject muons below this p_z (GeV)')
parser.add_argument('--pt_max', type=float, default=ShipMuonShieldCuda.PT_MAX,
                    help='gen: reject muons above this p_t (GeV)')
parser.add_argument('--n_muons', type=int, default=0,
                    help='muons for the walk (0 = whole file for data, 1e8 for gen)')
parser.add_argument('--n_grad', type=int, default=0,
                    help='muons for the gradient (0 = same as the walk)')
parser.add_argument('--n_points', type=int, default=7, help='points on the line (made odd)')
parser.add_argument('--t_max', type=float, default=1.0, help='half-length, normalized units')
parser.add_argument('--n_repeats', type=int, default=1, help='simulator runs per point')
parser.add_argument('--block', type=int, default=int(5e7), help='muons held in memory')
parser.add_argument('--chunk', type=int, default=2 ** 21, help='muon chunk for the surrogate')
parser.add_argument('--out_dir', default='outputs', help='output directory')
parser.add_argument('--device', default='cuda', help='torch device')
args = parser.parse_args()

DEVICE = torch.device(args.device)
os.makedirs(os.path.join(args.out_dir, 'figs'), exist_ok=True)
GEN = args.muons == 'gen'

# Below this envelope a muon crashes the CUDA transport; matters for tempered draws.
ShipMuonShieldCuda.PZ_MIN = args.pz_min
ShipMuonShieldCuda.PT_MAX = args.pt_max

shield = ShipMuonShieldCuda(n_samples=1, uniform_fields=True)
phi_0 = shield.initial_phi                                   # PHYSICAL
model, _ = load_surrogate(args.model, phi_0)
model.eval().to(DEVICE)
muons = Muons(shield, args.muons, args.seed, args.temperature)
n = int(1e8) if GEN and args.n_muons <= 0 else count_muons(shield, args.n_muons, GEN)
print(f'muons: {n:,} from {muons.describe()}, seed {args.seed}')

# --- gradient ---------------------------------------------------------------
# H is a sum over muons, so block gradients add up to the full-sample gradient. In
# physical units (the model takes physical phi).
n_grad = n if args.n_grad <= 0 else min(args.n_grad, n)
phi_0_d = phi_0.to(DEVICE)
grad = torch.zeros(phi_0.numel())
t0 = time.time()
for lo in range(0, n_grad, args.block):
    hi = min(lo + args.block, n_grad)
    raw, w = muons.read(lo, hi)
    grad += model.grad_phi(phi_0_d, raw[:, :7].to(DEVICE),
                           None if w is None else w.to(DEVICE)).cpu()
    del raw, w
    print(f'  gradient: {hi:,}/{n_grad:,} muons ({time.time() - t0:.0f}s)')

# To normalized units, dphi/du = (ub - lb)/2, so the components are comparable.
grad = grad * (model.upper_bound - model.lower_bound).cpu() / 2
direction = -grad / grad.norm()                              # descent: fewer hits at t > 0

labels = param_labels(shield)
print(f'|grad| = {grad.norm():.4g} hits per normalized unit; descent direction:')
for k in torch.argsort(direction.abs(), descending=True)[:8]:
    print(f'  {labels[k]:22s} {direction[k]:+.3f}')

# --- walk the line ----------------------------------------------------------
n_pts = args.n_points + (args.n_points + 1) % 2              # odd -> t = 0 on the grid
ts = np.linspace(-args.t_max, args.t_max, n_pts)
U = (torch.from_numpy(ts).float()[:, None] * direction[None, :]).clamp(-1, 1)
PHI = model.denormalize_phi(U.to(DEVICE)).cpu()              # back to PHYSICAL
R = args.n_repeats
print(f'\n{n_pts} points x {n:,} muons x {R} repeats = {n_pts * n * R / 1e9:.1f}e9 tracks '
      f'to simulate')
pred, pvar, sim, svar, sumw = evaluate_designs(model, shield, muons, PHI, n, args.block,
                                               args.chunk, R, True, DEVICE, label='walk')

# --- report and plot --------------------------------------------------------
# mc: spread over repeats (transport noise). count: sqrt(sum w^2 hit), the finite muon
# sample, shared by every repeat so it does not shrink by repeating.
pstd = np.sqrt(pvar)
mean = sim.mean(0)
mc = sim.std(0, ddof=1) if R > 1 else np.zeros(n_pts)
count = np.sqrt(svar).mean(0)
err = mc if R > 1 else count                       # what goes on the plot
i0 = n_pts // 2
HITS = 'hits' if GEN else 'weighted hits'
print(f'\n{n:,} muons, total weight {sumw:,.1f}' + ('' if GEN else ' (weighted sums)'))
print(f'\n{"t":>8s} {"surrogate":>15s} {"+- std":>11s} {"simulator":>15s} {"+- mc":>10s} '
      f'{"+- count":>10s} {"ratio":>8s}')
for i in range(n_pts):
    print(f'{ts[i]:+8.3f} {pred[i]:15,.1f} {pstd[i]:11,.1f} {mean[i]:15,.1f} {mc[i]:10,.1f} '
          f'{count[i]:10,.1f} {pred[i] / max(mean[i], 1e-12):8.2f}')
print(f'\nsimulated change along the direction: {mean[-1] / max(mean[i0], 1e-12):.3f}x at '
      f't=+{args.t_max:g}, {mean[0] / max(mean[i0], 1e-12):.3f}x at t=-{args.t_max:g}')
if R > 1:
    print(f'transport noise is {np.median(mc / np.maximum(count, 1e-12)):.2f}x the counting '
          f'error (median over the points)')

fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
ax.plot(ts, pred, color='#0072B2', lw=2, marker='o', ms=4, label='surrogate')
ax.fill_between(ts, pred - pstd, pred + pstd, color='#0072B2', alpha=0.25, lw=0,
                label='surrogate $\\pm$ std')
ax.errorbar(ts, mean, yerr=err, fmt='s', ms=6, color='#D55E00', capsize=3, lw=1.5,
            label=f'simulator (mean of {R})' if R > 1 else 'simulator')
if R > 1:
    ax.plot(np.repeat(ts[None], R, 0), sim, '.', ms=3, color='#D55E00', alpha=0.35,
            zorder=0, label='_')
ax.axvline(0, color='0.6', lw=1, ls='--')
ax.set_xlabel('step along $-\\nabla H/\\|\\nabla H\\|$ (normalized units)')
ax.set_ylabel(HITS)
ax.set_title('Surrogate vs. simulator along the gradient\n'
             + (f'generative muons, T = {args.temperature:g}' if GEN
                else 'real muons, weighted'))
if np.all(pred > 0) and np.all(mean > 0):
    ax.set_yscale('log')
ax.legend()
out = os.path.join(args.out_dir, 'figs', 'grad_test.png')
fig.savefig(out, dpi=150)
np.savez(os.path.join(args.out_dir, 'grad_test.npz'), t=ts, pred=pred, pred_std=pstd,
         sim=sim, sim_mean=mean, sim_mc_std=mc, sim_count_std=count,
         direction=direction.numpy(), phis=PHI.numpy(), muons=args.muons,
         temperature=args.temperature, n_muons=n, sum_weights=sumw)
print(f'saved {out}')
