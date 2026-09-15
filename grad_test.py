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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', default='outputs/lcso_model.pt')
parser.add_argument('--muons', default='gen', choices=['gen', 'data'],
                    help="'gen' samples the generative flow (unweighted, what lcso.py "
                         "trains on); 'data' reads the real file (per-muon weights, which "
                         "every sum then carries)")
parser.add_argument('--temperature', type=float, default=1.0,
                    help='gen: flow temperature. 1 is the trained distribution; lcso.py '
                         'trains at 1.5 and tests at 1')
parser.add_argument('--seed', type=int, default=0,
                    help='gen: base seed for the muon draw. Each block is seeded from it '
                         'and its offset, so the gradient pass and the walk see the same '
                         'muons')
parser.add_argument('--pz_min', type=float, default=None,
                    help='gen: reject sampled muons below this p_z (GeV). Only needed for '
                         'tempered draws -- at temperature 1 the flow stays near the real '
                         'sample. Below the envelope the transport kernel handles, a muon '
                         'surfaces as a CUDA illegal memory access')
parser.add_argument('--pt_max', type=float, default=None,
                    help='gen: reject sampled muons above this p_t (default 14)')
parser.add_argument('--n_muons', type=int, default=0,
                    help='Muons for the curves (0 = all of the file for --muons data, '
                         '50M for --muons gen)')
parser.add_argument('--n_grad', type=int, default=0,
                    help='Muons for the gradient (0 = all, same sample as the curves). With '
                         '--muons gen a value that is not a whole number of --block draws a '
                         'different last block than the walk does, which only makes the '
                         'direction a slightly different estimate')
parser.add_argument('--n_points', type=int, default=7,
                    help='Points on the line (forced odd so the nominal design is one)')
parser.add_argument('--t_max', type=float, default=1.0, help='Half-length, normalized units')
parser.add_argument('--n_repeats', type=int, default=1,
                    help='Simulate each point this many times and report the mean and the '
                         'spread. The transport is stochastic (seed=None draws a new seed '
                         'per call), so repeats on the SAME muons give the MC transport '
                         'noise. Costs a full pass each')
parser.add_argument('--block', type=int, default=int(5e7), help='Muons held in memory')
parser.add_argument('--chunk', type=int, default=2 ** 21, help='Muon chunk for the surrogate')
parser.add_argument('--out_dir', default='outputs')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

DEVICE = torch.device(args.device)
os.makedirs(os.path.join(args.out_dir, 'figs'), exist_ok=True)
COLS = ['px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight']
GEN = args.muons == 'gen'
N_GEN_DEFAULT = int(1e8)

# Set on the class, before the instance exists: check_inputs and _sample_pz_pt read them
# off self, and rejecting out-of-envelope draws is cheaper than a dead CUDA context.
for attr, val in (('PZ_MIN', args.pz_min), ('PT_MAX', args.pt_max)):
    if val is not None:
        setattr(ShipMuonShieldCuda, attr, float(val))
        print(f'kinematic envelope: {attr} = {val}')

shield = ShipMuonShieldCuda(n_samples=1, uniform_fields=True)
phi_0 = shield.initial_phi                                   # PHYSICAL
model, _ = load_surrogate(args.model, phi_0)
model.eval().to(DEVICE)

if GEN:
    # Build the flow now, not lazily inside the first block: constructing it initializes
    # its weights from the global RNG, so a lazy load would consume draws that the second
    # pass (flow already built) does not, and the two passes would see different muons.
    shield._load_flow()


def read_muons(lo, hi):
    """Muons [lo, hi) as (raw, weights); weights is None when there are none.

    'data' reads the real file with the transverse origin reset, as _load_muons does, and
    keeps its weight column -- those weights are far from uniform (O(0.1) to O(1e3)), so
    dropping them would not give the physical rate. 'gen' draws from the flow, where one
    sample is one muon and the weights are all 1; the block is seeded from its offset so
    the gradient pass and the walk below get the very same muons.
    """
    if GEN:
        torch.manual_seed(args.seed + lo)
        return shield.sample_gen(n_samples=hi - lo, temperature=args.temperature), None
    with h5py.File(shield.muons_file, 'r') as f:
        x = np.stack([np.asarray(f[k][lo:hi], dtype=np.float32) for k in COLS], axis=1)
    x[:, 3], x[:, 4], x[:, 5] = 0.0, 0.0, -1.0
    x = torch.from_numpy(x)
    return x, x[:, 7]


if GEN:
    n = args.n_muons if args.n_muons > 0 else N_GEN_DEFAULT
    print(f'muons: generative flow at temperature {args.temperature:g}, seed {args.seed} '
          f'(unweighted)')
else:
    with h5py.File(shield.muons_file, 'r') as f:
        n = f['px'].shape[0]
    n = n if args.n_muons <= 0 else min(args.n_muons, n)
    print(f'muons: {shield.muons_file} (weighted)')

# --- gradient ---------------------------------------------------------------
# The model takes PHYSICAL phi and RAW muons, so grad_phi is dH/dphi in physical units.
# H = sum_x w p is a sum over muons, so its gradient is too: accumulating block gradients
# gives exactly the full-sample gradient, and neither the file nor the GPU has to hold all
# the muons at once. grad_phi chunks again inside each block for the autograd graph, and
# applies the same weights there (weights=None -> the plain sum, right for 'gen'). This is
# a separate pass from the walk below -- the direction has to be known before the points
# to simulate exist.
n_grad = n if args.n_grad <= 0 else min(args.n_grad, n)
phi_0_d = phi_0.to(DEVICE)
grad = torch.zeros(phi_0.numel())
t0 = time.time()
for lo in range(0, n_grad, args.block):
    g, gw = read_muons(lo, min(lo + args.block, n_grad))
    grad += model.grad_phi(phi_0_d, g[:, :7].to(DEVICE),
                           None if gw is None else gw.to(DEVICE)).cpu()
    del g, gw
    print(f'  gradient: {min(lo + args.block, n_grad):,}/{n_grad:,} muons '
          f'({time.time() - t0:.0f}s)')

# Physical -> normalized frame: dphi/du = (ub - lb)/2. Without this the components are in
# mixed units (cm, A) and their relative sizes are meaningless.
grad = grad * (model.upper_bound - model.lower_bound).cpu() / 2
direction = -grad / grad.norm()                              # descent: fewer hits at t > 0

labels = [f'M{m}:{shield.idx_mag[i].split("[")[0]}' for m, i in shield.params_idx.tolist()]
print(f'|grad| = {grad.norm():.4g} hits per normalized unit; descent direction:')
for k in torch.argsort(direction.abs(), descending=True)[:8]:
    print(f'  {labels[k]:22s} {direction[k]:+.3f}')

# --- the points to evaluate -------------------------------------------------
n_pts = args.n_points + (args.n_points + 1) % 2              # odd -> t = 0 on the grid
ts = np.linspace(-args.t_max, args.t_max, n_pts)
U = (torch.from_numpy(ts).float()[:, None] * direction[None, :]).clamp(-1, 1)
PHI = model.denormalize_phi(U.to(DEVICE)).cpu()              # back to PHYSICAL
R = max(1, args.n_repeats)
print(f'\n{n_pts} points x {n:,} muons x {R} repeats = {n_pts * n * R / 1e9:.1f}e9 tracks '
      f'to simulate')

# --- walk the line ----------------------------------------------------------
# Each repeat accumulates over every block, so row r is one independent full-sample
# estimate and the spread across rows is the transport noise on the total. Every sum is
# weighted by w, which is 1 for every muon when there are no weights -- so the same four
# accumulators are the plain count, the Bernoulli variance, the hit count and its
# Poisson variance in the unweighted case.
pred = np.zeros(n_pts)         # sum_x w p        expected weighted hits
pvar = np.zeros(n_pts)         # sum_x w^2 p(1-p) its Bernoulli spread
sim = np.zeros((R, n_pts))     # sum_x w hit      simulated weighted hits, per repeat
svar = np.zeros((R, n_pts))    # sum_x w^2 hit    its counting error, per repeat
n_mu = 0                       # muons actually seen, for the report
sumw = 0.0                     # sum_x w, the total the hits are a fraction of
PHI_D = PHI.to(DEVICE)
t0 = time.time()

for lo in range(0, n, args.block):
    raw, w = read_muons(lo, min(lo + args.block, n))
    n_mu += raw.shape[0]
    sumw += float(raw.shape[0] if w is None else w.double().sum())
    with torch.no_grad():
        for c in range(0, raw.shape[0], args.chunk):
            xb = raw[c:c + args.chunk, :7].to(DEVICE)
            pr = model.predict_proba(PHI_D, xb.unsqueeze(0))         # (n_pts, chunk)
            q = pr * (1 - pr)
            if w is not None:
                wb = w[c:c + args.chunk].to(DEVICE)
                pr, q = pr * wb, q * wb ** 2
            pred += pr.sum(1).double().cpu().numpy()
            pvar += q.sum(1).double().cpu().numpy()
    wd = None if w is None else w.double()
    for i in range(n_pts):
        for r in range(R):
            hits = shield(PHI[i], raw).double()      # a new transport seed every call
            sim[r, i] += float(hits.sum() if wd is None else (hits * wd).sum())
            svar[r, i] += float(hits.sum() if wd is None else (hits * wd ** 2).sum())
    print(f'  {min(lo + args.block, n):,}/{n:,} muons ({time.time() - t0:.0f}s)')
    del raw, w, wd

# --- report and plot --------------------------------------------------------
# Two different uncertainties: mc is the spread over repeats (transport noise, needs
# R > 1), count is sqrt(sum w^2 hit) from the finite muon sample, which every repeat
# shares and so does NOT shrink by repeating.
pstd = np.sqrt(pvar)
mean = sim.mean(0)
mc = sim.std(0, ddof=1) if R > 1 else np.zeros(n_pts)
count = np.sqrt(svar).mean(0)
err = mc if R > 1 else count                       # what goes on the plot
i0 = n_pts // 2
HITS = 'weighted hits' if not GEN else 'hits'
print(f'\n{n_mu:,} muons, total weight {sumw:,.1f}' + ('' if GEN else ' (weighted sums)'))
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
# sim keeps every repeat, so the spread can be re-analyzed without re-simulating
np.savez(os.path.join(args.out_dir, 'grad_test.npz'), t=ts, pred=pred, pred_std=pstd,
         sim=sim, sim_mean=mean, sim_mc_std=mc, sim_count_std=count,
         direction=direction.numpy(), phis=PHI.numpy(), muons=args.muons,
         temperature=args.temperature, n_muons=n_mu, sum_weights=sumw)
print(f'saved {out}')
