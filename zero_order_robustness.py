"""Scan one shield parameter around its nominal value with the simulator (optionally the
surrogate too, on the same muons).

    python zero_order_robustness.py --magnet 2 --param 1
    python zero_order_robustness.py --magnet 2 --param 1 --surrogate_model outputs/lcso_model.pt
"""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import load_surrogate
from problems import ShipMuonShieldCuda


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter,
                   argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument('--magnet', type=int, required=True, help='magnet index (row of phi)')
parser.add_argument('--param', type=int, required=True, choices=range(15),
                    help='parameter index within the magnet (see ShipMuonShieldCuda.idx_mag)')
parser.add_argument('--rel_range', type=float, default=0.2,
                    help='scan +- this fraction of the nominal value')
parser.add_argument('--n_points', type=int, default=21, help='scan points')
parser.add_argument('--n_samples', type=int, default=0, help='muons (0 = whole file)')
parser.add_argument('--n_repeats', type=int, default=1,
                    help='simulator runs per point, on the same muons (transport noise)')
parser.add_argument('--surrogate_model', default=None,
                    help='LCSO checkpoint to draw on the same scan')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

FIGS_DIR = os.path.join('outputs', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)
torch.manual_seed(args.seed)

muon_shield = ShipMuonShieldCuda(uniform_fields=True, n_samples=args.n_samples,
                                 seed=args.seed)
phi_base = muon_shield.DEFAULT_PHI.clone()
magnet = args.magnet % muon_shield.n_magnets                 # allow -1 for the last one
nominal = phi_base[magnet, args.param].item()
param_name = muon_shield.idx_mag[args.param]
values = np.linspace(nominal * (1 - args.rel_range), nominal * (1 + args.rel_range), args.n_points)

model = None
if args.surrogate_model:
    # The surrogate takes the PHYSICAL free parameters (the params_idx subset of phi).
    # Caveat: add_fixed_params ties some columns (e.g. XmgapOut = XmgapIn) in its training
    # designs, while this scan moves the one column alone.
    j = muon_shield.params_idx.tolist().index([magnet, args.param])
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phi_0 = muon_shield.initial_phi
    model, cfg = load_surrogate(args.surrogate_model, phi_0)
    model.eval().to(DEV)
    PHI = phi_0.repeat(args.n_points, 1)
    PHI[:, j] = torch.from_numpy(values).float()
    print(f'surrogate {os.path.basename(args.surrogate_model)} ({cfg["model_type"]}, '
          f'p = {cfg["p"]}) on {DEV}; this scan is its parameter {j}')

muons = muon_shield.sample_x()
weight = muons[:, 7]

R = args.n_repeats
print(f"Scanning magnet {magnet}, '{param_name}' (index {args.param}): "
      f"nominal={nominal:.4f}, range=[{values[0]:.4f}, {values[-1]:.4f}] over "
      f"{args.n_points} points, "
      f"{muons.shape[0]} muons, {R} repeat(s)")

hits_weighted = np.empty((R, args.n_points), dtype=np.float64)
for k, v in enumerate(values):
    phi = phi_base.clone()
    phi[magnet, args.param] = float(v)
    for r in range(R):
        # The shield has a fixed seed, so give each repeat its own to make them independent.
        muon_shield.seed = args.seed + r
        hits = muon_shield(phi, muons)
        hits_weighted[r, k] = (hits.double() * weight.double()).sum().item()
    m, s = hits_weighted[:, k].mean(), hits_weighted[:, k].std(ddof=1) if R > 1 else 0.0
    print(f'[{k + 1}/{args.n_points}] {param_name} = {v:.4f}  ->  weighted hits = '
          f'{m:.2f}' + (f' +- {s:.2f} ({s / max(m, 1e-12):.2%})' if R > 1 else ''))

mean = hits_weighted.mean(0)
std = hits_weighted.std(0, ddof=1) if R > 1 else np.zeros(args.n_points)
if R > 1:
    print(f'transport noise over the scan: {np.median(std / np.maximum(mean, 1e-12)):.2%} '
          f'of the hits (median over the points)')

# --- the same scan through the surrogate: sum w p, band sqrt(sum w^2 p(1-p)) ----------
pred = pred_std = None
if model is not None:
    CHUNK = 2 ** 20
    pred, pvar = np.zeros(args.n_points), np.zeros(args.n_points)
    PHI_D = PHI.to(DEV)
    with torch.no_grad():
        for s in range(0, muons.shape[0], CHUNK):
            xb = muons[s:s + CHUNK, :7].to(DEV)
            wb = weight[s:s + CHUNK].to(DEV)
            pr = model.predict_proba(PHI_D, xb.unsqueeze(0))         # (n_points, chunk)
            pred += (pr * wb).sum(1).double().cpu().numpy()
            pvar += (pr * (1 - pr) * wb ** 2).sum(1).double().cpu().numpy()
    pred_std = np.sqrt(pvar)
    print(f'\nsurrogate {os.path.basename(args.surrogate_model)} on the same muons:')
    print(f'{param_name:>16s} {"simulated":>14s} {"+- std":>12s} {"surrogate":>14s} '
          f'{"+- std":>12s} {"ratio":>7s}')
    for k, v in enumerate(values):
        print(f'{v:16.4f} {mean[k]:14,.1f} {std[k]:12,.1f} {pred[k]:14,.1f} '
              f'{pred_std[k]:12,.1f} {pred[k] / max(mean[k], 1e-12):7.3f}')

SIM, SUR = '#D55E00', '#0072B2'
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
ax.plot(values, mean, marker='o', color=SIM,
        label=f'simulator (mean of {R})' if R > 1 else 'simulator')
if R > 1:
    ax.fill_between(values, mean - std, mean + std, color=SIM, alpha=0.25, lw=0)
if pred is not None:
    ax.plot(values, pred, marker='s', ms=4, color=SUR, label='surrogate')
    ax.fill_between(values, pred - pred_std, pred + pred_std, color=SUR, alpha=0.25,
                    lw=0)
ax.axvline(nominal, color='k', linestyle='--', linewidth=1, label='nominal')
if pred is not None and np.all(mean > 0) and np.all(pred > 0):
    ax.set_yscale('log')
ax.set_xlabel(f'Magnet {magnet} -- {param_name}')
ax.set_ylabel('Weighted number of hits')
ax.set_title(f'Zero-order robustness scan (magnet {magnet}, {param_name})')
ax.legend()
fig_path = os.path.join(FIGS_DIR, 'zero_order_robustness.png')
fig.savefig(fig_path, dpi=150)
print(f'Saved plot to {fig_path}')
