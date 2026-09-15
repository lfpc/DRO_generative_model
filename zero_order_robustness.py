import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from problems import ShipMuonShieldCuda

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--magnet', type=int, required=True,
                    help='Index m (0-based) of the magnet to vary, i.e. the row of phi')
parser.add_argument('--param', type=int, required=True, choices=range(15),
                    help='Index i (0-14) of the parameter to vary within the magnet '
                         '(see ShipMuonShieldCuda.idx_mag for the name of each index)')
parser.add_argument('--rel_range', type=float, default=0.2,
                    help='Fractional range to scan around the nominal value (default +-20%%)')
parser.add_argument('--n_points', type=int, default=21,
                    help='Number of scan points (odd keeps the nominal value on the grid)')
parser.add_argument('--n_samples', type=int, default=0,
                    help='Number of muons to simulate (0 = full sample file, or 50e6 if '
                         '--uniform_muons)')
parser.add_argument('--n_repeats', type=int, default=1,
                    help='Simulate every scan point this many times and report the mean '
                         'and the spread. All repeats run on the SAME muons, so the '
                         'spread is the transport noise alone and not the finite muon '
                         'sample, which is common to all of them. Costs a full scan each')
parser.add_argument('--surrogate_model', default=None,
                    help='Checkpoint of an LCSO surrogate trained as in lcso.py. When '
                         'given, its prediction is drawn on the same scan points and the '
                         'same muons: sum_x w_x p_x, with the Bernoulli spread '
                         'sqrt(sum_x w_x^2 p_x(1-p_x)) as its band. None skips it')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

FIGS_DIR = os.path.join('outputs', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)
out_path = os.path.join('outputs', 'zero_order_simulations.npz')

torch.manual_seed(args.seed)

muon_shield = ShipMuonShieldCuda(uniform_fields=True, n_samples=args.n_samples,
                                 seed=args.seed)

phi_base = muon_shield.DEFAULT_PHI.clone()
if not -muon_shield.n_magnets <= args.magnet < muon_shield.n_magnets:
    raise ValueError(f'--magnet must be in [-{muon_shield.n_magnets}, '
                     f'{muon_shield.n_magnets - 1}], got {args.magnet}')
# Negative indices work on phi_base, but the surrogate's parametrization is listed with
# non-negative rows, so resolve -1 to the last magnet once and use that everywhere.
magnet = args.magnet % muon_shield.n_magnets
nominal = phi_base[magnet, args.param].item()
param_name = muon_shield.idx_mag[args.param]

if nominal == 0.0:
    raise ValueError(f"Nominal value of '{param_name}' for magnet {magnet} is 0; "
                     f"a relative +-{args.rel_range:.0%} scan is undefined around zero.")

values = np.linspace(nominal * (1 - args.rel_range), nominal * (1 + args.rel_range), args.n_points)

# --- the surrogate, resolved before anything expensive happens --------------
# The model takes the PHYSICAL free parameters (the params_idx subset of phi), so a
# parameter outside that subset cannot move its input at all. Checking here, rather than
# after the scan, means a mismatch costs nothing instead of a full set of simulations.
model = PHI = cfg = None
if args.surrogate_model:
    from models import load_surrogate                  # pulls botorch: only when needed

    free = muon_shield.params_idx.tolist()
    where = [i for i, (m, p) in enumerate(free) if (m, p) == (magnet, args.param)]
    if not where:
        have = sorted({m for m, p in free if p == args.param})
        raise SystemExit(
            f"(magnet {magnet}, '{param_name}') is not one of the {len(free)} parameters "
            f"the surrogate is defined over, so it cannot see this scan.\n"
            + (f"'{param_name}' is free for magnet(s) {have}."
               if have else f"'{param_name}' is free for no magnet: the simulator reads "
                            f"it off the fixed columns of phi.")
            + "\nDrop --surrogate_model to run the simulator scan on its own.")
    # add_fixed_params ties a few columns together when the simulator is called with the
    # free vector, which is how the training designs were built. This scan moves the full
    # matrix instead, so for a tied parameter the two curves answer slightly different
    # questions.
    tied = {12: 13}
    if magnet <= 3:
        tied.update({2: 3, 6: 7, 8: 9})
    if magnet == 0:
        tied[4] = 5
    if args.param in tied:
        print(f"WARNING: the surrogate was trained on designs where '{param_name}' moves "
              f"together with '{muon_shield.idx_mag[tied[args.param]]}' "
              f"(add_fixed_params), while this scan moves it alone.")
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phi_0 = muon_shield.initial_phi                    # PHYSICAL, the free parameters
    model, cfg = load_surrogate(args.surrogate_model, phi_0)
    model.eval().to(DEV)
    PHI = phi_0.repeat(args.n_points, 1)
    PHI[:, where[0]] = torch.from_numpy(values).float()
    print(f'surrogate {os.path.basename(args.surrogate_model)} ({cfg["model_type"]}, '
          f'p = {cfg["p"]}) loaded on {DEV}; it sees this scan as its parameter '
          f'{where[0]} of {len(free)}')

muons = muon_shield.sample_x()
weight = muons[:, 7] if muons.shape[-1] > 7 else torch.ones(muons.shape[0])

R = max(1, args.n_repeats)
print(f"Scanning magnet {magnet}, '{param_name}' (index {args.param}): "
      f"nominal={nominal:.4f}, range=[{values[0]:.4f}, {values[-1]:.4f}] over {args.n_points} points, "
      f"{muons.shape[0]} muons, {R} repeat(s)")

hits_weighted = np.empty((R, args.n_points), dtype=np.float64)
for k, v in enumerate(values):
    phi = phi_base.clone()
    phi[magnet, args.param] = float(v)
    for r in range(R):
        # The transport is stochastic, but the shield was built with a fixed seed, so
        # repeating a call verbatim would return the very same hits. Each repeat gets its
        # own seed instead: the run stays reproducible and the repeats stay independent.
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

# --- the same scan through the surrogate ------------------------------------
# Its prediction is sum_x w_x p_x over the very same muons, and its band is the Bernoulli
# spread of that sum, sqrt(sum_x w_x^2 p_x (1-p_x)) -- the width of one simulator draw
# around the prediction, which is what makes it comparable to the curve above.
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
    ax.set_yscale('log')      # the two can sit decades apart; a linear axis hides one
ax.set_xlabel(f'Magnet {magnet} -- {param_name}')
ax.set_ylabel('Weighted number of hits')
ax.set_title(f'Zero-order robustness scan (magnet {magnet}, {param_name})')
ax.legend()
fig_path = os.path.join(FIGS_DIR, 'zero_order_robustness.png')
fig.savefig(fig_path, dpi=150)
print(f'Saved plot to {fig_path}')
