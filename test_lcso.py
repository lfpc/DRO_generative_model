"""Evaluate a trained LCSO surrogate against the stored simulation data.

Loads the checkpoint lcso.py wrote, loads the training and test designs, and plots
predicted vs. true hits for each.

    python test_lcso.py                       # everything
    python test_lcso.py --n_muons 1000000     # quick pass
"""
import argparse
import os

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
from models import load_surrogate
from problems import ShipMuonShieldCuda


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='outputs/lcso_model.pt')
parser.add_argument('--train_data', default='outputs/training_data.h5')
parser.add_argument('--test_data', default='outputs/test_data.h5')
parser.add_argument('--n_phi', type=int, default=0, help='Designs per file (0 = all)')
parser.add_argument('--n_muons', type=int, default=0, help='Muons per design (0 = all)')
parser.add_argument('--batch', type=int, default=2 ** 19, help='Muon batch for prediction')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

DEVICE = torch.device(args.device)
FIGS = 'outputs/figs'
os.makedirs(FIGS, exist_ok=True)

# --- model -----------------------------------------------------------------
phi_0 = ShipMuonShieldCuda(n_samples=1).initial_phi
model, cfg = load_surrogate(args.model, phi_0)

model.eval().to(DEVICE)
print(f'{args.model}: {cfg["model_type"]}, {sum(p.numel() for p in model.parameters()):,} params')

# The model carries its own muon/phi normalization (see models.NormalizedIO), so the
# designs and muons below are handed over exactly as they are stored: physical and raw.


@torch.no_grad()
def evaluate(path, label):
    """True and predicted hits per design. Muons are read one design at a time: the test
    store holds 30 x 5e7 x 7 floats (42 GB) and will not fit in memory at once."""
    rows = []
    with h5py.File(path, 'r') as f:
        phis = torch.from_numpy(f['phis'][:])                    # physical, as the model wants
        n_phi = phis.shape[0] if args.n_phi <= 0 else min(args.n_phi, phis.shape[0])
        n_mu = f['muons'].shape[1] if args.n_muons <= 0 else min(args.n_muons, f['muons'].shape[1])
        print(f'\n{label}: {n_phi} designs x {n_mu:,} muons  ({path})')
        for i in range(n_phi):
            true = int(np.asarray(f['hits'][i, :n_mu]).sum())
            muons = np.asarray(f['muons'][i, :n_mu])
            phi = phis[i].to(DEVICE)
            pred = 0.0
            for s in range(0, n_mu, args.batch):
                x = torch.from_numpy(muons[s:s + args.batch]).to(DEVICE)
                pred += float(model.predict_proba(phi[None], x[None]).sum())
            del muons
            dist = float(utils.compute_distance(phis[i], phi_0, norm='l2'))
            rows.append((true, pred, dist))
            print(f'  {i + 1:3d}/{n_phi}  true {true:>9,}  pred {pred:>11,.1f}  '
                  f'ratio {pred / max(true, 1):>7.2f}  |phi-phi_0| {dist:6.2f}')
    return rows


def plot(rows, name, title):
    t = np.array([r[0] for r in rows], float)
    p = np.array([r[1] for r in rows], float)
    d = np.array([r[2] for r in rows], float)
    ok = (t > 0) & (p > 0)
    if not ok.any():
        print(f'  nothing to plot for {name} (no design has both true and predicted hits)')
        return
    fig, ax = plt.subplots(figsize=(6, 5.2), constrained_layout=True)
    sc = ax.scatter(t[ok], p[ok], c=d[ok], cmap='viridis', edgecolors='k', linewidths=0.5)
    lim = [min(t[ok].min(), p[ok].min()), max(t[ok].max(), p[ok].max())]
    ax.plot(lim, lim, 'k--', label='identity')
    fig.colorbar(sc, ax=ax, label='distance from phi_0')
    ax.set(xscale='log', yscale='log', xlabel='True hits', ylabel='Predicted hits', title=title)
    ax.legend()
    out = os.path.join(FIGS, name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')
    if (~ok).any():
        print(f'  ({int((~ok).sum())} designs omitted: zero hits, undefined on log axes)')


for path, label, name, title in (
        (args.train_data, 'train', 'test_lcso_train.png', 'Training designs (in-sample)'),
        (args.test_data, 'test', 'test_lcso_test.png', 'Test designs')):
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        print(f'\n{label}: {path} missing or empty, skipped')
        continue
    plot(evaluate(path, label), name, title)
