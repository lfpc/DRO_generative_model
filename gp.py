import torch
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from problems import ShipMuonShieldCuda
import sys
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, ROOT)

from trainer import fit_gp
from models import GaussianProcess
import utils

DEVICE = torch.device("cuda")
OUT   = "outputs/"
DTYPE = torch.float64



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fields_map', dest='uniform_fields', action='store_false', help='Use uniform magnetic fields')
parser.add_argument('--load_data', action='store_true', help='Save simulation data')
parser.add_argument('--load_model', action='store_true', help='Load model from file')
parser.add_argument('--uniform_muons', action='store_true', help='Use uniform muon distribution')
parser.add_argument('--n_muons', type=int, default=50e6, help='Number of muons to simulate')
parser.add_argument('--n_phi_training', type=int, default=50, help='Number of phi samples to use for training')
parser.add_argument('--n_phi_test', type=int, default=10, help='Number of phi samples to use for testing')
args = parser.parse_args()

DEVICE = torch.device("cuda")
OUT_DIR = 'outputs'
FIGS_DIR = os.path.join(OUT_DIR, 'figs')

n_phi_training = int(args.n_phi_training)

n_muons = int(args.n_muons) if args.n_muons else muon_shield.muons.size(0) 
muon_shield = ShipMuonShieldCuda(uniform_fields=args.uniform_fields, n_samples=n_muons)
phi_0 = muon_shield.initial_phi
phi_dim = phi_0.shape[-1]

delta = 0.1
lower_bound = phi_0 - delta * phi_0
upper_bound = phi_0 + delta * phi_0

h5_path = os.path.join(OUT_DIR, 'training_data_gp.h5')
muons = muon_shield.sample_x()
weight = muons[..., 7]

if args.load_data:
    with h5py.File(h5_path, 'r') as h5f:
        phis_ds = h5f['phis']
        hits_ds = h5f['hits']
        hits = torch.from_numpy(hits_ds[:])
        phis_training = torch.from_numpy(phis_ds[:])
        print(f"Loaded phis {phis_ds.shape}, hits {hits_ds.shape} from {h5_path}")
else:
    with h5py.File(h5_path, 'w') as h5f:
        phis_training = utils.sample_phi(lower_bound, upper_bound, dim=phi_dim, n_samples=args.n_phi_training).to('cpu')
        n_phi_training = phis_training.shape[0]
        phis_ds = h5f.create_dataset('phis', shape=(n_phi_training, phi_dim), dtype=np.float32)
        hits_ds = h5f.create_dataset('hits', shape=(n_phi_training,), dtype=np.float64)

        for i, phi in enumerate(phis_training):
            print(f"\n Simulating for phi {i+1}/{n_phi_training}")

            hits = (muon_shield(phi.float(), muons).to(DTYPE) * weight).sum()

            phis_ds[i] = phi.numpy().astype(np.float32, copy=False)
            hits_ds[i] = hits.item()

        print(f"Saved phis {phis_ds.shape}, hits {hits_ds.shape} to {h5_path}")

        hits = torch.from_numpy(hits_ds[:])

hits = hits.view(-1, 1).to(DTYPE)  # (n_phi_training, 1)
phis_training = phis_training.to(DTYPE)  # (n_phi_training, phi_dim)
model = GaussianProcess(phis_training, hits)
fit_gp(model)


print("Testing the model on new phi samples...")
phis_testing = utils.sample_phi(lower_bound, upper_bound, dim=phi_dim, n_samples=args.n_phi_test).cpu().to(DTYPE)


PRED_BATCH_SIZE = 2**18

model.eval()
with torch.no_grad():
    hits_test = []
    for phi in phis_testing:
        hits_test.append((muon_shield(phi.float(), muons).to(DTYPE) * weight).sum().item())

    # batch_shape = (n,), q = 1 -> independent posteriors, no n x n joint covariance
    post_test = model.posterior(phis_testing.unsqueeze(-2))   # U_new in scaled units
    preds_test = post_test.mean.squeeze(-1).squeeze(-1).tolist()   # in hits
    stds_test = post_test.variance.squeeze(-1).squeeze(-1).sqrt().tolist()
    dists_test = utils.compute_distance(phis_testing, phi_0, norm='l2').tolist()

    hits_train = hits.view(-1).tolist()
    post_train = model.posterior(phis_training.unsqueeze(-2))  # U_new in scaled units
    preds_train = post_train.mean.squeeze(-1).squeeze(-1).tolist()   # in hits
    stds_train = post_train.variance.squeeze(-1).squeeze(-1).sqrt().tolist()
    dists_train = utils.compute_distance(phis_training, phi_0, norm='l2').tolist()



from matplotlib.lines import Line2D

dists_all = dists_train + dists_test
norm = plt.Normalize(vmin=min(dists_all), vmax=max(dists_all))
cmap = plt.cm.viridis

fig, ax = plt.subplots()

ax.errorbar(hits_train[1:], preds_train[1:], yerr=stds_train[1:], fmt='none',
            ecolor='gray', capsize=3, zorder=1)
ax.errorbar(hits_test, preds_test, yerr=stds_test, fmt='none',
            ecolor='gray', capsize=3, zorder=1)
ax.errorbar(hits_train[0], preds_train[0], yerr=stds_train[0], fmt='none',
            ecolor='gray', capsize=3, zorder=1)

ax.scatter(hits_train[1:], preds_train[1:], c=dists_train[1:], cmap=cmap, norm=norm,
           marker='s', edgecolors='k', zorder=2)
ax.scatter(hits_test, preds_test, c=dists_test, cmap=cmap, norm=norm,
           marker='o', edgecolors='k', zorder=2)
ax.scatter(hits_train[0], preds_train[0], c=[dists_train[0]], cmap=cmap, norm=norm,
           marker='*', s=250, edgecolors='k', zorder=2)

lims = [min(hits_train + preds_train + hits_test + preds_test),
        max(hits_train + preds_train + hits_test + preds_test)]
ax.plot(lims, lims, 'k--')

legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='train'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='test'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=14, label='phi_0'),
    Line2D([0], [0], color='k', linestyle='--', label='identity'),
]
ax.legend(handles=legend_elements)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Distance from phi_0')

ax.set_xlabel('True hits')
ax.set_ylabel('Predicted hits')
ax.set_title('Predicted vs. true hits')
fig.savefig(os.path.join(FIGS_DIR, 'gp_true_vs_pred_hits.png'))
print(f"Saved figure to {os.path.join(FIGS_DIR, 'gp_true_vs_pred_hits.png')}")
plt.close(fig)
