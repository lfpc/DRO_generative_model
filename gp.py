"""Gaussian-process baseline: fit a GP to the total (weighted) hits of simulated designs (GPU).

Simulates n_phi_training Sobol designs in the +-10% box around the nominal one, fits a
Matern-5/2 GP, tests it on n_phi_test new designs, and plots predicted vs. true hits
(outputs/figs/gp_true_vs_pred_hits.png).

    python gp.py --n_muons 50000000 --n_phi_training 50 --n_phi_test 10
    python gp.py --load_data          # reuse outputs/training_data_gp.h5
"""
import argparse
import os
from math import sqrt, log, exp

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

import utils
from problems import ShipMuonShieldCuda


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument('--fields_map', dest='uniform_fields', action='store_false',
                    help='simulate the field map instead of uniform fields')
parser.add_argument('--load_data', action='store_true', help='reuse the stored designs')
parser.add_argument('--n_muons', type=int, default=int(50e6), help='0 = all')
parser.add_argument('--n_phi_training', type=int, default=50)
parser.add_argument('--n_phi_test', type=int, default=10)
args = parser.parse_args()

DTYPE = torch.float64
OUT_DIR = 'outputs'
FIGS_DIR = os.path.join(OUT_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)
H5_PATH = os.path.join(OUT_DIR, 'training_data_gp.h5')


def GaussianProcess(train_X, train_Y, bounds=None):
    """SingleTaskGP, Matern-5/2 ARD with a dimension-scaled lengthscale prior."""
    d = train_X.shape[-1]
    loc = sqrt(2.0) + 0.5 * log(d)
    base = MaternKernel(nu=2.5, ard_num_dims=d,
                        lengthscale_prior=LogNormalPrior(loc, sqrt(3.0)))
    base.lengthscale = exp(loc)
    covar = ScaleKernel(base, outputscale_prior=LogNormalPrior(0.0, 1.0))
    covar.outputscale = 1.0
    return SingleTaskGP(train_X=train_X, train_Y=train_Y, mean_module=ConstantMean(),
                        covar_module=covar, input_transform=Normalize(d=d, bounds=bounds),
                        outcome_transform=Standardize(m=1)).to(torch.float64)


muon_shield = ShipMuonShieldCuda(uniform_fields=args.uniform_fields, n_samples=args.n_muons)
phi_0 = muon_shield.initial_phi
phi_dim = phi_0.shape[-1]
delta = 0.1
lower_bound = phi_0 - delta * phi_0
upper_bound = phi_0 + delta * phi_0

muons = muon_shield.sample_x()
weight = muons[..., 7]


def total_hits(phi):
    return (muon_shield(phi.float(), muons).to(DTYPE) * weight).sum().item()


if args.load_data:
    with h5py.File(H5_PATH, 'r') as f:
        phis_training = torch.from_numpy(f['phis'][:])
        hits = torch.from_numpy(f['hits'][:])
    print(f'Loaded phis {tuple(phis_training.shape)}, hits {tuple(hits.shape)} from {H5_PATH}')
else:
    phis_training = utils.sample_phi(lower_bound, upper_bound, dim=phi_dim,
                                     n_samples=args.n_phi_training)
    hits = torch.zeros(len(phis_training), dtype=torch.float64)
    for i, phi in enumerate(phis_training):
        print(f'\n Simulating for phi {i + 1}/{len(phis_training)}')
        hits[i] = total_hits(phi)
    with h5py.File(H5_PATH, 'w') as f:
        f.create_dataset('phis', data=phis_training.numpy().astype(np.float32))
        f.create_dataset('hits', data=hits.numpy())
    print(f'Saved phis {tuple(phis_training.shape)}, hits {tuple(hits.shape)} to {H5_PATH}')

hits = hits.view(-1, 1).to(DTYPE)
phis_training = phis_training.to(DTYPE)
model = GaussianProcess(phis_training, hits)
fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

print('Testing the model on new phi samples...')
phis_testing = utils.sample_phi(lower_bound, upper_bound, dim=phi_dim,
                                n_samples=args.n_phi_test).to(DTYPE)

model.eval()
with torch.no_grad():
    hits_test = [total_hits(phi) for phi in phis_testing]
    # q = 1 per design: independent posteriors, no joint covariance
    post_test = model.posterior(phis_testing.unsqueeze(-2))
    preds_test = post_test.mean.squeeze(-1).squeeze(-1).tolist()
    stds_test = post_test.variance.squeeze(-1).squeeze(-1).sqrt().tolist()
    dists_test = (phis_testing - phi_0).norm(dim=-1).tolist()

    hits_train = hits.view(-1).tolist()
    post_train = model.posterior(phis_training.unsqueeze(-2))
    preds_train = post_train.mean.squeeze(-1).squeeze(-1).tolist()
    stds_train = post_train.variance.squeeze(-1).squeeze(-1).sqrt().tolist()
    dists_train = (phis_training - phi_0).norm(dim=-1).tolist()

# --- plot: squares = train, circles = test, star = first training design -------
dists_all = dists_train + dists_test
norm = plt.Normalize(vmin=min(dists_all), vmax=max(dists_all))
cmap = plt.cm.viridis
fig, ax = plt.subplots()
for t, p, s in ((hits_train, preds_train, stds_train), (hits_test, preds_test, stds_test)):
    ax.errorbar(t, p, yerr=s, fmt='none', ecolor='gray', capsize=3, zorder=1)
ax.scatter(hits_train[1:], preds_train[1:], c=dists_train[1:], cmap=cmap, norm=norm,
           marker='s', edgecolors='k', zorder=2)
ax.scatter(hits_test, preds_test, c=dists_test, cmap=cmap, norm=norm,
           marker='o', edgecolors='k', zorder=2)
ax.scatter(hits_train[0], preds_train[0], c=[dists_train[0]], cmap=cmap, norm=norm,
           marker='*', s=250, edgecolors='k', zorder=2)
lims = [min(hits_train + preds_train + hits_test + preds_test),
        max(hits_train + preds_train + hits_test + preds_test)]
ax.plot(lims, lims, 'k--')
ax.legend(handles=[
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='train'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, label='test'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=14, label='phi_0'),
    Line2D([0], [0], color='k', linestyle='--', label='identity'),
])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Distance from phi_0')
ax.set_xlabel('True hits')
ax.set_ylabel('Predicted hits')
ax.set_title('Predicted vs. true hits')
fig.savefig(os.path.join(FIGS_DIR, 'gp_true_vs_pred_hits.png'))
print(f"Saved figure to {os.path.join(FIGS_DIR, 'gp_true_vs_pred_hits.png')}")
plt.close(fig)
