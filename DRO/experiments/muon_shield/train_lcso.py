"""Optimise the muon shield through an LCSO surrogate.

    python3 train_lcso.py
    python3 train_lcso.py --budget 1e10 --surrogate mlp

Runs the resample scheme (many designs, --resample_per_design muons each) and the nominal one
(--n_samples muons per design) at the same budget of simulated muons, and scores both against
the reference shield on a held-out set of --n_eval muons that LCSO never sees. Only --fit_keep
muons per design are kept for the surrogate fit, to bound memory.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optb.lcso import Budget, lcso                                # noqa: E402
from problem import MuonShield                                    # noqa: E402
from muon_shield import PARAM_NAMES                               # noqa: E402  (MuonShieldChallenge)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


class ForLCSO:
    """MuonShield behind the interface optb.lcso expects; the charge rides as a 7th column."""

    link = 'identity'

    def __init__(self, n_eval=100_000_000, fit_keep=None):
        self.p = MuonShield()
        self.dim = self.p.design_dim
        self.loss_scale = self.p.loss_scale        # the surrogate fits loss / N_mu
        self.fit_keep = fit_keep
        self._eval = self.sample(n_eval, torch.Generator().manual_seed(7))
        self._last = None

    def sample(self, n, generator=None):
        x, y = self.p.sample(n, generator)
        return torch.cat([x, y.unsqueeze(1)], 1)

    def loss(self, phi, xy):
        if phi.dim() == 1:
            out = self.p.loss(phi, xy[:, :6], xy[:, 6])
            self._last = (phi.detach().clone(), xy, out)
            return out
        return torch.stack([self.p.loss(q, xy[:, :6], xy[:, 6]) for q in phi])

    def objective(self, phi, xy):
        # lcso.simulate asks for the losses, then the objective of the same muons: reuse them
        if (self._last is not None and self._last[1] is xy and phi.dim() == 1
                and torch.equal(self._last[0], phi.detach())):
            return self._last[2].mean()
        return self.loss(phi, xy).mean(dim=-1)

    def true_objective(self, phi, n=None):
        with torch.no_grad():
            out = self.objective(phi, self._eval)
        return float(out) if out.dim() == 0 else out

    def design_bounds(self):
        return self.p.bounds

    def init_design(self):
        return self.p.init_design()

    def project(self, phi):
        return self.p.project(phi)


def design_panel(problem, phi, ax, title):
    """Each free parameter's move from the reference, in units of half its bound range."""
    lo, hi = problem.design_bounds()
    u = ((phi - problem.init_design()) / ((hi - lo) / 2)).numpy()
    sim = problem.p.sim
    labels = [f'{m}:{PARAM_NAMES[p]}' for m, p in sim.free_index.tolist()]
    ax.bar(np.arange(len(u)), u, color=np.where(u >= 0, 'C0', 'C3'))
    ax.axhline(0, color='.5', lw=.8)
    g = problem.p.constraints(phi)
    ax.set(title=f'{title}\nlength {g[0]:+.2f} m, cost {g[1]:+.3g} (<= 0 feasible)',
           ylabel='move / half range', xticks=np.arange(len(u)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--budget', type=float, default=2.4e10, help='simulated muons per run')
    ap.add_argument('--n_samples', type=float, default=1e8, help='muons per design, nominal scheme')
    ap.add_argument('--resample_per_design', type=float, default=1e7, help='muons per design, resample scheme')
    ap.add_argument('--fit_keep', type=float, default=1e6, help='muons per design kept for the fit')
    ap.add_argument('--n_eval', type=float, default=1e8, help='held-out muons')
    ap.add_argument('--surrogate', default='taylor', choices=['taylor', 'mlp'])
    ap.add_argument('--epochs', type=int, default=40, help='surrogate epochs per round')
    a = ap.parse_args()
    a.budget, a.n_samples, a.n_eval = int(a.budget), int(a.n_samples), int(a.n_eval)
    a.resample_per_design, a.fit_keep = int(a.resample_per_design), int(a.fit_keep)
    os.makedirs(OUT, exist_ok=True)
    problem = ForLCSO(a.n_eval, fit_keep=a.fit_keep)
    print(f'design dimension {problem.dim};  budget {a.budget:,} muons')

    cfg = dict(n_samples=a.n_samples, resample_factor=max(1, a.n_samples // a.resample_per_design),
               min_per_design=1000, n_designs=4 * problem.dim,
               surrogate=a.surrogate, step='tr-newton' if a.surrogate == 'taylor' else 'gd',
               sampler='sobol', epochs=a.epochs, rank=8)
    runs = {}
    for scheme in ('resample', 'nominal'):
        b = Budget(a.budget)
        phi, _ = lcso(problem, b, scheme=scheme, seed=1, verbose=True, **cfg)
        runs[f'LCSO ({scheme})'] = (phi, b.trace)
        print(f'  LCSO {scheme:9s} risk {problem.true_objective(phi):.4g}  ({b.designs} designs)')
    init = problem.true_objective(problem.init_design())
    print(f'  reference design    risk {init:.4g}')
    for name, (phi, _) in runs.items():
        v = problem.true_objective(phi)
        rel = f'{100 * (v - init) / init:+.1f}%' if init > 0 else f'{v - init:+.3g} (absolute)'
        print(f'  {name:20s} {rel} vs the reference; '
              f'constraints {problem.p.constraints(phi).tolist()}')
        np.savetxt(os.path.join(OUT, f'lcso_{name.split("(")[1][:-1]}.txt'), phi.numpy())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    for name, (_, trace) in runs.items():
        s_, v = zip(*trace)
        axes[0].plot(s_, v, lw=1.8, label=name)
    axes[0].axhline(init, color='.5', lw=1, ls=(0, (6, 3)))
    axes[0].text(.99, init, 'reference design ', color='.5', fontsize=7, ha='right',
                 va='bottom', transform=axes[0].get_yaxis_transform())
    axes[0].set(xlabel='muons simulated', ylabel='held-out expected hits',
                title='convergence (lower is better)')
    axes[0].legend(fontsize=8)
    for ax, (name, (phi, _)) in zip(axes[1:], runs.items()):
        design_panel(problem, phi, ax, f'{name}: risk {problem.true_objective(phi):.4g}')
    fig.savefig(os.path.join(OUT, 'lcso.png'), dpi=150)
    print(f'  wrote {OUT}/lcso.png  (designs as text in {OUT}/lcso_*.txt)')
