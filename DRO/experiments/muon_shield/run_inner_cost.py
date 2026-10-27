"""Inner-loop cost in simulated muons; also fixes models/design.pt and models/surrogate.pt.

    python3 run_inner_cost.py                 # after train_generator.py
    python3 run_inner_cost.py --phi my_design.txt

KL needs the hits once. Wasserstein on the true loss would need an input gradient cuda_muons
does not have, so it is drawn as a hollow marker at its would-be cost (steps x muons). The
transport and latent adversaries ascend the surrogate; every risk is re-simulated.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inner                                                       # noqa: E402
from optb.lcso import _fit, _make_surrogate                        # noqa: E402
from problem import MuonShield                                     # noqa: E402
from train_generator import Generator                              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
RHOS = [0.05, 0.2, 0.5, 1.0]
EPS = [0.05, 0.1, 0.2, 0.4]            # in standardised feature units
STYLE = {'KL-DRO': ('C2', 'o'), 'Wasserstein (surrogate)': ('C3', 's'),
         'Latent-DRO (surrogate)': ('C0', 'D')}


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


class Adapter:
    """MuonShield with the charge carried as a seventh column, for the surrogate."""

    link = 'identity'

    def __init__(self, m):
        self.m, self.dim = m, m.design_dim
        self.loss_scale = m.loss_scale             # the surrogate fits loss / N_mu

    def sample(self, n, generator=None):
        x, y = self.m.sample(n, generator)
        return torch.cat([x, y.unsqueeze(1)], 1)

    def loss(self, phi, xy):
        x, y = xy[:, :6], xy[:, 6]
        if phi.dim() == 1:
            return self.m.loss(phi, x, y)
        return torch.stack([self.m.loss(p, x, y) for p in phi])


def fit_surrogate(problem, phi, n_designs=60, n_samples=1000000, delta=0.1, epochs=400, seed=0):
    """Fit s(phi, x, y) on simulated hits of n_designs designs in a box of half-width
    delta * (half the bound range) around phi, fresh muons each. Returns (loss_fn, muons simulated)."""
    g = torch.Generator().manual_seed(seed)
    ad = Adapter(problem)
    half = (problem.bounds[1] - problem.bounds[0]) / 2
    designs = torch.stack([problem.project(phi + delta * half *
                                           (torch.rand(phi.numel(), generator=g) * 2 - 1))
                           for _ in range(n_designs)])
    designs[0] = phi
    X, Y = [], []
    for i in range(n_designs):
        xy = ad.sample(n_samples, g)
        X.append(xy)
        Y.append(ad.loss(designs[i], xy).detach())
    X, Y = torch.stack(X), torch.stack(Y)
    print(f'  surrogate data: {n_designs} designs x {n_samples:,} muons, '
          f'{int((Y > 0.5 * problem.loss_scale).sum()):,} hits '
          f'(mean hit rate {float(Y.mean()) / problem.loss_scale:.3g})')
    model = _make_surrogate(ad, 'mlp', seed)
    model, _ = _fit(model, designs, X, Y, epochs, seed, link='identity')
    return problem.wrap_surrogate(model), n_designs * n_samples


def load_surrogate(path, problem):
    """The surrogate saved by this script, as a (phi, x, y) -> per-muon loss."""
    model = _make_surrogate(Adapter(problem), 'mlp', 0)
    model.load_state_dict(torch.load(path, weights_only=False))
    model.eval()
    return problem.wrap_surrogate(model)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--phi', default=None, help='text file with the free parameters (default: reference)')
    ap.add_argument('--n', type=float, default=1e8, help='nominal muons')
    ap.add_argument('--n_designs', type=int, default=60, help='surrogate fit: designs')
    ap.add_argument('--per_design', type=float, default=1e6, help='surrogate fit: muons per design')
    ap.add_argument('--delta', type=float, default=0.1, help='design box half-width / half bound range')
    ap.add_argument('--epochs', type=int, default=400, help='surrogate epochs')
    ap.add_argument('--steps', type=int, default=200, help='transport adversary steps')
    a = ap.parse_args()
    a.n, a.per_design = int(a.n), int(a.per_design)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    problem = MuonShield()
    phi = (problem.init_design() if a.phi is None
           else torch.as_tensor(np.loadtxt(a.phi), dtype=torch.float32))
    x, y = problem.sample(a.n, torch.Generator().manual_seed(0))
    flow = Generator.load(os.path.join(MODELS, 'generator.pt')).flow
    surrogate, setup = fit_surrogate(problem, phi, a.n_designs, a.per_design, a.delta, a.epochs)
    print(f'surrogate fitted with {setup:,} simulated muons (one-off, shared by every solve)')
    torch.save(phi, os.path.join(MODELS, 'design.pt'))
    torch.save(surrogate.model.state_dict(), os.path.join(MODELS, 'surrogate.pt'))
    print(f'  saved design and surrogate to {MODELS}/')

    rows = {k: [] for k in STYLE}
    for rho in RHOS:
        c = inner.Counter(problem)
        v, _ = inner.kl_dro(problem, phi, x, y, rho, loss_fn=c.loss)
        rows['KL-DRO'].append((c.n, v))
    for eps in EPS:
        v, _ = inner.wasserstein_dro(problem, phi, x, y, eps, steps=a.steps, loss_fn=surrogate)
        rows['Wasserstein (surrogate)'].append((0, v))
    for rho in RHOS:
        v, _, _ = inner.latent_dro(problem, flow, phi, rho, n=a.n, loss_fn=surrogate)
        rows['Latent-DRO (surrogate)'].append((0, v))

    nom = inner.nominal(problem, phi, x, y)
    print(f'\n  nominal risk {nom:.4g}  (hit rate {problem.hit_rate(phi, x, y):.4g})')
    for name in STYLE:
        print(f'\n  {name}')
        for n, v in rows[name]:
            print(f'    worst risk {v:10.4g}   simulated muons {n:,}')
    print(f'\n  Wasserstein on the true loss: not runnable (no input gradient); it would cost '
          f'{a.steps * a.n:,} simulated muons per solve, plus the derivative.')

    fig, ax = plt.subplots(figsize=(6.4, 3.9), constrained_layout=True)
    for name, (c_, m) in STYLE.items():
        n, v = zip(*rows[name])
        ax.plot([max(k, 0.5) for k in n], v, color=c_, marker=m, ms=6, mfc='none', lw=1.7,
                label=name)
    # no truth to draw: placed at the risks the surrogate-driven transport reached
    ax.plot([a.steps * a.n] * len(EPS), [v for _, v in rows['Wasserstein (surrogate)']],
            color='C1', marker='v', ms=7, mfc='none', ls='none',
            label='Wasserstein (true loss): would cost, not runnable')
    ax.axhline(nom, color='.5', lw=1, ls=(0, (6, 3)))
    ax.text(.99, nom, 'nominal ', color='.5', fontsize=7, ha='right', va='bottom',
            transform=ax.get_yaxis_transform())
    ax.axvline(setup, color='C0', lw=1, ls=':')
    ax.text(setup, .5, ' surrogate fit (one-off)', color='C0', fontsize=7, rotation=90,
            va='center', transform=ax.get_xaxis_transform())
    ax.set(xscale='log', xlabel='simulated muons for ONE inner solve',
           ylabel='worst-case expected hits (re-simulated)',
           title='what the inner loop costs (muon shield)')
    ax.legend(fontsize=7, loc='upper left')
    fig.savefig(os.path.join(OUT, 'inner_cost.png'), dpi=150)
    print(f'\n  wrote {OUT}/inner_cost.png')
