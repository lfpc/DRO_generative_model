"""What does the inner loop cost, in true-loss evaluations? (portfolio)

    python3 run_inner_cost.py

Same accounting as the two-moons version: one unit is one evaluation of the real objective
on one sample. Reweighting needs the losses once, a data-space adversary needs them again
at every ascent step, and a latent adversary reading a surrogate needs none.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inner                                                       # noqa: E402
from optb.lcso import _fit, _make_surrogate                        # noqa: E402
from problem import BeamShield                                      # noqa: E402
from train_generator import train                                  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
RHOS = [0.05, 0.2, 0.5, 1.0]
# A W_2 ball is only as meaningful as the units its ground metric is written in. The four
# entry coordinates here have natural spreads of 40, 0.35, 0.05 and 0.004, and a shared
# squared-displacement budget spends 99.6% of itself on the last of them -- the adversary
# is not choosing a physically interesting shift, it is exploiting the coordinate with the
# smallest numbers. Radii are chosen on that coordinate's scale; at eps = 0.5 every
# particle is already in the detector and the worst case is flat. The latent ball has no
# such choice to get wrong, being invariant under any reparameterisation of the input.
EPS = [0.001, 0.002, 0.005, 0.01]
STYLE = {'KL-DRO': ('C2', 'o'), 'Wasserstein': ('C1', 'v'),
         'Latent-DRO (surrogate)': ('C0', 'D')}


class Adapter:
    link = 'identity'

    def __init__(self, p):
        self.p, self.dim = p, p.design_dim

    def sample(self, n, generator=None):
        return self.p.sample(n, generator)

    def loss(self, phi, x):
        if phi.dim() == 1:
            return self.p.loss(phi, x)
        return torch.stack([self.p.loss(q, x) for q in phi])


def fit_surrogate(problem, phi, n_designs=60, n_samples=2000, delta=0.1, seed=0):
    """Fit s(phi, x) near the incumbent. Returns (loss_fn, evaluations spent).

    The design perturbations are small and projected back onto the simplex: the surrogate
    only has to be right near the incumbent, which is the whole premise of a local model.
    """
    g = torch.Generator().manual_seed(seed)
    ad = Adapter(problem)
    designs = torch.stack([problem.project(phi + delta *
                                           (torch.rand(phi.numel(), generator=g) * 2 - 1))
                           for _ in range(n_designs)])
    designs[0] = phi
    X, Y = [], []
    for i in range(n_designs):
        x = problem.sample(n_samples, g)
        X.append(x)
        Y.append(problem.loss(designs[i], x).detach())
    model = _make_surrogate(ad, 'mlp', seed)
    model, _ = _fit(model, designs, torch.stack(X), torch.stack(Y), 400, seed,
                    link='identity')
    return surrogate_loss_fn(model), n_designs * n_samples


def surrogate_loss_fn(model):
    """The surrogate as a loss: (phi, x) -> per-sample loss."""
    loss_fn = lambda p, x: model(p, x)
    loss_fn.model = model
    return loss_fn


def load_surrogate(path, problem):
    """Rebuild the surrogate saved by this script."""
    model = _make_surrogate(Adapter(problem), 'mlp', 0)
    model.load_state_dict(torch.load(path, weights_only=False))
    model.eval()
    return surrogate_loss_fn(model)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    os.makedirs(OUT, exist_ok=True)
    problem = BeamShield()
    x = problem.sample(20000, torch.Generator().manual_seed(0))
    phi = problem.solve(x)
    flow, _ = train(problem, seed=0)
    surrogate, setup = fit_surrogate(problem, phi)
    print(f'surrogate fitted with {setup:,} evaluations (one-off, shared by every solve)')
    os.makedirs(MODELS, exist_ok=True)
    torch.save(phi, os.path.join(MODELS, 'design.pt'))
    torch.save(surrogate.model.state_dict(), os.path.join(MODELS, 'surrogate.pt'))
    print(f'  saved design and surrogate to {MODELS}/')

    rows = {k: [] for k in STYLE}
    for rho in RHOS:
        c = inner.Counter(problem)
        v, _ = inner.kl_dro(problem, phi, x, rho, loss_fn=c.loss)
        rows['KL-DRO'].append((c.n, v))
    for eps in EPS:
        c = inner.Counter(problem)
        v, _ = inner.wasserstein_dro(problem, phi, x, eps, loss_fn=c.loss)
        rows['Wasserstein'].append((c.n, v))
    for rho in RHOS:
        c = inner.Counter(problem)
        v, _ = inner.latent_dro(problem, flow, phi, rho, loss_fn=surrogate)
        rows['Latent-DRO (surrogate)'].append((c.n, v))

    for name in STYLE:
        print(f'\n  {name}')
        for n, v in rows[name]:
            print(f'    worst risk {v:8.4f}   true-loss evaluations {n:,}')

    nom = inner.nominal(problem, phi, x)
    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    for name, (c_, m) in STYLE.items():
        n, v = zip(*rows[name])
        ax.plot([max(k, 0.5) for k in n], v, color=c_, marker=m, ms=6, mfc='none',
                lw=1.7, label=name)
    ax.axhline(nom, color='.5', lw=1, ls=(0, (6, 3)))
    ax.text(.99, nom, 'nominal ', color='.5', fontsize=7, ha='right', va='bottom',
            transform=ax.get_yaxis_transform())
    ax.axvline(setup, color='C0', lw=1, ls=':')
    ax.text(setup, .5, ' surrogate fit (one-off)', color='C0', fontsize=7, rotation=90,
            va='center', transform=ax.get_xaxis_transform())
    ax.set(xscale='log', xlabel='true-loss evaluations for ONE inner solve',
           ylabel='worst-case mean-CVaR risk', title='what the inner loop costs (portfolio)')
    ax.legend(fontsize=8, loc='upper left')
    fig.savefig(os.path.join(OUT, 'inner_cost.png'), dpi=150)
    print(f'\n  wrote {OUT}/inner_cost.png')
