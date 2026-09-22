"""What does the inner loop cost, in true-loss evaluations?

    python3 run_inner_cost.py

The three adversaries reach comparable damage by very different routes, and the routes have
very different prices in calls to the real objective -- the thing a simulator charges for:

  KL-DRO        the losses at the observed points, once. The maximizer w propto exp(l/beta)
                is a closed form of those losses, so no second pass is needed.
  Wasserstein   the losses AND their input gradients at every ascent step, because the
                adversary is moving the points through the true loss surface.
  Latent-DRO    nothing. The adversary moves the generator's latent and reads a surrogate,
                so the inner loop never touches the simulator. The surrogate is not free,
                but it is fitted once for the outer loop and reused by every inner solve.

That last line is the claim this script is here to check rather than assert.
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
from problem import TwoMoons                                       # noqa: E402
from train_generator import train                                  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
RHOS = [0.05, 0.2, 0.5, 1.0]
EPS = [0.05, 0.1, 0.15, 0.2]
STYLE = {'KL-DRO': ('C2', 'o'), 'Wasserstein': ('C1', 'v'),
         'Latent-DRO (surrogate)': ('C0', 'D')}


class Adapter:
    """TwoMoons with the label carried as a third coordinate, for the surrogate."""

    link = 'identity'

    def __init__(self, m):
        self.m, self.dim = m, m.design_dim

    def sample(self, n, generator=None):
        x, y = self.m.sample(n, generator)
        return torch.cat([x, y.unsqueeze(1)], 1)

    def loss(self, phi, xy):
        x, y = xy[:, :2], xy[:, 2]
        if phi.dim() == 1:
            return self.m.loss(phi, x, y)
        return torch.stack([self.m.loss(p, x, y) for p in phi])


def fit_surrogate(problem, phi, n_designs=60, n_samples=2000, delta=0.25, seed=0):
    """Fit s(phi, x) near the incumbent. Returns (loss_fn, evaluations spent)."""
    g = torch.Generator().manual_seed(seed)
    ad = Adapter(problem)
    designs = phi + delta * (torch.rand(n_designs, phi.numel(), generator=g) * 2 - 1)
    designs[0] = phi
    X, Y = [], []
    for i in range(n_designs):
        x, y = problem.sample(n_samples, g)
        xy = torch.cat([x, y.unsqueeze(1)], 1)
        X.append(xy)
        Y.append(problem.loss(designs[i], x, y).detach())
    model = _make_surrogate(ad, 'mlp', seed)
    model, _ = _fit(model, designs, torch.stack(X), torch.stack(Y), 400, seed,
                    link='identity')

    def loss_fn(p, x, y):
        return model(p, torch.cat([x, y.unsqueeze(1)], 1))
    return loss_fn, n_designs * n_samples


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    os.makedirs(OUT, exist_ok=True)
    problem = TwoMoons(kind='poly', degree=5)
    x, y = problem.sample(20000, torch.Generator().manual_seed(0))
    phi = problem.fit(x, y, steps=3000)
    flows, _ = train(problem, n_train=4000, epochs=1500, seed=0)
    surrogate, setup = fit_surrogate(problem, phi)
    print(f'surrogate fitted with {setup:,} evaluations (one-off, shared by every solve)')

    rows = {k: [] for k in STYLE}
    for rho in RHOS:
        c = inner.Counter(problem)
        v, _ = inner.kl_dro(problem, phi, x, y, rho, loss_fn=c.loss)
        rows['KL-DRO'].append((c.n, v))
    for eps in EPS:
        c = inner.Counter(problem)
        v, _ = inner.wasserstein_dro(problem, phi, x, y, eps, loss_fn=c.loss)
        rows['Wasserstein'].append((c.n, v))
    for rho in RHOS:
        c = inner.Counter(problem)
        v, _, _ = inner.latent_dro(problem, flows, phi, rho, loss_fn=surrogate)
        rows['Latent-DRO (surrogate)'].append((c.n, v))

    for name in STYLE:
        print(f'\n  {name}')
        for n, v in rows[name]:
            print(f'    worst risk {v:.4f}   true-loss evaluations {n:,}')

    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    for name, (c_, m) in STYLE.items():
        n, v = zip(*rows[name])
        n = [max(k, 0.5) for k in n]          # 0 is not plottable on a log axis
        ax.plot(n, v, color=c_, marker=m, ms=6, mfc='none', lw=1.7, label=name)
    ax.axhline(inner.nominal(problem, phi, x, y), color='.5', lw=1, ls=(0, (6, 3)))
    ax.text(.99, inner.nominal(problem, phi, x, y), 'nominal ', color='.5', fontsize=7,
            ha='right', va='bottom', transform=ax.get_yaxis_transform())
    ax.axvline(setup, color='C0', lw=1, ls=':')
    ax.text(setup, .5, ' surrogate fit (one-off)', color='C0', fontsize=7, rotation=90,
            va='center', transform=ax.get_xaxis_transform())
    ax.set(xscale='log', xlabel='true-loss evaluations for ONE inner solve',
           ylabel='worst-case risk', title='what the inner loop costs (two moons)')
    ax.legend(fontsize=8, loc='upper left')
    fig.savefig(os.path.join(OUT, 'inner_cost.png'), dpi=150)
    print(f'\n  wrote {OUT}/inner_cost.png')
