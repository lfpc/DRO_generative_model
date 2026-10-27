"""Train the two-moons classifier through an LCSO surrogate, against exact-gradient Adam.

    python3 train_lcso.py

LCSO only sees per-sample losses at the designs it evaluates; the budget is counted in
simulated samples, so `resample` and `nominal` are charged the same.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optb.lcso import Budget, lcso                                # noqa: E402
from problem import TwoMoons                                      # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')


class MoonsForLCSO:
    """TwoMoons behind the interface optb.lcso expects.

    A sample is (x1, x2, y). The design box is init +- `box`.
    """

    link = 'identity'

    def __init__(self, kind='poly', par=5, n_eval=40000, box=3.0, seed=0):
        self.m = TwoMoons(kind=kind, hidden=par, degree=par)
        self.dim = self.m.design_dim
        self.phi0 = self.m.init_design(seed)
        self.box = box
        self._eval = self.sample(n_eval, torch.Generator().manual_seed(7))

    def sample(self, n, generator=None):
        x, y = self.m.sample(n, generator)
        return torch.cat([x, y.unsqueeze(1)], dim=1)

    def loss(self, phi, xy):
        x, y = xy[:, :2], xy[:, 2]
        if phi.dim() == 1:
            return self.m.loss(phi, x, y)
        return torch.stack([self.m.loss(p, x, y) for p in phi])

    def objective(self, phi, xy):
        return self.loss(phi, xy).mean(dim=-1)

    def true_objective(self, phi):
        """Risk on a fixed large evaluation sample; float for one design, tensor for a batch."""
        with torch.no_grad():
            out = self.objective(phi, self._eval)
        return float(out) if out.dim() == 0 else out

    def design_bounds(self):
        return self.phi0 - self.box, self.phi0 + self.box

    def init_design(self):
        return self.phi0.clone()

    def project(self, phi):
        lo, hi = self.design_bounds()
        return torch.minimum(torch.maximum(phi, lo), hi)


def exact_gradient_descent(problem, budget_samples, n_samples=2000, lr=0.05, seed=0):
    """Reference: Adam on the true loss, projected onto the same design box as LCSO."""
    g = torch.Generator().manual_seed(seed)
    phi = problem.init_design().requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=lr)
    trace, spent = [], 0
    while spent < budget_samples:
        xy = problem.sample(n_samples, g)
        opt.zero_grad()
        problem.objective(phi, xy).backward()
        opt.step()
        with torch.no_grad():
            phi.copy_(problem.project(phi.detach()))
        spent += n_samples
        trace.append((spent, problem.true_objective(phi.detach())))
    return phi.detach(), trace


def boundary(problem, phi, ax, title):
    g = torch.linspace(-2, 3, 220)
    h = torch.linspace(-1.5, 2, 220)
    X, Y = torch.meshgrid(g, h, indexing='ij')
    with torch.no_grad():
        z = problem.m.logits(phi, torch.stack([X.reshape(-1), Y.reshape(-1)], 1))
    ax.contourf(X, Y, z.reshape(220, 220), levels=[-1e9, 0, 1e9], colors=['#cfe3f5', '#fbe0cf'])
    ax.contour(X, Y, z.reshape(220, 220), levels=[0], colors='k', linewidths=1.2)
    xy = problem._eval[:3000]
    for s, c in ((1.0, 'C0'), (-1.0, 'C1')):
        m = xy[:, 2] == s
        ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=.4, c=c)
    ax.set(title=title, xticks=[], yticks=[])


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    problem = MoonsForLCSO()
    budget = 2_000_000
    print(f'design dimension {problem.dim};  budget {budget:,} samples')

    # from tune_lcso.py: 4D designs per fit (2D starves the gradient, 8D leaves too few
    # steps), small per-design samples to buy more steps, and the mlp beats taylor here
    cfg = dict(n_samples=800, min_per_design=100, n_designs=4 * problem.dim,
               surrogate='mlp', step='gradient', sampler='sobol', epochs=40)

    runs = {}
    for scheme in ('resample', 'nominal'):
        b = Budget(budget)
        phi, _ = lcso(problem, b, scheme=scheme, seed=0, **cfg)
        runs[f'LCSO ({scheme})'] = (phi, b.trace)
        print(f'  LCSO {scheme:9s} risk {problem.true_objective(phi):.4f}  '
              f'({b.designs} designs)')

    phi_ref, tr_ref = exact_gradient_descent(problem, budget)
    runs['exact gradient'] = (phi_ref, tr_ref)
    print(f'  exact gradient      risk {problem.true_objective(phi_ref):.4f}')
    print(f'  at initialisation   risk {problem.true_objective(problem.init_design()):.4f}')

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), constrained_layout=True)
    for name, (_, trace) in runs.items():
        s_, v = zip(*trace)
        axes[0].plot(s_, v, lw=1.8, label=name)
    axes[0].axhline(problem.true_objective(problem.init_design()), color='.5', lw=1,
                    ls=(0, (6, 3)))
    axes[0].text(.99, problem.true_objective(problem.init_design()), 'initial ', color='.5',
                 fontsize=7, ha='right', va='bottom',
                 transform=axes[0].get_yaxis_transform())
    axes[0].set(xlabel='samples simulated', ylabel='true risk', yscale='log',
                title='convergence')
    axes[0].legend(fontsize=8)
    for ax, (name, (phi, _)) in zip(axes[1:], runs.items()):
        boundary(problem, phi, ax, f'{name}\nrisk {problem.true_objective(phi):.4f}')
    fig.savefig(os.path.join(OUT, 'lcso.png'), dpi=150)
    print(f'  wrote {OUT}/lcso.png')

