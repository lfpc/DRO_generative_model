"""Optimise the five-column section sizes through an LCSO surrogate.

    python3 train_lcso.py

The decision is the width and depth of each of the five columns, design_dim = 10. The
objective trades material against a smoothed system failure probability, so the
surrogate has to represent a tail event from per-sample losses -- the hardest case for
it, and the one closest to the target application.

The surrogate never sees an analytic gradient: it is fitted to per-sample losses alone, and
the exact gradient is computed only to score the result. Feasibility is whatever the problem
says it is -- LCSO projects rather than clamps.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optb.lcso import Budget, lcso                                # noqa: E402
from problem import ShortColumn                                       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')


class ForLCSO:
    """ShortColumn behind the interface optb.lcso expects."""

    link = 'identity'

    def __init__(self, n_eval=200000):
        self.p = ShortColumn()
        self.dim = self.p.design_dim
        self._eval = self.sample(n_eval, torch.Generator().manual_seed(7))

    def sample(self, n, generator=None):
        return self.p.sample(n, generator)

    def loss(self, phi, x):
        if phi.dim() == 1:
            return self.p.loss(phi, x)
        return torch.stack([self.p.loss(q, x) for q in phi])

    def objective(self, phi, x):
        return self.loss(phi, x).mean(dim=-1)

    def true_objective(self, phi, n=None):
        with torch.no_grad():
            out = self.objective(phi, self._eval)
        return float(out) if out.dim() == 0 else out

    def design_bounds(self):
        return (torch.full((self.dim,), self.p.bounds[0]),
                torch.full((self.dim,), self.p.bounds[1]))

    def init_design(self):
        return self.p.init_design()

    def project(self, phi):
        return self.p.project(phi)


def exact_gradient_descent(problem, budget_samples, n_samples=20000, lr=0.002, seed=0):
    """Reference: momentum SGD on the true loss, projected onto the same feasible set.

    SGD rather than Adam, for the reason documented in the portfolio's version of this
    script: Adam's sign-like update is a near-uniform translation when the gradient
    coordinates share a sign, which a projection annihilates. The step size is set against
    the gradient noise on the batch a budget-matched run can afford, not copied from the
    clean solve.
    """
    g = torch.Generator().manual_seed(seed)
    phi = problem.init_design().requires_grad_(True)
    opt = torch.optim.SGD([phi], lr=lr, momentum=0.9)
    trace, spent = [], 0
    while spent < budget_samples:
        x = problem.sample(n_samples, g)
        opt.zero_grad()
        problem.objective(phi, x).backward()
        opt.step()
        with torch.no_grad():
            phi.copy_(problem.project(phi.detach()))
        spent += n_samples
        trace.append((spent, problem.true_objective(phi.detach())))
    return phi.detach(), trace


def design_panel(problem, phi, ax, title, ref=None):
    """The decision itself, next to the reference -- the analogue of a decision boundary."""
    n = problem.p.n_cols
    idx = torch.arange(1, n + 1)
    ax.bar(idx - .17, phi[:n], width=.32, color='C0', label='width $b$')
    ax.bar(idx + .17, phi[n:], width=.32, color='C1', label='depth $h$')
    if ref is not None:
        ax.plot(idx - .17, ref[:n], 'k_', ms=10, mew=1.6)
        ax.plot(idx + .17, ref[n:], 'k_', ms=10, mew=1.6, label='reference')
    ax.set(title=title, xlabel='column', ylabel='section dimension (m)', xticks=idx)
    ax.legend(fontsize=7)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    os.makedirs(OUT, exist_ok=True)
    problem = ForLCSO()
    budget = 2_000_000
    print(f'design dimension {problem.dim};  budget {budget:,} samples')

    cfg = dict(n_samples=800, min_per_design=100, n_designs=4 * problem.dim,
               surrogate='taylor', step='tr-newton', sampler='sobol', epochs=40, rank=8)
    runs = {}
    for scheme in ('resample', 'nominal'):
        b = Budget(budget)
        phi, _ = lcso(problem, b, scheme=scheme, seed=1, **cfg)
        runs[f'LCSO ({scheme})'] = (phi, b.trace)
        print(f'  LCSO {scheme:9s} risk {problem.true_objective(phi):.4f}  ({b.designs} designs)')

    phi_ref, tr_ref = exact_gradient_descent(problem, budget)
    runs['exact gradient'] = (phi_ref, tr_ref)
    init = problem.true_objective(problem.init_design())
    ref = problem.true_objective(phi_ref)
    print(f'  exact gradient      risk {ref:.4f}')
    print(f'  initial design      risk {init:.4f}')
    for name, (phi, _) in runs.items():
        print(f'  {name:20s} captures '
              f'{100 * (problem.true_objective(phi) - init) / (ref - init):.0f}%'
              f' of the available improvement')

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4), constrained_layout=True)
    for name, (_, trace) in runs.items():
        s_, v = zip(*trace)
        axes[0].plot(s_, v, lw=1.8, label=name)
    axes[0].axhline(init, color='.5', lw=1, ls=(0, (6, 3)))
    axes[0].text(.99, init, 'initial design ', color='.5', fontsize=7, ha='right',
                 va='bottom', transform=axes[0].get_yaxis_transform())
    axes[0].set(xlabel='samples simulated', ylabel='true risk',
                title='convergence (lower is better)')
    axes[0].legend(fontsize=8)
    for ax, (name, (phi, _)) in zip(axes[1:], runs.items()):
        design_panel(problem, phi, ax,
                     f'{name}\nrisk {problem.true_objective(phi):.4f}', ref=phi_ref)
    fig.savefig(os.path.join(OUT, 'lcso.png'), dpi=150)
    print(f'  wrote {OUT}/lcso.png')
