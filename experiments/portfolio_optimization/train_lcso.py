"""Optimize the mean-CVaR portfolio through an LCSO surrogate.

    python3 train_lcso.py

The decision is (w, tau): the weights on the simplex, and the CVaR level of the
Rockafellar-Uryasev form. Both move together, and both are recovered from per-sample
losses alone -- the surrogate never sees the analytic gradient, which is available here
only to score the result.

Feasibility is a simplex, not a box, so designs are projected rather than clamped; LCSO
asks the problem what feasible means.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optb.lcso import Budget, lcso                                # noqa: E402
from problem import Portfolio                                     # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')


class PortfolioForLCSO:
    """Portfolio behind the interface optb.lcso expects."""

    link = 'identity'

    def __init__(self, n_assets=10, n_eval=200000, tau_range=0.6, seed=0):
        self.p = Portfolio(n_assets=n_assets)
        self.dim = n_assets + 1                      # weights, then tau
        self.tau_range = tau_range
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
        lo = torch.cat([torch.zeros(self.p.n_assets), torch.tensor([-self.tau_range])])
        hi = torch.cat([torch.ones(self.p.n_assets), torch.tensor([self.tau_range])])
        return lo, hi

    def init_design(self):
        return self.p.init_design()

    def project(self, phi):
        """Weights onto the simplex, tau into its range -- what feasibility means here."""
        w = self.p.project(phi)[:-1]
        tau = phi[-1].clamp(-self.tau_range, self.tau_range)
        return torch.cat([w, tau.reshape(1)])


def exact_gradient_descent(problem, budget_samples, n_samples=20000, lr=0.005, seed=0):
    """Reference: momentum SGD on the true loss, projected onto the same feasible set.

    SGD, not Adam, and this is not a detail. Adam's update is sign-like, so when the
    gradient coordinates share a sign the step is a near-uniform translation -- exactly
    what projection onto a simplex annihilates, leaving the iterate near where it started.
    Here that costs 30%: Adam lands between -0.98 and -1.03 depending on its learning rate
    and never leaves the neighbourhood of equal weights, while SGD reaches -1.352 from
    every learning rate tried and drops the four lowest-return assets to zero. The same
    trap is documented in dro/method.py::make_optimizer.

    The step size has to be set against the gradient noise, not copied from the clean
    case: momentum SGD at lr 0.2 converges from every start when the gradient is taken on
    100k samples, and diverges outright (risk +9.8) on the 2k batches a budget-matched run
    can afford. At 20k samples and lr 0.005 it reaches -1.352, matching the clean solve.
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


def portfolio_panel(problem, phi, ax, title, ref=None):
    """The portfolio itself. What matters here is which assets get the weight, so the
    natural analogue of a decision boundary is the allocation next to the reference."""
    n = problem.p.n_assets
    idx = torch.arange(1, n + 1)
    ax.bar(idx, phi[:n], color='C0', width=.62, label='this design')
    if ref is not None:
        ax.plot(idx, ref[:n], 'k_', ms=14, mew=1.8, label='reference')
    ax.set(title=title, xlabel='asset (increasing return and risk)', ylabel='weight',
           xticks=idx[::2], ylim=(0, .28))
    ax.legend(fontsize=7)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    os.makedirs(OUT, exist_ok=True)
    problem = PortfolioForLCSO()
    budget = 2_000_000
    print(f'design dimension {problem.dim} (weights + tau);  budget {budget:,} samples')

    # Settled by a sweep over designs per fit, samples per design, fit length and rank.
    # The trust-region step is what benefits from the normalised-coordinate fix: the tau
    # gradient is 60x a weight gradient, so a step of fixed length in raw coordinates goes
    # almost entirely into tau.
    cfg = dict(n_samples=800, min_per_design=100, n_designs=4 * problem.dim,
               surrogate='taylor', step='tr-newton', sampler='sobol', epochs=40, rank=8)

    runs = {}
    for scheme in ('resample', 'nominal'):
        b = Budget(budget)
        phi, _ = lcso(problem, b, scheme=scheme, seed=1, **cfg)
        runs[f'LCSO ({scheme})'] = (phi, b.trace)
        print(f'  LCSO {scheme:9s} risk {problem.true_objective(phi):.4f}  '
              f'({b.designs} designs)')

    phi_ref, tr_ref = exact_gradient_descent(problem, budget)
    runs['exact gradient'] = (phi_ref, tr_ref)
    init = problem.true_objective(problem.init_design())
    ref = problem.true_objective(phi_ref)
    print(f'  exact gradient      risk {ref:.4f}')
    print(f'  equal weights       risk {init:.4f}')
    for name, (phi, _) in runs.items():
        print(f'  {name:20s} captures {100 * (problem.true_objective(phi) - init) / (ref - init):.0f}%'
              f' of the available improvement')

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4), constrained_layout=True)
    for name, (_, trace) in runs.items():
        s_, v = zip(*trace)
        axes[0].plot(s_, v, lw=1.8, label=name)
    axes[0].axhline(init, color='.5', lw=1, ls=(0, (6, 3)))
    axes[0].text(.99, init, 'equal weights ', color='.5', fontsize=7, ha='right',
                 va='bottom', transform=axes[0].get_yaxis_transform())
    axes[0].set(xlabel='samples simulated', ylabel='true mean-CVaR risk',
                title='convergence (lower is better)')
    axes[0].legend(fontsize=8)
    for ax, (name, (phi, _)) in zip(axes[1:], runs.items()):
        portfolio_panel(problem, phi, ax,
                        f'{name}\nrisk {problem.true_objective(phi):.4f}',
                        ref=phi_ref)
    fig.savefig(os.path.join(OUT, 'lcso.png'), dpi=150)
    print(f'  wrote {OUT}/lcso.png')
