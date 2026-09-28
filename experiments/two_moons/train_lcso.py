"""Train the classifier through an LCSO surrogate instead of the true loss.

    python3 train_lcso.py

Two moons is a testbed here, not an application: the logistic loss is analytic and
differentiable, so the exact gradient is available and LCSO can be scored against it. The
surrogate never sees that gradient. It sees only what a simulator would return -- per-sample
losses at designs someone chose to evaluate -- and has to recover a descent direction from
them.

One iteration: draw classifier weights around the incumbent, evaluate the true loss on a
fresh batch at each, fit a branch-trunk surrogate to the result, step on the surrogate, and
accept or reject on the ratio of actual to predicted improvement. The budget is counted in
simulated samples, so `resample` (many designs, few samples each) and `nominal` (few
designs, many samples) are charged the same.
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

    The label travels with the input as a third coordinate, so the surrogate's trunk sees
    (x1, x2, y) and the design side stays the classifier weights alone. Design bounds are
    a box around the initialisation -- the trust region of a weight space has no natural
    scale otherwise.
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

    def true_objective(self, phi, n=None):
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


def exact_gradient_descent(problem, budget_samples, n_samples=2000, lr=0.05, seed=0,
                           project=True):
    """Reference: Adam on the true loss, charged the same way LCSO is.

    `project` keeps the reference inside the same design box LCSO is confined to. Without
    it the comparison is not between two optimizers but between two feasible sets: the
    logistic loss on separable data is minimized by scaling the weights up without bound,
    so an unconstrained reference walks to ||theta|| = 21 with coordinates of 12 against a
    box half-width of 3, and reaches a risk no box-constrained method can attain.
    """
    g = torch.Generator().manual_seed(seed)
    phi = problem.init_design().requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=lr)
    trace, spent = [], 0
    while spent < budget_samples:
        xy = problem.sample(n_samples, g)
        opt.zero_grad()
        problem.objective(phi, xy).backward()
        opt.step()
        if project:
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
    torch.set_default_dtype(torch.float32)
    os.makedirs(OUT, exist_ok=True)
    problem = MoonsForLCSO()          # polynomial features: the design enters the score
    budget = 2_000_000                # linearly, which is what a branch-trunk surrogate
    print(f'design dimension {problem.dim};  budget {budget:,} samples')

    # Settings that matter, and why.
    #  * `n_samples` sets how many LCSO steps the budget buys. At 20000 per design with 4D
    #    designs a round costs ~100k samples and the whole budget is 19 steps, against a
    #    thousand for the reference; at 100 per design it buys ~240.
    #  * 4D designs per fit is the interior optimum: 2D starves the gradient (the measured
    #    cosine against the true gradient collapses below ~2D), 8D buys a better direction
    #    but too few steps to use it.
    #  * the branch network beats the low-rank quadratic here, as it does on the binary
    #    problems in optb/ -- the design dependence is not well captured by a fixed rank.
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


class MoonsScoreTarget(MoonsForLCSO):
    """Same problem, but the surrogate models the SCORE instead of the loss.

    The score theta' phi(x) is linear in the design, which <b(theta), t(x)> represents
    exactly; the logistic loss is a nonlinear warp of it that the surrogate would otherwise
    have to learn, and whose curvature varies over the design box. Here the warp is applied
    afterwards, in closed form, so the fit only has to get the easy part right.
    """

    def loss(self, phi, xy):                      # what the simulator reports per sample
        x = xy[:, :2]
        if phi.dim() == 1:
            return self.m.logits(phi, x)
        return torch.stack([self.m.logits(p, x) for p in phi])

    def transform(self, score, xy):               # known function, applied to the fit
        return torch.nn.functional.softplus(-xy[:, 2] * score)

    def objective(self, phi, xy):
        return self.transform(self.loss(phi, xy), xy).mean(dim=-1)
