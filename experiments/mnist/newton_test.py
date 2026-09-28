"""Does a low-rank Newton step on a Taylor local model beat SGD for training the classifier?

    python3 newton_test.py

The surrogate used everywhere else in this project is a Taylor model: a quadratic in the
design whose Hessian is low rank. The question here is whether that model, stepped with
Newton rather than gradient descent, is competitive with the optimiser people actually use
to train a network -- on the 101,770 weights of the MNIST MLP.

Three methods, all given the SAME budget in gradient evaluations, which is what either one
actually spends:

  Adam / SGD        the reference, exact stochastic gradients.
  low-rank Newton   exact gradient plus a rank-r curvature model built from Hessian-vector
                    products (which autograd gives exactly, at the cost of one extra
                    backward pass each), stepped as (H_r + lambda I)^-1 g inside a trust
                    region. This is the Taylor model with its curvature measured rather
                    than fitted, which is the most favourable version of the idea: fitting
                    it from sampled designs, as LCSO does, would need O(rank * D) designs
                    and D here is 101,770.
  L-BFGS            the standard quasi-Newton baseline, which also builds a low-rank
                    inverse-Hessian model, so it says whether any gain is from curvature or
                    from this particular way of estimating it.

Accounting: one Hessian-vector product costs about one gradient evaluation, so a rank-r
Newton step costs r+1 of them against SGD's 1. A fair comparison charges for that.
"""
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem import MNIST                                          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')


def _grad(problem, phi, x, y, create_graph=False):
    phi = phi.detach().requires_grad_(True)
    loss = problem.risk(phi, x, y)
    g, = torch.autograd.grad(loss, phi, create_graph=create_graph)
    return float(loss), g


def lanczos_hessian(problem, phi, x, y, rank, seed=0):
    """Rank-r symmetric curvature model by Hessian-vector products.

    Each product is one extra backward pass through the graph of the gradient, so the cost
    is r+1 gradient evaluations. Returns (V, d) with H ~ V diag(d) V', V orthonormal.
    """
    g = torch.Generator().manual_seed(seed)
    phi = phi.detach().requires_grad_(True)
    loss = problem.risk(phi, x, y)
    gr, = torch.autograd.grad(loss, phi, create_graph=True)
    V = torch.linalg.qr(torch.randn(phi.numel(), rank, generator=g))[0]
    HV = torch.stack([torch.autograd.grad(gr @ V[:, j], phi, retain_graph=True)[0]
                      for j in range(rank)], 1)
    # Rayleigh-Ritz in the sampled subspace: the best rank-r symmetric model it supports
    T = 0.5 * (V.T @ HV + (V.T @ HV).T)
    d, U = torch.linalg.eigh(T)
    return (V @ U).detach(), d.detach()


def newton_step(gr, V, d, radius, damping=1e-2):
    """(H_r + lambda I)^-1 g, capped at a trust radius.

    Curvature is only known inside the sampled subspace, so the step uses it there and
    falls back to a damped gradient step in the orthogonal complement -- which is what the
    low-rank model actually licenses.
    """
    c = V.T @ gr
    d_eff = (d + damping).clamp_min(damping)
    step = V @ (c / d_eff) + (gr - V @ c) / (1.0 / damping)
    n = step.norm()
    return step * (radius / n if n > radius else 1.0)


def run(problem, method, budget, batch=256, seed=0, rank=2, lr=None, radius=5.0):
    """Returns (trace of (gradient evaluations, test accuracy), wall-clock seconds)."""
    g = torch.Generator().manual_seed(seed)
    phi = problem.init_design(seed)
    trace, spent, t0 = [], 0, time.time()
    opt = None
    if method in ('adam', 'sgd'):
        phi = phi.requires_grad_(True)
        opt = (torch.optim.Adam([phi], lr=lr or 2e-3) if method == 'adam'
               else torch.optim.SGD([phi], lr=lr or 0.2, momentum=0.9))
    hist = []
    while spent < budget:
        x, y = problem.sample(batch, g)
        if opt is not None:
            opt.zero_grad()
            problem.risk(phi, x, y).backward()
            opt.step()
            spent += 1
        elif method == 'newton':
            _, gr = _grad(problem, phi, x, y)
            V, d = lanczos_hessian(problem, phi, x, y, rank, seed=spent)
            phi = phi - newton_step(gr, V, d, radius)
            spent += rank + 1
        elif method == 'lbfgs':
            _, gr = _grad(problem, phi, x, y)
            q = gr.clone()
            a = []
            for s_, y_, rho in reversed(hist):
                al = rho * (s_ @ q); q = q - al * y_; a.append(al)
            if hist:
                s_, y_, _ = hist[-1]
                q = q * ((s_ @ y_) / (y_ @ y_).clamp_min(1e-12))
            for (s_, y_, rho), al in zip(hist, reversed(a)):
                q = q + s_ * (al - rho * (y_ @ q))
            step = q * min(1.0, radius / q.norm().clamp_min(1e-12))
            new = phi - 0.5 * step
            _, gnew = _grad(problem, new, x, y)
            s_, y_ = (new - phi), (gnew - gr)
            if float(s_ @ y_) > 1e-10:
                hist.append((s_, y_, 1.0 / (s_ @ y_)))
                hist[:] = hist[-10:]
            phi = new
            spent += 2
        if len(trace) < spent // 200:
            p = phi.detach()
            trace.append((spent, problem.accuracy(p, problem.test_features,
                                                  problem.test_labels)))
    return phi.detach(), trace, time.time() - t0


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    os.makedirs(OUT, exist_ok=True)
    problem = MNIST()
    budget = 6000
    print(f'design dimension {problem.design_dim:,};  budget {budget:,} gradient evaluations')

    # every method at its own best setting from a sweep, so the comparison is not a
    # comparison of tuning effort: SGD diverges outright at 0.2 and Newton is better at
    # rank 2 than at rank 8, because what it buys in curvature it loses in steps taken
    tuned = {'adam': dict(lr=1e-3), 'sgd': dict(lr=1e-2),
             'newton': dict(rank=2, radius=5.0), 'lbfgs': dict(radius=5.0)}
    runs = {}
    for method in ('adam', 'sgd', 'newton', 'lbfgs'):
        phi, trace, secs = run(problem, method, budget, **tuned[method])
        acc = problem.accuracy(phi, problem.test_features, problem.test_labels)
        runs[method] = trace
        print(f'  {method:8s} {str(tuned[method]):34s} test accuracy {acc:.4f}   risk '
              f'{float(problem.risk(phi, problem.features, problem.labels)):.4f}   '
              f'{secs:5.0f}s')

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for m, tr in runs.items():
        s_, a = zip(*tr)
        ax.plot(s_, a, lw=1.8, label=m)
    ax.set(xlabel='gradient evaluations (Hessian-vector products charged as one each)',
           ylabel='test accuracy', ylim=(0.75, 1.0),
           title=f'MNIST MLP, {problem.design_dim:,} weights')
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, 'newton.png'), dpi=150)
    print(f'  wrote {OUT}/newton.png')
