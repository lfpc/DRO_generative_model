"""Inner loop for two moons: three ambiguity sets against a fixed classifier.

  KL-DRO        reweights the observed sample, KL(w || 1/n) <= rho.
  Wasserstein   moves the inputs x (labels fixed), (1/n) sum ||delta_i||^2 <= eps^2.
  Latent-DRO    perturbs the latent of each class-conditional flow, KL <= rho per class.

Each returns (worst-case risk, the distribution that achieved it).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dro.baselines import kl_weights                              # noqa: E402
from dro.transforms import TRANSFORMS, project_to_ball            # noqa: E402

CLASSES = (1.0, -1.0)


def nominal(problem, phi, x, y):
    return float(problem.risk(phi, x, y))


def kl_dro(problem, phi, x, y, rho, loss_fn=None):
    """Reweight the sample. Returns (worst risk, weights)."""
    l = (loss_fn or problem.loss)(phi, x, y).detach()
    w = kl_weights(l, rho)
    return float((w * l).sum()), w


def wasserstein_dro(problem, phi, x, y, eps, steps=200, lr=0.05, loss_fn=None):
    """Move the points under a shared W_2 budget. Returns (worst risk, moved points).

    The budget is on the AVERAGE squared displacement, so it can be spent unevenly.
    """
    d = torch.zeros_like(x, requires_grad=True)
    for _ in range(steps):
        loss = (loss_fn or problem.loss)(phi, x + d, y).mean()
        g, = torch.autograd.grad(loss, d)
        with torch.no_grad():
            d += lr * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rms = (d ** 2).sum(-1).mean().sqrt()
            if rms > eps:
                d *= eps / rms
    xa = (x + d).detach()
    return float(problem.risk(phi, xa, y)), xa


def latent_dro(problem, flows, phi, rho, family='shift', n=20000, steps=300, lr=0.05,
               seed=0, loss_fn=None):
    """Shake the generator's latent, one transform per class. Returns (worst risk, x, y).

    Labels are balanced, so each class gets the same budget rho.
    """
    g = torch.Generator().manual_seed(seed)
    tf = {s: TRANSFORMS[family](2) for s in CLASSES}
    opt = torch.optim.Adam([p for s in CLASSES for p in tf[s].parameters()], lr=lr)
    z = {s: torch.randn(n // 2, 2, generator=g) for s in CLASSES}

    for _ in range(steps):
        loss = 0.0
        for s in CLASSES:
            x, _ = flows[s](tf[s](z[s]))
            loss = loss + (loss_fn or problem.loss)(phi, x, torch.full((x.shape[0],), s)).mean()
        opt.zero_grad()
        (-loss / len(CLASSES)).backward()
        opt.step()
        for s in CLASSES:
            project_to_ball(tf[s], rho)

    with torch.no_grad():
        xs, ys = [], []
        for s in CLASSES:
            xi, _ = flows[s](tf[s](z[s]))
            xs.append(xi)
            ys.append(torch.full((xi.shape[0],), s))
        xa, ya = torch.cat(xs), torch.cat(ys)
        return float(problem.risk(phi, xa, ya)), xa, ya


def plausibility(problem, x, y):
    """Mean log-density under the TRUE model: the common axis for the three radii."""
    return float(problem.log_prob(x, y).mean())


class Counter:
    """Wraps problem.loss and counts true-loss evaluations, in (design, sample) pairs."""

    def __init__(self, problem):
        self.problem, self.n = problem, 0

    def loss(self, phi, x, y):
        self.n += x.shape[0]
        return self.problem.loss(phi, x, y)
