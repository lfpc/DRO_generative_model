"""Inner loop: the worst-case risk of a fixed design under three ambiguity sets.

  KL-DRO        reweights the sample, KL(w || 1/n) <= rho.
  Wasserstein   moves the sample points, (1/n) sum ||delta||^2 <= eps^2 (a W_2 ball).
  Latent-DRO    perturbs the latent of the fitted flow inside a KL ball.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dro.baselines import kl_weights                              # noqa: E402
from dro.transforms import TRANSFORMS, project_to_ball            # noqa: E402


def nominal(problem, phi, x):
    return float(problem.risk(phi, x))


def kl_dro(problem, phi, x, rho, loss_fn=None):
    l = (loss_fn or problem.loss)(phi, x).detach()
    w = kl_weights(l, rho)
    return float((w * l).sum()), w


def wasserstein_dro(problem, phi, x, eps, steps=200, lr=0.02, loss_fn=None):
    """Normalised gradient ascent on the displacement, rescaled into the W_2 ball."""
    d = torch.zeros_like(x, requires_grad=True)
    for _ in range(steps):
        g, = torch.autograd.grad((loss_fn or problem.loss)(phi, x + d).mean(), d)
        with torch.no_grad():
            d += lr * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rms = (d ** 2).sum(-1).mean().sqrt()
            if rms > eps:
                d *= eps / rms
    xa = (x + d).detach()
    return float(problem.risk(phi, xa)), xa


def latent_dro(problem, flow, phi, rho, family='full-affine', n=20000, steps=300,
               lr=0.03, seed=0, loss_fn=None):
    """Adam ascent on an affine map of the flow's latent, projected onto the KL ball."""
    g = torch.Generator().manual_seed(seed)
    tf = TRANSFORMS[family](flow.dim)
    opt = torch.optim.Adam(tf.parameters(), lr=lr)
    z = torch.randn(n, flow.dim, generator=g)
    for _ in range(steps):
        x, _ = flow(tf(z))
        opt.zero_grad()
        (-(loss_fn or problem.loss)(phi, x).mean()).backward()
        opt.step()
        project_to_ball(tf, rho)
    with torch.no_grad():
        xa, _ = flow(tf(z))
    return float(problem.risk(phi, xa)), xa


def plausibility(problem, x):
    """Mean log-density of x under the TRUE model."""
    return float(problem.true_log_prob(x).mean())


class Counter:
    """Wraps problem.loss and counts true-loss evaluations, in (design, sample) pairs."""

    def __init__(self, problem):
        self.problem, self.n = problem, 0

    def loss(self, phi, x):
        self.n += x.shape[0]
        return self.problem.loss(phi, x)
