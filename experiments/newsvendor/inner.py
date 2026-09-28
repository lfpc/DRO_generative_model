"""The inner loop: three ambiguity sets against a fixed design.

Given a fixed decision, each method solves sup_{Q in U(P)} E_Q[l] and returns the
worst-case risk with the distribution that achieved it.

  KL-DRO        reweights the observed sample, KL(w || 1/n) <= rho, maximiser
                w propto exp(l/beta) (Hu & Hong 2013; Namkoong & Duchi 2016).
  Wasserstein   moves the sample points under a shared budget (1/n) sum ||delta||^2
                <= eps^2 -- the W_2 ball of Mohajerin Esfahani & Kuhn (2018).
  Latent-DRO    perturbs the latent of the fitted flow, which by diffeomorphic
                invariance is the same divergence in input space.
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
    """Shared W_2 budget on the displacement of the return vectors."""
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
    """Shake the generator's latent; the affine families keep the KL in closed form."""
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
    """Mean log-density of the adversarial returns under the TRUE model -- the only axis
    on which the three radii can be compared."""
    return float(problem.true_log_prob(x).mean())

class Counter:
    """Counts true-loss evaluations, in (design, sample) pairs.

    The unit is one call of the real objective on one sample -- what a simulator charges
    for. The inner loops differ enormously in this: reweighting needs the losses once,
    a data-space adversary needs them again at every ascent step, and a latent adversary
    reading a surrogate needs none at all.
    """

    def __init__(self, problem):
        self.problem, self.n = problem, 0

    def loss(self, phi, x, *a):
        self.n += x.shape[0]
        return self.problem.loss(phi, x, *a)
