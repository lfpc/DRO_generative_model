"""The inner loop: three ambiguity sets, each asked how much damage it can do.

Given a FIXED classifier, each method solves

    sup_{Q in U(P)}  E_Q[ l(theta; x, y) ]

and returns the worst-case risk together with the distribution that achieved it. The outer
loop is not here; this is only about what each adversary can reach.

The three sets differ in what they are allowed to move, which is the whole point:

  KL-DRO        reweights the observed sample, w in the simplex with KL(w || 1/n) <= rho.
                The maximizer is w propto exp(l/beta) with beta from the constraint
                (Hu & Hong 2013; Namkoong & Duchi 2016). It cannot put mass anywhere it
                has not already seen a point.
  Wasserstein   moves the sample points themselves, with a shared transport budget
                (1/n) sum ||delta_i||^2 <= eps^2 -- the W_2 ball of Mohajerin Esfahani &
                Kuhn (2018); the Lagrangian relaxation of Sinha et al. (2018) is the same
                adversary with the budget priced instead of bounded. It can invent points,
                but nothing stops it inventing implausible ones.
  Latent-DRO    perturbs the latent of the fitted generator, Q_eta = law(G(T_eta(z))),
                with KL(T_eta # p_0 || p_0) <= rho. By the diffeomorphic invariance of an
                f-divergence that constraint IS a data-space divergence, and every point
                it produces is a point the generator can produce.

Labels are never moved: the adversary shifts the class-conditional inputs, which is the
semantics adversarial training assumes.
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


# ---------------------------------------------------------------------------
def kl_dro(problem, phi, x, y, rho, loss_fn=None):
    """Reweight the sample. Returns (worst risk, weights)."""
    l = (loss_fn or problem.loss)(phi, x, y).detach()
    w = kl_weights(l, rho)
    return float((w * l).sum()), w


# ---------------------------------------------------------------------------
def wasserstein_dro(problem, phi, x, y, eps, steps=200, lr=0.05, loss_fn=None):
    """Move the points under a shared W_2 budget. Returns (worst risk, moved points).

    Ascent on the perturbations followed by a projection that rescales them all to the
    budget -- the budget is on the AVERAGE squared displacement, not per point, so the
    adversary is free to spend it unevenly, which is what makes it a Wasserstein ball
    rather than a box around every sample.
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


# ---------------------------------------------------------------------------
def latent_dro(problem, flows, phi, rho, family='shift', n=20000, steps=300, lr=0.05,
               seed=0, loss_fn=None):
    """Shake the generator's latent. Returns (worst risk, samples, labels).

    One transform per class, since the generator is class-conditional. The labels are
    balanced, so the divergence of the joint is the average of the two class divergences
    and each class gets the same budget.
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


# ---------------------------------------------------------------------------
def plausibility(problem, x, y):
    """Mean log-density of the adversarial points under the TRUE model.

    The common axis the three radii do not provide. A reweighting radius measures a
    divergence between weightings of one sample, a transport radius measures a distance in
    input units, and a latent radius measures a data-space divergence; they are not
    comparable. What is comparable is whether the points the adversary certifies against
    are points the world can produce.
    """
    return float(problem.log_prob(x, y).mean())

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
