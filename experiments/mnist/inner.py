
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dro.baselines import kl_weights                              # noqa: E402
from dro.transforms import TRANSFORMS, project_to_ball            # noqa: E402

from problem import CLASSES                                    # noqa: E402


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
    tf = {s: TRANSFORMS[family](flows[s].dim) for s in CLASSES}
    opt = torch.optim.Adam([p for s in CLASSES for p in tf[s].parameters()], lr=lr)
    z = {s: torch.randn(n // len(CLASSES), flows[s].dim, generator=g) for s in CLASSES}

    for _ in range(steps):
        loss = 0.0
        for s in CLASSES:
            x, _ = flows[s](tf[s](z[s]))
            loss = loss + (loss_fn or problem.loss)(phi, x, torch.full((x.shape[0],), s, dtype=torch.long)).mean()
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
            ys.append(torch.full((xi.shape[0],), s, dtype=torch.long))
        xa, ya = torch.cat(xs), torch.cat(ys)
        return float(problem.risk(phi, xa, ya)), xa, ya


# ---------------------------------------------------------------------------
def plausibility(judge, x, y=None):
    """Mean log-density of the adversarial points under the HELD-OUT judge flow.

    Elsewhere in this suite this is the exact density of the true model. MNIST has none, so
    the judge is a flow fitted to the 10,000 test images -- data neither the classifier nor
    the class-conditional generator ever saw. It answers the same question, less sharply:
    the judge and the generator are the same model class, so a shift the judge cannot see is
    not thereby proven plausible.

    The common axis the three radii do not provide. A reweighting radius measures a
    divergence between weightings of one sample, a transport radius measures a distance in
    input units, and a latent radius measures a data-space divergence; they are not
    comparable. What is comparable is whether the points the adversary certifies against
    are points the world can produce.
    """
    with torch.no_grad():
        return float(judge.log_prob(x).mean())

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
