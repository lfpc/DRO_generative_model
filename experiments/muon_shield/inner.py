"""The inner loop: three ambiguity sets against a fixed shield.

  KL-DRO        reweights the muons, KL(w || 1/n) <= rho. Needs the per-muon losses once.
  Wasserstein   moves the features under a shared budget (1/n) sum ||delta||^2 <= eps^2.
  Latent-DRO    perturbs the latent of the pretrained flow.

The simulator has no gradient in the muon, so the last two ascend a surrogate (`loss_fn`);
the risk they report is always re-simulated at their points. The charge is a label and is
never moved, except by the latent adversary, for which it is a flow output.
Heavy arithmetic runs on problem.dro_device in chunks of problem.chunk; results are on the CPU.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dro.baselines import kl_weights                              # noqa: E402
from dro.transforms import TRANSFORMS, project_to_ball            # noqa: E402


def nominal(problem, phi, x, y):
    return float(problem.risk(phi, x, y))


def kl_dro(problem, phi, x, y, rho, loss_fn=None):
    """Reweight the sample. Returns (worst risk, weights on the CPU)."""
    l = (loss_fn or problem.loss)(phi, x, y).detach().to(problem.dro_device)
    # per-muon scale keeps the temperature bisection inside float32 range; w is invariant to it
    w = kl_weights(l / problem.loss_scale, rho)
    return float((w * l).sum()), w.cpu()


def wasserstein_dro(problem, phi, x, y, eps, steps=200, lr=0.05, loss_fn=None):
    """Move the features under a shared W_2 budget, ascending the surrogate `loss_fn`.
    Returns (worst risk under the simulator, moved features)."""
    dev, chunk = problem.dro_device, problem.chunk
    xd, yd = x.detach().to(dev), y.detach().to(dev)
    d = torch.zeros_like(xd)
    for _ in range(steps):
        for s in range(0, xd.shape[0], chunk):   # each point ascends its own loss
            dc = d[s:s + chunk].clone().requires_grad_(True)
            g, = torch.autograd.grad(loss_fn(phi, xd[s:s + chunk] + dc, yd[s:s + chunk]).sum(), dc)
            with torch.no_grad():
                d[s:s + chunk] += lr * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        with torch.no_grad():                     # ... under one shared budget
            rms = (d ** 2).sum(-1).mean().sqrt()
            if rms > eps:
                d *= eps / rms
    xa = (xd + d).cpu()
    del xd, yd, d                             # free the GPU before simulating 1e8 muons
    with torch.no_grad():
        return float(problem.risk(phi, xa, y)), xa


def latent_dro(problem, flow, phi, rho, family='full-affine', n=200000, steps=300, lr=0.03,
               seed=0, loss_fn=None, batch=4096):
    """Shake the flow's latent, ascending the surrogate `loss_fn`.
    Returns (worst risk under the simulator, features, charges).

    Each step ascends on a minibatch of `batch` latents (the spline flow's graph is heavy);
    all n latents are pushed through only at the end.
    """
    dev, chunk = problem.dro_device, problem.chunk
    g = torch.Generator().manual_seed(seed)
    flow_d = flow.to(dev)
    tf = TRANSFORMS[family](flow.dim).to(dev)
    opt = torch.optim.Adam(tf.parameters(), lr=lr)
    z = torch.randn(n, flow.dim, generator=g)
    extra = torch.randn(n, 2, generator=g)    # azimuth and dphi sign: not perturbed
    for _ in range(steps):
        idx = torch.randint(n, (min(batch, n),), generator=g) if batch < n else slice(None)
        x, y = flow_d(tf(z[idx].to(dev)), extra[idx].to(dev))
        opt.zero_grad()
        (-loss_fn(phi, x, y).mean()).backward()
        opt.step()
        project_to_ball(tf, rho)
    with torch.no_grad():
        out = [flow_d(tf(z[s:s + chunk].to(dev)), extra[s:s + chunk].to(dev))
               for s in range(0, n, chunk)]
        xa = torch.cat([o[0].cpu() for o in out])
        ya = torch.cat([o[1].cpu() for o in out])
        del out
    flow.cpu()                                # hand the flow back on the CPU
    del z, extra
    with torch.no_grad():
        return float(problem.risk(phi, xa, ya)), xa, ya


def plausibility(problem, x, y):
    """Mean log-density of the muons under the true generator: the common axis of the three radii."""
    return float(problem.log_prob(x, y).mean())


class Counter:
    """Counts simulated muons."""

    def __init__(self, problem):
        self.problem, self.n = problem, 0

    def loss(self, phi, x, *a):
        self.n += x.shape[0]
        return self.problem.loss(phi, x, *a)
