"""Fixed-pool baselines: empirical risk, and the KL-reweighting worst case."""
import torch

from .method import optimize_design


def erm(problem, loss_fn, x_pool, steps=300, lr=0.05, trace_every=0, **kw):
    """Empirical risk on a fixed pool."""
    return optimize_design(problem, lambda p: loss_fn(p, x_pool).mean(), steps=steps,
                           lr=lr, trace_every=trace_every)


def kl_weights(losses, rho, iters=80):
    """argmax_w w'l s.t. sum w = 1, KL(w || uniform) <= rho.  w propto exp(l / beta).

    KL uses xlogy: w * log(clamp(w)) gives nan in float32 once a weight underflows, and a
    nan comparison would send the bisection to all mass on the largest loss.
    """
    l = losses.detach()
    n = l.numel()
    if rho <= 0:
        return torch.full_like(l, 1.0 / n)

    def kl_of(beta):
        w = torch.softmax(l / beta, dim=0)
        return float(torch.xlogy(w, w * n).sum())

    lo, hi = 1e-8, max(1.0, float(l.std()) * 50 + 1.0)
    while kl_of(hi) > rho and hi < 1e12:      # KL decreases as beta grows
        hi *= 2
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if kl_of(mid) > rho:
            lo = mid
        else:
            hi = mid
    return torch.softmax(l / hi, dim=0)
