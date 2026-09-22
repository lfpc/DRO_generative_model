"""Baselines. Every method returns a design phi, and every robust method has exactly one
radius knob so the robustness-performance frontier can be swept on equal terms.
"""
import torch

from .method import optimize_design


# ---------------------------------------------------------------------------
# No robustness
# ---------------------------------------------------------------------------
def erm(problem, loss_fn, x_pool, steps=300, lr=0.05, trace_every=0, **kw):
    """Empirical risk on a fixed pool (real sample, or generator samples)."""
    return optimize_design(problem, lambda p: loss_fn(p, x_pool).mean(), steps=steps,
                           lr=lr, trace_every=trace_every)


def erm_resample(problem, loss_fn, sampler, steps=300, lr=0.05, batch=4096, seed=0, **kw):
    """Empirical risk against fresh samples each step (generator, possibly tempered)."""
    g = torch.Generator().manual_seed(seed)
    return optimize_design(problem, lambda p: loss_fn(p, sampler(batch, g)).mean(),
                           steps=steps, lr=lr)


# ---------------------------------------------------------------------------
# Non-adversarial robustness
# ---------------------------------------------------------------------------
def random_latent(problem, flow, loss_fn, rho, steps=300, lr=0.05, batch=4096, seed=0, trace_every=0, **kw):
    """Random latent shift of the same KL radius: robustness without an adversary."""
    g = torch.Generator().manual_seed(seed)
    dtype = next(flow.parameters()).dtype
    radius = (2 * rho) ** 0.5

    def obj(phi):
        z = torch.randn(batch, flow.dim, generator=g, dtype=dtype)
        d = torch.randn(flow.dim, generator=g, dtype=dtype)
        mu = radius * d / torch.linalg.norm(d)
        with torch.no_grad():
            x, _ = flow(z + mu)
        return loss_fn(phi, x).mean()

    return optimize_design(problem, obj, steps=steps, lr=lr, trace_every=trace_every)


# ---------------------------------------------------------------------------
# Reweighting DRO over a fixed pool
# ---------------------------------------------------------------------------
def kl_weights(losses, rho, iters=80):
    """argmax_w w'l s.t. sum w = 1, KL(w || uniform) <= rho.  w propto exp(l / beta).

    KL(w || u) = sum_i w_i log(n w_i) is evaluated with `xlogy`, which is not an
    optional nicety: writing it as w * log(clamp_min(w, tiny)) silently returns nan in
    float32 as soon as one weight underflows to zero, because `tiny` itself underflows
    and 0 * log(0) = nan. The bisection below compares that value against rho, and a nan
    compares False, so the search would walk to the smallest beta -- i.e. all mass on the
    largest loss -- at *every* radius, turning the baseline into an unintended worst-case
    method.
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


def chi2_weights(losses, rho):
    """argmax_w w'l s.t. w in the simplex and 0.5 sum (n w_i - 1)^2 / n <= rho.

    The maximizer is w_i propto (1 + (l_i - eta) / beta)_+ with TWO free constants: beta
    sets the scale and eta is the threshold below which a point gets zero weight. Fixing
    eta at the mean (i.e. using the single-parameter family (1 + c (l - lbar))_+) is wrong
    as soon as the truncation binds: that family converges to w propto (l - lbar)_+ as
    c grows and so cannot reach a large radius at all, which silently caps the baseline.
    Here eta is solved for by bisection at each beta, and beta by an outer bisection.
    """
    l = losses.detach()
    n = l.numel()
    if rho <= 0:
        return torch.full_like(l, 1.0 / n)
    spread = float(l.max() - l.min()) + 1e-12

    def w_of(beta):
        lo, hi = float(l.min()) - 2 * beta - spread, float(l.max()) + 2 * beta + spread
        for _ in range(80):                      # sum is decreasing in eta
            eta = 0.5 * (lo + hi)
            if float((1.0 + (l - eta) / beta).clamp_min(0.0).sum()) > n:
                lo = eta
            else:
                hi = eta
        w = (1.0 + (l - 0.5 * (lo + hi)) / beta).clamp_min(0.0)
        return w / w.sum().clamp_min(torch.finfo(w.dtype).tiny)

    def chi2_of(beta):
        w = w_of(beta)
        return float(0.5 * ((n * w - 1.0) ** 2).sum() / n)

    lo, hi = spread / n * 1e-4, spread * 10 + 1.0   # chi2 decreases in beta
    while chi2_of(hi) > rho and hi < 1e12:
        hi *= 4
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if chi2_of(mid) > rho:
            lo = mid
        else:
            hi = mid
    return w_of(hi)


def reweight_dro(problem, loss_fn, x_pool, rho, kind='chi2', steps=300, lr=0.05, trace_every=0, **kw):
    """Reweight a fixed pool, then descend on the reweighted mean.

    Note the radius is only useful up to a point that the *loss* fixes. For a 0/1 loss
    with a fraction q of the pool failing, the most divergent admissible weighting is the
    one supported on the failure set, at KL = log(1/q) (chi2 = (1/q - 1)/2). Any larger
    radius returns exactly that weighting, the reweighted risk saturates at its maximum,
    and the only surviving gradient is the one from the non-random part of the objective
    -- for P2 that is the area, so the design shrinks to the smallest box corner and
    fails with probability one. The sweep keeps such radii in the frontier and the
    best-knob selection discards them; it is a real property of f-divergence sets on a
    fixed sample, not an implementation artifact, and one the latent set does not share
    because every admissible eta still yields a proper distribution over inputs.
    """

    def obj(phi):
        l = loss_fn(phi, x_pool)
        w = chi2_weights(l, rho) if kind == 'chi2' else kl_weights(l, rho)
        return (w.detach() * l).sum()

    return optimize_design(problem, obj, steps=steps, lr=lr, trace_every=trace_every)


def cvar(problem, loss_fn, x_pool, alpha=0.2, steps=300, lr=0.05, trace_every=0, **kw):
    k = max(1, int(round(alpha * x_pool.shape[0])))

    def obj(phi):
        l = loss_fn(phi, x_pool)
        return torch.topk(l, k).values.mean()

    return optimize_design(problem, obj, steps=steps, lr=lr, trace_every=trace_every)


# ---------------------------------------------------------------------------
# Data-space adversaries
# ---------------------------------------------------------------------------
def wrm(problem, loss_fn, x_pool, gamma, steps=300, lr=0.05, inner_steps=15,
        inner_lr=0.05, scale=None, trace_every=0, **kw):
    """Sinha et al.: per-sample ascent on x of loss - gamma ||x - x0||^2 (scaled units)."""
    sc = torch.ones(x_pool.shape[1], dtype=x_pool.dtype) if scale is None else scale

    def obj(phi):
        x = x_pool.clone().requires_grad_(True)
        for _ in range(inner_steps):
            pen = gamma * (((x - x_pool) / sc) ** 2).sum(-1)
            adv = (loss_fn(phi.detach(), x) - pen).sum()
            gx, = torch.autograd.grad(adv, x)
            x = (x + inner_lr * sc * sc * gx).detach().requires_grad_(True)
        return loss_fn(phi, x.detach()).mean()

    return optimize_design(problem, obj, steps=steps, lr=lr, trace_every=trace_every)


def pgd_adv(problem, loss_fn, x_pool, eps, steps=300, lr=0.05, inner_steps=10,
            scale=None, trace_every=0, **kw):
    """L2-ball adversarial training in standardized input units."""
    sc = torch.ones(x_pool.shape[1], dtype=x_pool.dtype) if scale is None else scale

    def obj(phi):
        delta = torch.zeros_like(x_pool, requires_grad=True)
        for _ in range(inner_steps):
            l = loss_fn(phi.detach(), x_pool + delta * sc).sum()
            gd, = torch.autograd.grad(l, delta)
            gd = gd / gd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            delta = (delta + (eps / inner_steps * 2.5) * gd).detach()
            nrm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            delta = (delta * (eps / nrm).clamp(max=1.0)).requires_grad_(True)
        return loss_fn(phi, (x_pool + delta * sc).detach()).mean()

    return optimize_design(problem, obj, steps=steps, lr=lr, trace_every=trace_every)


# ---------------------------------------------------------------------------
# Cheap version of our own method: the first-order expansion as a regularizer
# ---------------------------------------------------------------------------
def gradnorm_reg(problem, flow, loss_fn, rho, steps=300, lr=0.05, batch=4096, seed=0, trace_every=0, **kw):
    """Proposition 8 to first order: Psi(phi,0) + rho' ||grad_eta Psi(phi,0)||."""
    g = torch.Generator().manual_seed(seed)
    dtype = next(flow.parameters()).dtype
    radius = (2 * rho) ** 0.5

    def obj(phi):
        z = torch.randn(batch, flow.dim, generator=g, dtype=dtype)
        mu = torch.zeros(flow.dim, dtype=dtype, requires_grad=True)
        x, _ = flow(z + mu)
        base = loss_fn(phi, x).mean()
        ge, = torch.autograd.grad(base, mu, create_graph=True)
        return base + radius * torch.linalg.norm(ge)

    return optimize_design(problem, obj, steps=steps, lr=lr, trace_every=trace_every)
