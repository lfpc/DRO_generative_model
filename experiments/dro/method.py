"""The exact trust-region subproblem, and the projected first-order design loop."""
import torch


def trust_region_max(g, B, radius):
    """argmax of g'v + 0.5 v'Bv over ||v|| <= radius, globally, by More-Sorensen.

    Returns (v, lam, kkt_residual). The maximiser satisfies (lam I - B) v = g with
    lam >= max(0, lam_max(B)) and lam (radius - ||v||) = 0; ||v(lam)|| decreases in lam,
    so one bisection finds it. Works for indefinite B. Solved in float64, returned in g's dtype.
    """
    dtype_in = g.dtype
    g = g.double()
    B = B.double()
    B = 0.5 * (B + B.T)
    ev, Q = torch.linalg.eigh(B)
    gh = Q.T @ g
    top = float(ev[-1])

    def resid(v, lam):
        return float(torch.linalg.norm(lam * v - B @ v - g) / max(float(torch.linalg.norm(g)), 1e-300))

    if top < 0:                                  # concave: unconstrained max may be inside
        v = -torch.linalg.solve(B, g)
        if float(torch.linalg.norm(v)) <= radius:
            return v.to(dtype_in), 0.0, resid(v, 0.0)

    def v_of(lam):
        return Q @ (gh / (lam - ev))

    lo = top + 1e-12 * max(1.0, abs(top))
    if float(torch.linalg.norm(v_of(lo))) < radius:
        # hard case: g has no component on the top eigenvector
        keep = ev < top - 1e-10 * max(1.0, abs(top))
        v = Q[:, keep] @ (gh[keep] / (top - ev[keep]))
        extra = max(radius ** 2 - float(v @ v), 0.0) ** 0.5
        sign = 1.0 if float(gh[-1]) >= 0 else -1.0
        v = v + extra * Q[:, -1] * sign
        return v.to(dtype_in), top, resid(v, top)

    hi = top + max(1e-12, float(torch.linalg.norm(g)) / radius)
    while float(torch.linalg.norm(v_of(hi))) > radius:
        hi = top + 2 * (hi - top)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(torch.linalg.norm(v_of(mid))) > radius:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    v = v_of(lam)
    return v.to(dtype_in), lam, resid(v, lam)


def make_optimizer(problem, phi, lr):
    """Adam for box-constrained designs, momentum SGD when problem.optimizer == 'sgd'.

    Under a simplex projection Adam's sign-like step is a near-uniform translation, which
    the projection removes, so the iterate never moves; an SGD step keeps the differences.
    """
    if getattr(problem, 'optimizer', 'adam') == 'sgd':
        return torch.optim.SGD([phi], lr=lr, momentum=0.9)
    return torch.optim.Adam([phi], lr=lr)


def optimize_design(problem, grad_obj, phi0=None, steps=300, lr=0.05, trace_every=0):
    """Projected first-order loop on grad_obj(phi), a differentiable scalar.

    With trace_every > 0 also returns [(step, phi)], recorded before each step.
    """
    phi = (problem.init_design() if phi0 is None else phi0.clone()).to(torch.get_default_dtype())
    phi.requires_grad_(True)
    opt = make_optimizer(problem, phi, lr)
    trace = []
    for t in range(steps):
        if trace_every and t % trace_every == 0:
            trace.append((t, phi.detach().clone()))
        opt.zero_grad()
        obj = grad_obj(phi)
        obj.backward()
        opt.step()
        with torch.no_grad():
            phi.copy_(problem.project(phi.detach()))
    if trace_every:
        trace.append((steps, phi.detach().clone()))
    return (phi.detach(), trace) if trace_every else phi.detach()
