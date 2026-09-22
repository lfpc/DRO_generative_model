"""Algorithm 1: DRO with a latent adversary, plus the exact trust-region inner solve.

The objective is

    Psi(phi, eta) = E_{z ~ p_0}[ loss(phi, G_theta(T_eta(z))) ],
    R_rho(phi)    = max_{KL(T_eta # p_0 || p_0) <= rho} Psi(phi, eta).

Inner: projected gradient ascent on eta, or (shift family) the exact trust-region solution
of the second-order model, which is globally optimal even when the model is nonconcave.
Outer: projected gradient descent on phi with eta frozen at its optimum (Danskin).
"""
import torch

from .transforms import TRANSFORMS, CouplingTransform, project_to_ball


# ---------------------------------------------------------------------------
# Exact trust-region subproblem (Proposition 7)
# ---------------------------------------------------------------------------
def trust_region_max(g, B, radius):
    """argmax of g'v + 0.5 v'Bv over ||v|| <= radius, globally, by More-Sorensen.

    Returns (v, lam, kkt_residual). The maximizer satisfies (lam I - B) v = g with
    lam >= max(0, lam_max(B)) and lam (radius - ||v||) = 0; ||v(lam)|| decreases in lam,
    so one bisection finds it. Works for indefinite B, which is the point.
    """
    # Solve in double regardless of the global dtype -- the eigendecomposition and the
    # secular bisection below need it -- but hand the answer back in the caller's dtype,
    # so a float32 caller does not get a Double tensor that blows up on its next dot
    # product. (It did: study_diagnostics died on `gvec @ v_trs` after the float32 switch.)
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
        # Hard case: g has no component on the top eigenvector.
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


# ---------------------------------------------------------------------------
# The latent adversary
# ---------------------------------------------------------------------------
class LatentAdversary:
    """Worst-case latent transformation inside KL(T_eta # p_0 || p_0) <= rho.

    The transform and its optimizer are kept as state across outer steps: the design moves
    slowly, so the worst case at step t is a good warm start for step t+1. That is the
    timescale separation Theorem 6 asks for, and it is what makes the inner loop cheap
    after the first solve.
    """

    def __init__(self, flow, family='shift', rho=0.02, hidden=32, seed=0, lr=0.05,
                 dtype=None):
        dtype = dtype or torch.get_default_dtype()
        self.flow = flow
        self.family = family
        self.rho = float(rho)
        self.dim = flow.dim
        self.dtype = dtype
        self.hidden = hidden
        self.seed = seed
        self.lr = lr
        self.T = self.new_transform()
        self.opt = torch.optim.Adam(self.T.parameters(), lr=lr)

    def new_transform(self):
        if self.family == 'coupling':
            return CouplingTransform(self.dim, hidden=self.hidden, dtype=self.dtype,
                                     seed=self.seed)
        return TRANSFORMS[self.family](self.dim, dtype=self.dtype)

    def psi(self, loss_fn, phi, T, z):
        """Psi(phi, eta) on a fixed latent batch."""
        x, _ = self.flow(T(z))
        return loss_fn(phi, x).mean()

    # -- inner solve: projected gradient ascent -----------------------------
    def worst_case(self, loss_fn, phi, z, steps=8):
        T, opt = self.T, self.opt
        if isinstance(T, CouplingTransform):
            T.set_kl_samples(z)
        for _ in range(steps):
            opt.zero_grad()
            obj = -self.psi(loss_fn, phi, T, z)
            obj.backward()
            opt.step()
            project_to_ball(T, self.rho)
        return T

    # -- inner solve: exact trust region on the second-order model ----------
    def worst_case_exact(self, loss_fn, phi, z):
        """Only for the shift family, where the ambiguity set is the ball ||mu||<=sqrt(2 rho)."""
        assert self.family == 'shift', 'exact inner solve is defined for the shift family'
        mu = torch.zeros(self.dim, dtype=self.dtype, requires_grad=True)

        def f(m):
            x, _ = self.flow(z + m)
            return loss_fn(phi, x).mean()

        g = torch.autograd.grad(f(mu), mu, create_graph=True)[0]
        B = torch.stack([torch.autograd.grad(g[i], mu, retain_graph=True)[0]
                         for i in range(self.dim)])
        radius = (2 * self.rho) ** 0.5
        v, lam, kkt = trust_region_max(g.detach(), B.detach(), radius)
        T = self.new_transform()
        T.set_flat(v.to(self.dtype))
        return T, {'lam': lam, 'kkt': kkt, 'grad_norm': float(torch.linalg.norm(g.detach())),
                   'B_eig_max': float(torch.linalg.eigvalsh(0.5 * (B + B.T).detach())[-1])}

    def sample_worst(self, T, n, generator=None):
        with torch.no_grad():
            z = torch.randn(n, self.dim, generator=generator, dtype=self.dtype)
            x, _ = self.flow(T(z))
        return x


# ---------------------------------------------------------------------------
# Trust-region Newton on the design
# ---------------------------------------------------------------------------
def objective_derivatives(problem, model, phi, x):
    """(value, grad, Hessian) of area(phi) + lam * mean_x f_w(phi, x).

    The surrogate term is closed form (see TaylorBranchTrunk.design_derivatives); the
    design-only term is a couple of parameters, so autograd on it is free.
    """
    with torch.no_grad():        # the surrogate term is closed form: no graph wanted,
        v, g, H = model.design_derivatives(phi, x)   # and leaving one leaks into phi
    p = phi.detach().clone().requires_grad_(True)
    A = problem.area_term(p)
    gA = torch.autograd.grad(A, p, create_graph=True)[0]
    HA = torch.stack([torch.autograd.grad(gA[i], p, retain_graph=True)[0]
                      for i in range(p.numel())])
    lam = problem.lam
    return float(A) + lam * v, gA.detach() + lam * g, HA.detach() + lam * H


def tr_newton_step(g, H, delta):
    """argmin of g'p + 0.5 p'Hp over ||p|| <= delta.

    Minimizing f is maximizing -f, so this is the same More-Sorensen solve used for the
    adversary, applied to (-g, -H). It is exact even when H is indefinite, which matters:
    the design Hessian of a failure probability is routinely indefinite, and a plain
    Newton step would then head for a saddle or diverge.
    """
    p, lam, kkt = trust_region_max(-g, -H, delta)
    return p, lam, kkt


def latent_dro_newton(problem, flow, model, rho, family='shift', outer_steps=60,
                      inner_steps=6, inner_steps_first=40, inner_lr=0.05, batch=4096,
                      inner_batch=1024, seed=0, delta0=None, delta_max=None,
                      trace_every=0):
    """Algorithm 1 with a trust-region Newton outer step instead of a gradient step.

    Same alternation: solve for the worst-case latent transform, freeze it (Danskin), then
    move the design. The difference is that the design move uses the closed-form gradient
    AND Hessian of the frozen-adversary objective, inside a trust region whose radius
    adapts to how well the quadratic model predicted the actual decrease.
    """
    dtype = next(flow.parameters()).dtype
    g = torch.Generator().manual_seed(seed)
    adv = LatentAdversary(flow, family=family, rho=rho, dtype=dtype, seed=seed,
                          lr=inner_lr)
    phi = problem.init_design().to(dtype)
    lo, hi = problem.design_bounds()
    span = float((hi - lo).min())
    delta = delta0 or 0.10 * span
    delta_max = delta_max or 0.5 * span

    def loss_fn(p, xx):
        return problem.area_term(p) + problem.lam * model(p, xx)

    hist = {'psi': [], 'kkt': [], 'delta': [], 'kl': [], 'phi': [], 't': []}
    for t in range(outer_steps):
        if trace_every and t % trace_every == 0:
            hist['phi'].append(phi.detach().clone())
            hist['t'].append(t)
        z = torch.randn(batch, flow.dim, generator=g, dtype=dtype)
        with torch.enable_grad():
            T = adv.worst_case(loss_fn, phi, z[:inner_batch],
                               steps=inner_steps_first if t == 0 else inner_steps)
        with torch.no_grad():
            x_worst, _ = flow(T(z))
            hist['kl'].append(float(T.kl()))

        val, grad, hess = objective_derivatives(problem, model, phi.detach(),
                                                x_worst)
        step, lam_tr, kkt = tr_newton_step(grad, hess, delta)
        cand = problem.project(phi.detach() + step.to(dtype)).detach()
        with torch.no_grad():
            new_val = float(problem.area_term(cand)
                            + problem.lam * model(cand, x_worst).mean())
        predicted = -(float(grad @ step) + 0.5 * float(step @ hess @ step))
        actual = val - new_val
        ratio = actual / predicted if abs(predicted) > 1e-14 else -1.0
        if ratio > 0:                       # accept only a genuine decrease
            phi = cand
        if ratio < 0.25:
            delta *= 0.25
        elif ratio > 0.75 and float(torch.linalg.norm(step)) > 0.9 * delta:
            delta = min(2 * delta, delta_max)
        hist['psi'].append(val)
        hist['kkt'].append(kkt)
        hist['delta'].append(delta)
    if trace_every:
        hist['phi'].append(phi.detach().clone())
        hist['t'].append(outer_steps)
    return phi, hist


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------
def latent_dro(problem, flow, loss_fn, rho, family='shift', outer_steps=300,
               outer_lr=0.05, inner_steps=6, inner_steps_first=40, inner_lr=0.05,
               batch=1024, inner_batch=512, seed=0, exact_inner=False, phi0=None,
               log_every=0, trace_every=0):
    """Algorithm 1. Returns (phi, history).

    `trace_every` > 0 additionally records the design iterate, so the run's progress can
    be replayed and scored after the fact.
    """
    dtype = next(flow.parameters()).dtype
    g = torch.Generator().manual_seed(seed)
    adv = LatentAdversary(flow, family=family, rho=rho, dtype=dtype, seed=seed,
                          lr=inner_lr)
    phi = (problem.init_design() if phi0 is None else phi0.clone()).to(dtype)
    phi.requires_grad_(True)
    opt = make_optimizer(problem, phi, outer_lr)
    hist = {'psi': [], 'kkt': [], 'lam': [], 'eta_norm': [], 'kl': [], 'phi': [], 't': []}
    for t in range(outer_steps):
        if trace_every and t % trace_every == 0:      # before the step, as above
            hist['phi'].append(phi.detach().clone())
            hist['t'].append(t)
        z = torch.randn(batch, flow.dim, generator=g, dtype=dtype)
        zi = z[:inner_batch]
        with torch.enable_grad():
            if exact_inner:
                T, diag = adv.worst_case_exact(loss_fn, phi.detach(), zi)
                hist['kkt'].append(diag['kkt'])
                hist['lam'].append(diag['lam'])
            else:
                T = adv.worst_case(loss_fn, phi.detach(), zi,
                                   steps=inner_steps_first if t == 0 else inner_steps)
        with torch.no_grad():
            x_worst, _ = flow(T(z))
            hist['eta_norm'].append(float(torch.linalg.norm(T.flat())))
            hist['kl'].append(float(T.kl()))
        opt.zero_grad()
        obj = loss_fn(phi, x_worst).mean()          # Danskin: eta frozen
        obj.backward()
        opt.step()
        with torch.no_grad():
            phi.copy_(problem.project(phi.detach()))
        hist['psi'].append(float(obj.detach()))
        if log_every and t % log_every == 0:
            print(f'    outer {t:4d}  Psi {hist["psi"][-1]:.5f}  |eta| {hist["eta_norm"][-1]:.3f}')
    if trace_every:
        hist['phi'].append(phi.detach().clone())
        hist['t'].append(outer_steps)
    return phi.detach(), hist


def make_optimizer(problem, phi, lr):
    """Adam for box-constrained designs, plain momentum SGD for simplex-constrained ones.

    Adam must not be used under a simplex projection. Its update is sign-like, so when
    every coordinate of the gradient has the same sign the step is a near-uniform
    translation -- and a uniform translation is exactly what projection onto the simplex
    annihilates. The iterate then never leaves its starting point. A plain gradient step
    is proportional to the gradient, so the differences between coordinates, which are
    what the simplex responds to, survive the projection.
    """
    kind = getattr(problem, 'optimizer', 'adam')
    if kind == 'sgd':
        return torch.optim.SGD([phi], lr=lr, momentum=0.9)
    return torch.optim.Adam([phi], lr=lr)


def optimize_design(problem, grad_obj, phi0=None, steps=300, lr=0.05, dtype=None,
                    trace_every=0):
    """Generic projected first-order loop; grad_obj(phi) returns a differentiable scalar.

    With trace_every > 0 the design iterate is recorded every that many steps and returned
    alongside the final design, so a run's convergence can be scored after the fact.
    """
    dtype = dtype or torch.get_default_dtype()
    phi = (problem.init_design() if phi0 is None else phi0.clone()).to(dtype)
    phi.requires_grad_(True)
    opt = make_optimizer(problem, phi, lr)
    trace = []
    for t in range(steps):
        # record BEFORE stepping, so iteration 0 is the initial design every method
        # shares -- recording after the first step makes each curve start somewhere
        # different and the comparison is no longer like for like.
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
