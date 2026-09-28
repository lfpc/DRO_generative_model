
import torch

from dro.method import trust_region_max
from dro.surrogate import BranchTrunk, TaylorBranchTrunk
from .samplers import draw_designs


class Budget:
    """Counts SAMPLES handed to the simulator, not designs.

    This is the accounting that matters, and getting it wrong inverts the conclusion. In
    the muon shield the cost of evaluating a design is dominated by transporting its
    muons, not by the fixed setup: `lcso_resample.py` runs 64 designs at 5e6 muons each,
    so the per-design overhead is negligible against 3.2e8 transports. Charging per design
    therefore makes "many designs, few samples each" look expensive when it is in fact the
    same price -- and that trade is the whole idea of the resample scheme.

    Under this accounting a fixed budget buys either a few designs known precisely or many
    designs known roughly, and which is better depends on what the budget is for. To fit
    the *design dependence* of a surrogate the second is worth far more: per-design noise
    averages out across designs during the fit, while design coverage cannot be recovered
    by sampling one design harder.

    The trace records the true objective of the design the method currently believes is
    best, chosen on its own noisy observations. No method may consult the true objective.
    """

    def __init__(self, limit):
        self.limit = limit                 # total samples
        self.spent = 0
        self.designs = 0
        self.trace = []                    # (samples spent, best true objective so far)

    def charge(self, n_samples, n_designs=1):
        self.spent += n_samples
        self.designs += n_designs
        return self.spent <= self.limit

    def exhausted(self):
        return self.spent >= self.limit

    def remaining(self):
        return max(0, self.limit - self.spent)

    def record(self, value):
        best = value if not self.trace else min(self.trace[-1][1], value)
        self.trace.append((self.spent, best))


def simulate(problem, phi, n_samples, generator):
    """One design handed to the simulator: fresh inputs, per-sample losses, and the
    scalar the simulator reports. The scalar is all a derivative-free method sees; the
    per-sample losses are what a surrogate method additionally exploits, which is the
    real asymmetry between the two families and not an unfair one -- the simulator does
    return per-particle outcomes."""
    x = problem.sample(n_samples, generator)
    with torch.no_grad():
        y = problem.loss(phi, x)
        obs = float(problem.objective(phi, x))
    return x, y, obs


def _feasible(problem, phi, lo, hi):
    """Put a design back in the feasible set.

    A box clamp is right when the constraints are bounds, and wrong when they are not:
    the portfolio's weights live on a simplex, where clamping gives a point that sums to
    something other than one and is not a portfolio at all. Any problem that defines
    `project` gets to say what feasibility means for it.
    """
    if hasattr(problem, 'project'):
        if phi.dim() == 1:
            return problem.project(phi)
        return torch.stack([problem.project(p) for p in phi])
    return torch.minimum(torch.maximum(phi, lo), hi)


def _make_surrogate(problem, kind, seed, rank=8, p=32, hidden=64):
    cls = TaylorBranchTrunk if kind == 'taylor' else BranchTrunk
    kw = dict(rank=rank) if kind == 'taylor' else dict(p=p)
    x_dim = problem.sample(2).shape[1]
    return cls(problem.dim, x_dim, hidden=hidden, link=problem.link, seed=seed, **kw)


def _score_batch(model, designs, Xb):
    """Score every design against its OWN input batch, in one pass.

    designs: (M, D).  Xb: (M, B, dx).  Returns (M, B).

    Scoring design by design in a Python loop costs M forward passes per epoch, which at
    128 designs and 400 epochs is fifty thousand of them and dominates the whole study.
    Both surrogates factor into a design part and an input part, so the per-design inputs
    only change which trunk outputs pair with which branch outputs -- an einsum, not a
    loop.
    """
    dn = (designs - model.design_loc) / model.design_scale             # (M, D)
    xn = (Xb - model.input_loc) / model.input_scale                    # (M, B, dx)
    if hasattr(model, 'Q_raw'):                                        # Taylor quadratic
        tau = model.trunk(xn)                                          # (M, B, 1+D+r)
        a = tau[..., 0]
        c = tau[..., 1:1 + model.D]
        nu = tau[..., 1 + model.D:]
        proj = dn @ model.Q                                            # (M, r)
        lin = torch.einsum('md,mbd->mb', dn, c)
        quad = 0.5 * torch.einsum('mr,mbr->mb', proj ** 2, nu)
        return a + lin + quad + model.beta
    b = model.branch(dn)                                               # (M, p)
    t = model.trunk(xn)                                                # (M, B, p)
    return torch.einsum('mp,mbp->mb', b, t) + model.beta


def _fit(model, designs, X, Y, epochs, seed, lr=3e-3, batch=2048, link='identity',
         opt=None):
    """Fit on per-design input samples, which is what a simulator actually returns.

    `train_surrogate` in dro/surrogate.py assumes one input pool shared by every design --
    reasonable when the pool is drawn once, and it makes the trunk evaluation cheap. Here
    each simulated design carries its own fresh inputs, exactly as `lcso_resample.py`
    holds a separate muon block per design, so the batch is a shared *index* into each
    design's own block.
    """
    if opt is None:                       # first fit: set the normalisation and the state
        model.set_norm(designs, X.reshape(-1, X.shape[-1]))
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    M, N = Y.shape
    bce = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(N, generator=g)[:batch]
        s = _score_batch(model, designs, X[:, idx])
        y = Y[:, idx]
        loss = bce(s, y) if link == 'sigmoid' else ((s - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, opt


def predicted_objective(problem, model, phi, x):
    """What the surrogate says the objective is at `phi`.

    If the problem defines `transform`, the surrogate is modelling an intermediate
    quantity -- a score, say -- and the known function of it is applied here rather than
    being left for the surrogate to learn. That is worth doing whenever the intermediate
    is better matched to the architecture than the objective is: a branch-trunk surrogate
    represents <b(phi), t(x)> exactly, so a score linear in the design is exact, while a
    nonlinear function of that score is only approximated.
    """
    s = model(phi, x)
    return problem.transform(s, x).mean() if hasattr(problem, 'transform') else s.mean()


def _derivatives(problem, model, phi, x, need_hessian):
    """Gradient (and Hessian) of the surrogate's predicted objective in the design.

    The Taylor surrogate has both in closed form. For the branch network the gradient
    comes from autograd; its Hessian is not requested, which is exactly why the
    trust-region variant is paired with the Taylor surrogate.
    """
    if hasattr(problem, 'transform'):          # closed forms do not apply through it
        p = phi.detach().clone().requires_grad_(True)
        g, = torch.autograd.grad(predicted_objective(problem, model, p, x), p)
        return g.detach(), None
    if hasattr(model, 'design_derivatives') and need_hessian:
        # Detach: these come back attached to the surrogate's parameters, and the step
        # built from them becomes the next incumbent, which becomes the centre of the next
        # round's designs. Left attached, the next fit tries to backward through the
        # previous model's graph.
        _, g, H = model.design_derivatives(phi, x)
        return g.detach(), H.detach()
    p = phi.detach().clone().requires_grad_(True)
    out = model(p, x).mean()
    g, = torch.autograd.grad(out, p)
    return g.detach(), None


def lcso(problem, budget, n_samples=20000, surrogate='taylor', step='tr-newton',
         scheme='resample', delta0=0.35, n_designs=None, epochs=250, seed=0,
         min_delta=5e-3, rank=8, resample_factor=8, refresh_frac=0.5,
         sampler='uniform', radial=True, min_per_design=50, verbose=False):
    """LCSO to exhaustion of the budget, where the budget is total simulated samples.

    `scheme` is the choice this study is really about, and both arms cost the same:

      nominal   few designs, each with the full `n_samples`. Every design is measured
                precisely, which is what you want if you intend to *trust* an individual
                measurement -- and what a derivative-free baseline needs.
      resample  `resample_factor` times as many designs, each with `n_samples` divided by
                that factor, held in a rotating buffer of which a fraction is re-simulated
                at fresh designs every iteration. Each design is measured badly; the fit
                does not care, because the design dependence is identified across designs
                and the per-design noise averages out, while design coverage is something
                no amount of sampling one design can recover.

    The measurement in optb/gradcheck.py is what decides between them: the surrogate
    gradient needs ~4D designs to align with the truth, and below 2D it is uncorrelated or
    worse. At a fixed sample budget only the resample arm can afford 4D designs once D is
    large.
    """
    g = torch.Generator().manual_seed(1000 + seed)
    lo, hi = problem.design_bounds()
    phi = problem.init_design().clone()
    delta = delta0

    if n_designs is None:
        n_designs = max(8, 4 * problem.dim)
    if scheme == 'resample':
        per_design = max(min_per_design, n_samples // resample_factor)
        k_round = n_designs
    else:                                    # nominal: full samples, proportionally fewer
        per_design = n_samples
        k_round = max(4, n_designs // resample_factor)

    buf_designs = buf_x = buf_y = None
    model, opt = _make_surrogate(problem, surrogate, seed, rank=rank), None

    x0, y0, f_obs = simulate(problem, phi, per_design, g)
    budget.charge(per_design)
    best_phi, best_obs = phi.clone(), f_obs
    budget.record(problem.true_objective(best_phi))

    while not budget.exhausted():
        k = min(k_round, max(1, budget.remaining() // per_design))
        if k < 2 or budget.remaining() < 2 * per_design:
            break
        u = draw_designs(sampler, k, problem.dim, g, radial=radial).to(phi.dtype)
        new_designs = _feasible(problem, phi + delta * u * (hi - lo) / 2, lo, hi).detach()
        new_designs[0] = phi                                  # pin the incumbent
        nx, ny, obs = [], [], []
        for i in range(k):
            xi, yi, oi = simulate(problem, new_designs[i], per_design, g)
            nx.append(xi)
            ny.append(yi)
            obs.append(oi)
        budget.charge(k * per_design, k)
        f_obs = obs[0]
        nx, ny = torch.stack(nx), torch.stack(ny)

        j = int(torch.tensor(obs).argmin())
        if obs[j] < best_obs:
            best_obs, best_phi = obs[j], new_designs[j].clone()

        if scheme == 'resample' and buf_designs is not None:
            # rotate: keep the newest designs, retire the oldest, as lcso_resample.py
            # re-simulates half the buffer at fresh designs every few epochs
            keep = int((1.0 - refresh_frac) * buf_designs.shape[0])
            buf_designs = torch.cat([buf_designs[-keep:], new_designs]) if keep else new_designs
            buf_x = torch.cat([buf_x[-keep:], nx]) if keep else nx
            buf_y = torch.cat([buf_y[-keep:], ny]) if keep else ny
        else:
            buf_designs, buf_x, buf_y = new_designs, nx, ny

        # One surrogate for the whole run, trained on as the buffer rotates -- which is
        # what `lcso_resample.py` does, and the reason the buffer rotates at all. Rebuilding
        # it every round throws away everything it learned, so with few epochs per round it
        # never converges, and it makes each round cost a full fit.
        model, opt = _fit(model, buf_designs, buf_x, buf_y, epochs, seed,
                          link=problem.link, opt=opt)

        xs = buf_x[-k]                                        # the incumbent's inputs
        grad, hess = _derivatives(problem, model, phi, xs, need_hessian=(step == 'tr-newton'))
        # Take the step in NORMALISED design coordinates, u = phi / scale, and map back.
        # A step of fixed length in raw coordinates is isotropic in units that have no
        # reason to be comparable: on the portfolio the gradient in tau is 60x the
        # gradient in a weight, so a normalised raw step goes almost entirely into tau and
        # the weights barely move. The surrogate already works in these coordinates.
        scale = (hi - lo) / 2
        g_u = grad * scale                       # chain rule: dF/du = dF/dphi * scale
        if step == 'tr-newton' and hess is not None:
            h_u = hess * scale.unsqueeze(1) * scale.unsqueeze(0)
            p_u, _, _ = trust_region_max(-g_u, -h_u, delta)
        else:
            p_u = -g_u / max(float(torch.linalg.norm(g_u)), 1e-12) * delta
        p_step = p_u * scale
        cand = _feasible(problem, phi + p_step, lo, hi).detach()

        if budget.remaining() < per_design:
            break
        _, _, f_cand = simulate(problem, cand, per_design, g)
        budget.charge(per_design)
        with torch.no_grad():
            pred_red = float(predicted_objective(problem, model, phi, xs)
                             - predicted_objective(problem, model, cand, xs))
        act_red = f_obs - f_cand

        ratio = act_red / pred_red if abs(pred_red) > 1e-12 else (1.0 if act_red > 0 else -1.0)
        if act_red > 0:
            phi = cand
            if f_cand < best_obs:
                best_obs, best_phi = f_cand, cand.clone()
        if ratio < 0.25:
            delta *= 0.5
        elif ratio > 0.75:
            delta = min(delta * 2.0, 1.0)
        # Re-expand rather than ratchet down: a run of uninformative ratios early on --
        # which noisy observations produce -- otherwise collapses delta to its floor in a
        # few rounds and freezes the method there for the rest of the budget.
        if delta <= min_delta:
            delta = delta0 * 0.25
        budget.record(problem.true_objective(best_phi))
        if verbose:
            print(f'    spent {budget.spent:8d} ({budget.designs:4d} designs) '
                  f'delta {delta:.3f} ratio {ratio:+.2f} '
                  f'best(true) {problem.true_objective(best_phi):.4f}')
    return best_phi, problem.true_objective(best_phi)
