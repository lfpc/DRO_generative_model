"""LCSO: local surrogate fitted to per-sample losses around the incumbent, then a trust-region step.

The problem object needs: dim, link, sample(n, g), loss(phi, x), objective(phi, x),
true_objective(phi), design_bounds(), init_design(), and optionally project(phi),
transform(score, x) (surrogate models an intermediate score), loss_scale, fit_keep.
"""
import torch

from dro.method import trust_region_max
from dro.surrogate import BranchTrunk, TaylorBranchTrunk, TaylorResidualBranchTrunk
from .samplers import draw_designs


class Budget:
    """Counts simulated SAMPLES (not designs): a muon-shield design costs its muons, so
    many designs with few samples each cost the same as few designs with many.

    trace: (samples spent, true objective of the best design so far, chosen on noisy
    observations only).
    """

    def __init__(self, limit):
        self.limit = limit
        self.spent = 0
        self.designs = 0
        self.trace = []

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
    """One design through the simulator: (inputs, per-sample losses, observed objective).

    The observation uses every sample; with problem.fit_keep only a uniform subset of the
    per-sample rows is kept for the fit (memory: the muon shield runs 1e7-1e8 per design).
    """
    x = problem.sample(n_samples, generator)
    with torch.no_grad():
        y = problem.loss(phi, x)
        obs = float(problem.objective(phi, x))
    keep = getattr(problem, 'fit_keep', None)
    if keep is not None and x.shape[0] > keep:
        idx = torch.randperm(x.shape[0], generator=generator)[:keep]
        x, y = x[idx], y[idx]
    return x, y, obs


def _feasible(problem, phi, lo, hi):
    """problem.project if it has one (e.g. a simplex), else clamp to [lo, hi]."""
    if hasattr(problem, 'project'):
        if phi.dim() == 1:
            return problem.project(phi)
        return torch.stack([problem.project(p) for p in phi])
    return torch.minimum(torch.maximum(phi, lo), hi)


def _make_surrogate(problem, kind, seed, rank=8, p=32, hidden=64, res_penalty=0.0):
    """A fresh surrogate.

    kind: 'taylor' (TaylorBranchTrunk), 'taylor-full' (centred quadratic, no residual),
    'taylor-res[:lam[:keep|reset|fold]]' (TaylorResidualBranchTrunk with residual penalty
    lam and that recentring), anything else BranchTrunk ('mlp', 'deeponet').
    rank=None is a full-rank Hessian for the taylor-full/-res models.

    model.y_scale = problem.loss_scale (default 1): the model fits loss / y_scale, and every
    value and derivative read from it is scaled back. A plain attribute, not in the state_dict.
    """
    kind, *opts = kind.split(':')
    res_penalty = float(opts[0]) if opts and opts[0] else res_penalty
    recentre = opts[1] if len(opts) > 1 else 'keep'
    x_dim = problem.sample(2).shape[1]
    if kind in ('taylor-res', 'taylor-full'):
        model = TaylorResidualBranchTrunk(problem.dim, x_dim, rank=rank, hidden=hidden,
                                          p_res=p if kind == 'taylor-res' else 0,
                                          link=problem.link, seed=seed,
                                          res_penalty=res_penalty, recentre=recentre)
    elif kind == 'taylor':
        model = TaylorBranchTrunk(problem.dim, x_dim, rank=rank, hidden=hidden,
                                  link=problem.link, seed=seed)
    else:
        model = BranchTrunk(problem.dim, x_dim, p=p, hidden=hidden, link=problem.link,
                            seed=seed)
    model.y_scale = float(getattr(problem, 'loss_scale', 1.0))
    return model


def _fit(model, designs, X, Y, epochs, seed, lr=3e-3, batch=2048, link='identity',
         opt=None):
    """Fit on per-design inputs: designs (M, D), X (M, N, dx), Y (M, N).

    Each step uses the same random index into every design's own sample block. Pass the
    returned optimiser back in to continue training the same model; the first call
    (opt=None) sets the normalisation.
    """
    if opt is None:
        model.set_norm(designs, X.reshape(-1, X.shape[-1]))
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif hasattr(model, 'set_gate'):      # the trust region may have grown or shrunk
        model.set_gate(designs)
    g = torch.Generator().manual_seed(seed)
    M, N = Y.shape
    Y = Y / model.y_scale
    bce = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(N, generator=g)[:batch]
        s = model.score_batch(designs, X[:, idx])
        y = Y[:, idx]
        loss = bce(s, y) if link == 'sigmoid' else ((s - y) ** 2).mean()
        if getattr(model, 'res_penalty', 0):
            loss = loss + model.res_penalty * model.last_residual.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, opt


def predicted_objective(problem, model, phi, x):
    """The surrogate's objective at phi, through problem.transform if there is one."""
    s = model(phi, x) * model.y_scale
    return problem.transform(s, x).mean() if hasattr(problem, 'transform') else s.mean()


def _derivatives(problem, model, phi, x, need_hessian):
    """(grad, Hessian or None) of the predicted objective in the design.

    Closed form from model.design_derivatives when a Hessian is wanted and there is no
    transform; otherwise the gradient by autograd and no Hessian.
    """
    if hasattr(problem, 'transform'):
        p = phi.detach().clone().requires_grad_(True)
        g, = torch.autograd.grad(predicted_objective(problem, model, p, x), p)
        return g.detach(), None
    if hasattr(model, 'design_derivatives') and need_hessian:
        # detached: the step becomes the next incumbent and must not carry this model's graph
        _, g, H = model.design_derivatives(phi, x)
        k = model.y_scale
        return k * g.detach(), k * H.detach()
    p = phi.detach().clone().requires_grad_(True)
    out = model(p, x).mean() * model.y_scale
    g, = torch.autograd.grad(out, p)
    return g.detach(), None


def lcso(problem, budget, n_samples=20000, surrogate='taylor', step='tr-newton',
         scheme='resample', delta0=0.35, n_designs=None, epochs=250, seed=0,
         min_delta=5e-3, rank=8, resample_factor=8, refresh_frac=0.5,
         sampler='uniform', radial=True, min_per_design=50, verbose=False):
    """Run LCSO until `budget` (total simulated samples) is spent. Returns (best design,
    its true objective).

    scheme:
      nominal   n_designs / resample_factor designs per round, n_samples each.
      resample  n_designs per round, n_samples / resample_factor each, in a rotating buffer
                where refresh_frac of the old designs are replaced every round.
    Both cost the same per round. step: 'tr-newton' (needs a Hessian) or a normalised
    gradient step of length delta. One surrogate is trained on for the whole run.
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
    else:
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
            # keep the newest designs, retire the oldest
            keep = int((1.0 - refresh_frac) * buf_designs.shape[0])
            buf_designs = torch.cat([buf_designs[-keep:], new_designs]) if keep else new_designs
            buf_x = torch.cat([buf_x[-keep:], nx]) if keep else nx
            buf_y = torch.cat([buf_y[-keep:], ny]) if keep else ny
        else:
            buf_designs, buf_x, buf_y = new_designs, nx, ny

        # re-centre an expansion on the incumbent so its derivatives are read there
        # (the first fit sets the centre itself through set_norm)
        if opt is not None and hasattr(model, 'phi_0'):
            model.phi_0 = phi
        model, opt = _fit(model, buf_designs, buf_x, buf_y, epochs, seed,
                          link=problem.link, opt=opt)

        xs = buf_x[-k]                                        # the incumbent's inputs
        grad, hess = _derivatives(problem, model, phi, xs, need_hessian=(step == 'tr-newton'))
        # step in normalised coordinates u = phi / scale, then map back
        scale = (hi - lo) / 2
        g_u = grad * scale
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
        # re-expand instead of freezing at the floor after a run of noisy ratios
        if delta <= min_delta:
            delta = delta0 * 0.25
        budget.record(problem.true_objective(best_phi))
        if verbose:
            print(f'    spent {budget.spent:8d} ({budget.designs:4d} designs) '
                  f'delta {delta:.3f} ratio {ratio:+.2f} '
                  f'best(true) {problem.true_objective(best_phi):.4f}')
    return best_phi, problem.true_objective(best_phi)
