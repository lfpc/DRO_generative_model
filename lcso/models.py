"""The architectures of the LCSO surrogate of p(hit | u, x) = sigmoid(<B(u), T(x)> + bias), u the
normalized design.

Every architecture is a branch B: R^D -> R^P and a trunk T: R^dx -> R^P, so they share the fit,
the predictions and the derivatives. make(kind, D, dx, ...) builds one; kinds:

  deeponet          B and T are MLPs (hidden, depth, p, act)
  taylor:r          B = [1, u, (Q'u)^2 / 2]: each sample has its own value, gradient and rank-r
                    curvature diag(nu(x)) in a learned shared eigenbasis Q (D x r); r = D: full rank
  taylor-sub:r      B = [1, Q'u, (Q'u)^2 / 2]: the design enters only through r directions
                    (a low-dimensional manifold), gradient included
  taylor-res:r      taylor:r plus a residual MLP gated by (mean u^2)^2 / radius^4, which vanishes
                    with its first three derivatives at u = 0, so the value, gradient and Hessian
                    there are the Taylor part's
  taylor-sub-res:r  taylor-sub:r plus the same gated residual
  taylor-res4:r,    the same with the gate mean(u^4) / radius^4: it also opens along a single
  taylor-sub-res4:r axis, so one parameter at its limit can get a one-sided response
  quad              B = [1, u, u_i u_j (i <= j)]: a full quadratic per sample
  rsm              quad with a constant trunk: a response surface of the hit rate, blind to x
  mlp              an MLP on [u, x]: not separable, the expressivity ceiling (mlp:k: depth k)

Ensemble(models) is the mean logit of several of them, itself a branch-trunk surrogate.

The expected hit rate of a sample X is H(u) = mean_x sigmoid(<B(u), T(x)> + bias); derivatives()
gives its gradient and Hessian exactly, from one pass over X.
"""
import math

import torch
import torch.nn as nn
from torch.func import hessian, jacrev

ACTS = {'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU, 'tanh': nn.Tanh, 'softplus': nn.Softplus}


def mlp(d_in, hidden, depth, d_out, act='silu'):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), ACTS[act]()]
        d = hidden
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class Surrogate(nn.Module):
    """logit(u, x) = <branch(u), trunk(x)> + bias. Subclasses define branch; the trunk is an MLP
    on the standardized x, or a constant (const_trunk)."""
    separable = True          # the logit is a dot product of a design part and a sample part

    def __init__(self, D, dx, P, hidden=64, depth=2, trunk_act='relu', const_trunk=False):
        super().__init__()
        self.D, self.P = D, P
        self.trunk_net = None if const_trunk else mlp(dx, hidden, depth, P, trunk_act)
        self.const = nn.Parameter(torch.zeros(P)) if const_trunk else None
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer('x_loc', torch.zeros(dx))
        self.register_buffer('x_scale', torch.ones(dx))

    def set_norm(self, x):
        self.x_loc.copy_(x.mean(0))
        self.x_scale.copy_(x.std(0).clamp_min(1e-6))

    def trunk(self, x):
        if self.const is not None:
            return self.const.expand(x.shape[0], self.P)
        return self.trunk_net((x - self.x_loc) / self.x_scale)

    def branch(self, u):
        raise NotImplementedError

    def logit_rows(self, U, d, x):
        """Rows: design index d (R,) into U (M, D), samples x (R, dx) -> (R,)."""
        return (self.branch(U[d]) * self.trunk(x)).sum(-1) + self.bias

    def logits(self, U, x):
        """Every design against every sample: (B, N)."""
        return self.branch(U) @ self.trunk(x).T + self.bias


class DeepONet(Surrogate):
    def __init__(self, D, dx, p=16, hidden=64, depth=2, act='silu', trunk_act='relu'):
        super().__init__(D, dx, p, hidden, depth, trunk_act)
        self.branch_net = mlp(D, hidden, depth, p, act)
        with torch.no_grad():
            for net in (self.branch_net, self.trunk_net):
                net[-1].weight.div_(math.sqrt(p))

    def branch(self, u):
        return self.branch_net(u)


class Taylor(Surrogate):
    """A per-sample Taylor expansion in u: B(u) = [1, u or Q'u, (Q'u)^2 / 2], optionally plus a
    gated residual gate(u) * MLP(u).

    The gate is zero, with zero gradient and Hessian, at u = 0, so the value, gradient and Hessian
    there are the Taylor part's coefficients. It is (mean u^2)^2, or with quartic mean(u^4) (which
    also opens it along a single axis), divided by the training designs' radius^4 (set by fit()).

    near > 0 (training only) splits what the two parts learn from: each row's gradient reaches the
    Taylor coefficients with the weight exp(-sqrt(dist4(u)) / near^2), the residual with weight 1.
    The predictions are unchanged."""
    near = 0.0

    def __init__(self, D, dx, rank, sub=False, p_res=0, hidden=64, depth=2, trunk_act='relu', quartic=False):
        r = min(rank, D)
        P = 1 + (r if sub else D) + r + p_res
        super().__init__(D, dx, P, hidden, depth, trunk_act)
        self.r, self.sub, self.p_res, self.quartic = r, sub, p_res, quartic
        self.Q_raw = nn.Parameter(torch.linalg.qr(torch.randn(D, r))[0])
        self.res = mlp(D, hidden, depth, p_res) if p_res else None
        self.register_buffer('gate', torch.ones(()))       # radius (rms, per coordinate) of the training designs
        with torch.no_grad():               # unit-variance logits; start near-linear, as the
            self.trunk_net[-1].weight.div_(math.sqrt(P))        # curvature is amplified by |u|^2
            self.trunk_net[-1].weight[-r - p_res:].mul_(0.1)

    @property
    def Q(self):
        return torch.linalg.qr(self.Q_raw)[0]

    def dist4(self, u):
        """The gate's measure of the distance from u = 0, to the fourth power (smooth at 0)."""
        return u.pow(4).mean(-1) if self.quartic else u.pow(2).mean(-1) ** 2

    def branch(self, u):
        z = u @ self.Q
        feats = [torch.ones_like(u[..., :1]), z if self.sub else u, 0.5 * z ** 2]
        if self.p_res:
            feats.append(self.dist4(u)[..., None] / self.gate ** 4 * self.res(u))
        return torch.cat(feats, -1)

    def logit_rows(self, U, d, x):
        if not (self.training and self.near and self.p_res):
            return super().logit_rows(U, d, x)
        u = U[d]
        f = self.branch(u) * self.trunk(x)
        k = self.P - self.p_res                         # the Taylor features come first
        taylor = f[:, :k].sum(-1)
        w = torch.exp(-self.dist4(u).sqrt() / self.near ** 2)
        taylor = w * taylor + (1 - w) * taylor.detach()  # same value, gradient scaled by w
        return taylor + f[:, k:].sum(-1) + self.bias


class Quad(Surrogate):
    def __init__(self, D, dx, hidden=64, depth=2, trunk_act='relu', const_trunk=False):
        super().__init__(D, dx, 1 + D + D * (D + 1) // 2, hidden, depth, trunk_act, const_trunk)
        i, j = torch.triu_indices(D, D)
        self.register_buffer('i', i)
        self.register_buffer('j', j)
        if not const_trunk:
            with torch.no_grad():
                self.trunk_net[-1].weight.div_(math.sqrt(self.P))
                self.trunk_net[-1].weight[1 + D:].mul_(0.1)

    def branch(self, u):
        return torch.cat([torch.ones_like(u[..., :1]), u, u[..., self.i] * u[..., self.j]], -1)


class MLP(nn.Module):
    """logit(u, x) = MLP([u, standardized x]): no separable structure (the expressivity ceiling).
    Predictions need the network per design and sample; derivatives need autograd
    (derivatives_autograd)."""
    separable = False

    def __init__(self, D, dx, hidden=256, depth=3, act='silu'):
        super().__init__()
        self.D = D
        self.net = mlp(D + dx, hidden, depth, 1, act)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer('x_loc', torch.zeros(dx))
        self.register_buffer('x_scale', torch.ones(dx))

    set_norm = Surrogate.set_norm

    def logit_rows(self, U, d, x):
        return self.net(torch.cat([U[d], (x - self.x_loc) / self.x_scale], -1))[:, 0] + self.bias

    def logits(self, U, x):
        """Every design against every sample: (B, N)."""
        xn = (x - self.x_loc) / self.x_scale
        B, N = U.shape[0], x.shape[0]
        z = torch.cat([U[:, None].expand(B, N, -1), xn[None].expand(B, N, -1)], -1)
        return self.net(z)[..., 0] + self.bias


class Ensemble(Surrogate):
    """The mean logit of several surrogates. If they are all separable, it is itself a
    branch-trunk surrogate (branch and trunk features concatenated, branches scaled by 1/K), so
    derivatives() etc. apply; otherwise only logits and logit_rows."""

    def __init__(self, models):
        torch.nn.Module.__init__(self)
        self.models = nn.ModuleList(models)
        self.separable = all(m.separable for m in models)
        self.D = models[0].D
        self.P = sum(m.P for m in models) if self.separable else None
        self.bias = nn.Parameter(torch.stack([m.bias for m in models]).mean(0).detach())

    def branch(self, u):
        return torch.cat([m.branch(u) for m in self.models], -1) / len(self.models)

    def trunk(self, x):
        return torch.cat([m.trunk(x) for m in self.models], -1)

    def logit_rows(self, U, d, x):
        if self.separable:
            return super().logit_rows(U, d, x)
        return sum(m.logit_rows(U, d, x) for m in self.models) / len(self.models)

    def logits(self, U, x):
        if self.separable:
            return super().logits(U, x)
        return sum(m.logits(U, x) for m in self.models) / len(self.models)


def derivatives_autograd(model, points, u0=None, chunk=2 ** 11):
    """(H, grad, Hessian) of H(u) = sum_w w sigmoid(logit(u, x)) at u0 (default 0), by autograd
    through the whole network, over weighted points (x, w): for non-separable surrogates."""
    dev = next(model.parameters()).device
    u0 = torch.zeros(model.D, device=dev) if u0 is None else u0.to(dev)
    H, g, h = 0.0, 0.0, 0.0
    for x, w in points:
        for c in range(0, len(x), chunk):
            xb, wb = x[c:c + chunk].to(dev), w[c:c + chunk].to(dev)

            def F(u):
                return (torch.sigmoid(model.logits(u[None], xb)[0]).double() * wb).sum()

            H += float(F(u0).detach())
            g = g + jacrev(F)(u0).detach()
            h = h + hessian(F)(u0).detach()
    return H, g.cpu(), (0.5 * (h + h.T)).cpu()


def make(kind, D, dx, p=16, hidden=64, depth=2, act='silu', trunk_act='relu', p_res=16):
    """A surrogate from its kind string, e.g. 'deeponet', 'taylor:8', 'taylor-sub:4', 'taylor-sub-res:4',
    'taylor-res4:8' (residual gated by mean(u^4))."""
    name, _, r = kind.partition(':')
    r = int(r) if r else D
    if name == 'deeponet':
        return DeepONet(D, dx, p, hidden, depth, act, trunk_act)
    if name in ('taylor', 'taylor-sub', 'taylor-res', 'taylor-sub-res', 'taylor-res4', 'taylor-sub-res4'):
        return Taylor(D, dx, r, sub=name.startswith('taylor-sub'), p_res=p_res if '-res' in name else 0,
                      hidden=hidden, depth=depth, trunk_act=trunk_act, quartic=name.endswith('-res4'))
    if name in ('quad', 'rsm'):
        return Quad(D, dx, hidden, depth, trunk_act, const_trunk=name == 'rsm')
    if name == 'mlp':
        return MLP(D, dx, hidden, r if ':' in kind else depth, act)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Predictions and derivatives of H(u) = mean_x sigmoid(logit(u, x))
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, X, U, chunk=2 ** 16, offset=0.0):
    """H at designs U (B, D) over samples X (N, dx): (B,), float64. offset is added to the logit."""
    b = model.branch(U)
    H = torch.zeros(U.shape[0], dtype=torch.float64, device=U.device)
    for c in range(0, X.shape[0], chunk):
        H += torch.sigmoid(b @ model.trunk(X[c:c + chunk]).T + model.bias + offset).double().sum(1)
    return H / X.shape[0]


def derivatives(model, X, u0=None, chunk=2 ** 15):
    """(H, grad, Hessian) at u0 (default 0) over samples X, float64. With w = s(1 - s):

        grad = J' mean[w t],   Hess = J' mean[w (1 - 2s) t t'] J + d2/du2 <B(u), mean[w t]>

    with t = T(x), s = sigmoid(logit) and J = dB/du at u0: exact, one pass over X."""
    dev = model.bias.device                  # X may sit in (pinned) host memory: chunks are moved
    u0 = torch.zeros(model.D, device=dev) if u0 is None else u0.to(dev)
    P = model.P
    Hs = torch.zeros((), dtype=torch.float64, device=dev)
    Tw = torch.zeros(P, dtype=torch.float64, device=dev)
    A = torch.zeros(P, P, dtype=torch.float64, device=dev)
    with torch.no_grad():
        b0 = model.branch(u0[None])[0]
        for c in range(0, X.shape[0], chunk):
            t = model.trunk(X[c:c + chunk].to(dev, non_blocking=True))
            s = torch.sigmoid(t @ b0 + model.bias)
            w = s * (1 - s)
            Hs += s.double().sum()
            Tw += (w[:, None] * t).double().sum(0)
            A += ((t * (w * (1 - 2 * s))[:, None]).T @ t).double()       # float32 per chunk, summed in float64
    n = X.shape[0]
    Hs, Tw, A = Hs / n, Tw / n, A / n

    def br(u):
        return model.branch(u[None])[0]

    J = jacrev(br)(u0).detach().double()                              # (P, D)
    Hb = hessian(lambda u: br(u) @ Tw.float())(u0).detach().double()
    hess = J.T @ A @ J + Hb
    return float(Hs), (J.T @ Tw).cpu(), (0.5 * (hess + hess.T)).cpu()


def features(model, X, chunk=2 ** 16):
    """Trunk features of X, for repeated evaluations (worst-case searches)."""
    with torch.no_grad():
        return torch.cat([model.trunk(X[c:c + chunk]) for c in range(0, X.shape[0], chunk)])


def n_params(model):
    return sum(q.numel() for q in model.parameters())
