"""Surrogates of the per-sample loss f(phi, x), split into a design part and an input part.

    BranchTrunk                s(phi, x) = <b(phi), t(x)> + beta                  (DeepONet)
    TaylorBranchTrunk          quadratic in the design, low-rank Hessian
    TaylorResidualBranchTrunk  quadratic about one design + gated residual network

The prediction is h(s) with h = sigmoid (link='sigmoid', probabilities) or the identity.
"""
import torch
import torch.nn as nn


def _mlp(d_in, hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, d_out))


class _Surrogate(nn.Module):
    """Normalisation buffers and the link, shared by the three surrogates."""

    def _norm_buffers(self, design_dim, input_dim):
        self.register_buffer('design_loc', torch.zeros(design_dim))
        self.register_buffer('design_scale', torch.ones(design_dim))
        self.register_buffer('input_loc', torch.zeros(input_dim))
        self.register_buffer('input_scale', torch.ones(input_dim))

    def set_norm(self, phi_samples, x_samples):
        with torch.no_grad():
            self.design_loc.copy_(phi_samples.mean(0))
            self.design_scale.copy_(phi_samples.std(0).clamp_min(1e-8))
            self.input_loc.copy_(x_samples.mean(0))
            self.input_scale.copy_(x_samples.std(0).clamp_min(1e-8))
        return self

    def forward(self, phi, x):
        s = self.score(phi, x)
        return torch.sigmoid(s) if self.link == 'sigmoid' else s


class BranchTrunk(_Surrogate):
    def __init__(self, design_dim, input_dim, p=48, hidden=64, link='sigmoid', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.branch = _mlp(design_dim, hidden, p)
        self.trunk = _mlp(input_dim, hidden, p)
        self.beta = nn.Parameter(torch.zeros(1))
        self.link, self.p = link, p
        self._norm_buffers(design_dim, input_dim)
        with torch.no_grad():
            self.branch[-1].weight.div_(p ** 0.5)
            self.trunk[-1].weight.div_(p ** 0.5)

    def score(self, phi, x):
        """phi: (D,) or (B, D).  x: (N, dx).  Returns (N,) or (B, N)."""
        pn = (phi - self.design_loc) / self.design_scale
        xn = (x - self.input_loc) / self.input_scale
        single = pn.dim() == 1
        b = self.branch(pn.unsqueeze(0) if single else pn)      # (B, p)
        t = self.trunk(xn)                                      # (N, p)
        out = b @ t.T + self.beta                               # (B, N)
        return out.squeeze(0) if single else out

    def score_batch(self, designs, Xb):
        """Every design against its OWN inputs: designs (M, D), Xb (M, B, dx) -> (M, B)."""
        b = self.branch((designs - self.design_loc) / self.design_scale)      # (M, p)
        t = self.trunk((Xb - self.input_loc) / self.input_scale)              # (M, B, p)
        return torch.einsum('mp,mbp->mb', b, t) + self.beta


class TaylorBranchTrunk(_Surrogate):
    """s(phi, x) = a(x) + c(x)'u + 0.5 sum_j nu_j(x) (q_j'u)^2,  u = normalised design.

    Only the basis Q (D x r) is learned on the design side, so the design dependence has
    O(D r) parameters instead of a full branch network.
    """

    def __init__(self, design_dim, input_dim, rank=4, hidden=96, link='sigmoid', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.D, self.r, self.link = design_dim, min(rank, design_dim), link
        self.trunk = _mlp(input_dim, hidden, 1 + design_dim + self.r)
        self.Q_raw = nn.Parameter(torch.randn(design_dim, self.r) * 0.5)
        self.beta = nn.Parameter(torch.zeros(1))
        self._norm_buffers(design_dim, input_dim)

    @property
    def Q(self):
        return torch.linalg.qr(self.Q_raw, mode='reduced')[0]

    def score(self, phi, x):
        u = (phi - self.design_loc) / self.design_scale
        single = u.dim() == 1
        if single:
            u = u.unsqueeze(0)
        tau = self.trunk((x - self.input_loc) / self.input_scale)       # (N, 1+D+r)
        a = tau[:, 0]
        c = tau[:, 1:1 + self.D]
        nu = tau[:, 1 + self.D:]
        proj = u @ self.Q                                               # (B, r)
        out = a.unsqueeze(0) + u @ c.T + 0.5 * (proj ** 2) @ nu.T + self.beta
        return out.squeeze(0) if single else out

    def score_batch(self, designs, Xb):
        """Every design against its OWN inputs: designs (M, D), Xb (M, B, dx) -> (M, B)."""
        u = (designs - self.design_loc) / self.design_scale                   # (M, D)
        tau = self.trunk((Xb - self.input_loc) / self.input_scale)            # (M, B, 1+D+r)
        a = tau[..., 0]
        c = tau[..., 1:1 + self.D]
        nu = tau[..., 1 + self.D:]
        proj = u @ self.Q                                                     # (M, r)
        lin = torch.einsum('md,mbd->mb', u, c)
        quad = 0.5 * torch.einsum('mr,mbr->mb', proj ** 2, nu)
        return a + lin + quad + self.beta

    def design_derivatives(self, phi, x):
        """(value, grad, Hessian) of mean_x h(s(phi, x)) w.r.t. phi, in closed form.

        Sigmoid link, with p = sigmoid(s) and m(x) = c + C(x)u the per-input score gradient:
            grad = mean[p(1-p) m],   Hess = mean[p(1-p)(1-2p) m m'] + Q diag(mean[p(1-p) nu]) Q'
        """
        sc = self.design_scale
        u = (phi - self.design_loc) / sc
        tau = self.trunk((x - self.input_loc) / self.input_scale)
        a, c, nu = tau[:, 0], tau[:, 1:1 + self.D], tau[:, 1 + self.D:]
        Q = self.Q                                    # (D, r)
        proj = u @ Q                                  # (r,)
        Cu = (nu * proj) @ Q.T                        # (N, D) = C(x) u
        s = a + c @ u + 0.5 * (nu * proj ** 2).sum(-1) + self.beta
        m = c + Cu                                    # (N, D)
        n = x.shape[0]
        if self.link != 'sigmoid':
            grad_u = m.mean(0)
            hess_u = Q @ torch.diag(nu.mean(0)) @ Q.T
            return float(s.mean().detach()), grad_u / sc, hess_u / (sc[:, None] * sc[None, :])
        p = torch.sigmoid(s)
        w = p * (1 - p)                               # (N,)
        grad_u = (w.unsqueeze(1) * m).mean(0)
        H1 = (m * (w * (1 - 2 * p)).unsqueeze(1)).T @ m / n
        H2 = Q @ torch.diag((w.unsqueeze(1) * nu).mean(0)) @ Q.T
        hess_u = H1 + H2
        return float(p.mean()), grad_u / sc, hess_u / (sc[:, None] * sc[None, :])


class TaylorResidualBranchTrunk(_Surrogate):
    """Taylor expansion about one design phi_c, with a gated network past second order:

        s(phi, x) = a(x) + c(x)'u + 0.5 u'C(x)u + rho(u) <r_b(u), r_t(x)> + beta
        C(x) = Q diag(nu(x)) Q',   rho(u) = (||u||^2 / (D gate_scale^2))^(k/2),   u = (phi - phi_c) / scale

    With k = 4 the residual contributes nothing to the value, gradient or Hessian at phi_c,
    so a, c, C are exactly the per-input value, gradient and Hessian there.
    phi_c is the FIRST design given to set_norm (the pinned incumbent), not the mean.
    rank=None is a full-rank Hessian; p_res=0 drops the residual. The residual's last layer
    starts at zero, so the model starts as the quadratic.
    res_penalty: weight of mean(residual^2) in the fit (optb.lcso._fit).
    recentre: what moving phi_0 does to the residual ('keep', 'reset' or 'fold', see phi_0).
    """

    def __init__(self, design_dim, input_dim, rank=None, p_res=32, hidden=96, res_order=4,
                 link='sigmoid', seed=0, res_penalty=0.0, recentre='keep'):
        super().__init__()
        torch.manual_seed(seed)
        D = design_dim
        self.D, self.link, self.p_res, self.k = D, link, p_res, res_order
        self.res_penalty, self.last_residual = res_penalty, None
        self.recentre = recentre
        self.r = D if rank is None else min(rank, D)
        self.trunk = _mlp(input_dim, hidden, 1 + D + self.r + p_res)
        self.Q_raw = nn.Parameter(torch.randn(D, self.r) * 0.5)
        self.res_branch = _mlp(D, hidden, p_res) if p_res else None
        if p_res:
            with torch.no_grad():
                self.res_branch[-1].weight.zero_()
                self.res_branch[-1].bias.zero_()
        self.beta = nn.Parameter(torch.zeros(1))
        self._norm_buffers(D, input_dim)
        # radius, in u, at which the gate reaches 1
        self.register_buffer('gate_scale', torch.ones(()))

    @property
    def Q(self):
        return torch.linalg.qr(self.Q_raw, mode='reduced')[0]

    def set_gate(self, designs):
        """Gate radius = RMS spread of these designs about the centre. Leaves the polynomial
        (and design_scale) untouched, so the trust region can grow or shrink under it."""
        with torch.no_grad():
            u = (designs - self.design_loc) / self.design_scale
            self.gate_scale.copy_(u.pow(2).sum(-1).mean().div(self.D).sqrt().clamp_min(1e-6))

    @property
    def phi_0(self):
        """The expansion centre. Set it to the incumbent whenever the incumbent moves."""
        return self.design_loc

    @phi_0.setter
    def phi_0(self, phi):
        """Move the expansion centre to phi, re-expanding the quadratic exactly.

        With d the move in normalised units:
            a' = a + c'd + 0.5 sum_j (q_j'd)^2 nu_j,   c' = c + sum_j nu_j (q_j'd) q_j,   nu' = nu
        All linear in the trunk outputs, so it is a matrix folded into the trunk's last layer.
        The residual: 'keep' re-gates it about the new centre; 'reset' zeroes it; 'fold' first
        moves its value, gradient and Q-diagonal curvature at phi into the polynomial, then
        zeroes it.
        """
        with torch.no_grad():
            phi = phi.to(self.design_loc.dtype)
            d = (phi - self.design_loc) / self.design_scale
            Q = self.Q
            qd = Q.T @ d
            D, r = self.D, self.r
            last = self.trunk[-1]
            M = torch.eye(last.out_features, dtype=last.weight.dtype)
            M[0, 1:1 + D] = d
            M[0, 1 + D:1 + D + r] = 0.5 * qd ** 2
            M[1:1 + D, 1 + D:1 + D + r] = Q * qd
            if self.p_res and self.recentre == 'fold':
                rs = slice(1 + D + r, 1 + D + r + self.p_res)
                with torch.enable_grad():
                    gv = self._residual_branch(d)
                    J = torch.autograd.functional.jacobian(self._residual_branch, d)
                    Hs = torch.stack([torch.autograd.functional.hessian(
                        lambda v, i=i: self._residual_branch(v)[i], d)
                        for i in range(self.p_res)])                     # (p, D, D)
                M[0, rs] = gv
                M[1:1 + D, rs] = J.T
                M[1 + D:1 + D + r, rs] = torch.einsum('dj,pde,ej->jp', Q, Hs, Q)
            last.weight.copy_(M @ last.weight)
            last.bias.copy_(M @ last.bias)
            if self.p_res and self.recentre in ('reset', 'fold'):
                self.res_branch[-1].weight.zero_()
                self.res_branch[-1].bias.zero_()
            self.design_loc.copy_(phi)

    def set_norm(self, phi_samples, x_samples):
        super().set_norm(phi_samples, x_samples)
        with torch.no_grad():
            self.design_loc.copy_(phi_samples[0])                  # the expansion centre
        self.set_gate(phi_samples)
        return self

    def _split(self, tau):
        D, r = self.D, self.r
        return (tau[..., 0], tau[..., 1:1 + D], tau[..., 1 + D:1 + D + r],
                tau[..., 1 + D + r:])

    def _residual_branch(self, u):
        """rho(u) r_b(u), shape (..., p_res). Odd k gets an epsilon to stay differentiable at 0."""
        sq = u.pow(2).sum(-1, keepdim=True) / (self.D * self.gate_scale ** 2)
        gate = sq.pow(self.k // 2) if self.k % 2 == 0 else (sq + 1e-12).pow(self.k / 2)
        return self.res_branch(u) * gate

    def score(self, phi, x):
        u = (phi - self.design_loc) / self.design_scale
        single = u.dim() == 1
        if single:
            u = u.unsqueeze(0)
        a, c, nu, rt = self._split(self.trunk((x - self.input_loc) / self.input_scale))
        out = a.unsqueeze(0) + u @ c.T + 0.5 * ((u @ self.Q) ** 2) @ nu.T + self.beta
        if self.p_res:
            out = out + self._residual_branch(u) @ rt.T
        return out.squeeze(0) if single else out

    def score_batch(self, designs, Xb):
        """Every design against its OWN inputs: designs (M, D), Xb (M, B, dx) -> (M, B)."""
        u = (designs - self.design_loc) / self.design_scale
        a, c, nu, rt = self._split(self.trunk((Xb - self.input_loc) / self.input_scale))
        out = (a + torch.einsum('md,mbd->mb', u, c)
               + 0.5 * torch.einsum('mr,mbr->mb', (u @ self.Q) ** 2, nu) + self.beta)
        if self.p_res:
            res = torch.einsum('mp,mbp->mb', self._residual_branch(u), rt)
            self.last_residual = res                   # for the fit's penalty on it
            out = out + res
        return out

    def design_derivatives(self, phi, x):
        """(value, grad, Hessian) of mean_x f(phi, x) w.r.t. phi.

        Identity link: quadratic part in closed form, residual by autograd through the
        design branch against the mean trunk features. Sigmoid link: autograd on the mean.
        Enables grad itself because callers may hold no_grad.
        """
        sc = self.design_scale
        with torch.enable_grad():
            if self.link == 'sigmoid':
                def f(p):
                    return torch.sigmoid(self.score(p, x)).mean()
                p = phi.detach().clone()
                g = torch.autograd.functional.jacobian(f, p)
                H = torch.autograd.functional.hessian(f, p)
                return float(f(p).detach()), g.detach(), H.detach()
            u = ((phi - self.design_loc) / sc).detach()
            with torch.no_grad():
                a, c, nu, rt = self._split(
                    self.trunk((x - self.input_loc) / self.input_scale).mean(0))
            Q = self.Q.detach()
            proj = u @ Q
            value = a + c @ u + 0.5 * (nu * proj ** 2).sum() + self.beta.detach()
            grad_u = c + Q @ (nu * proj)
            hess_u = Q @ torch.diag(nu) @ Q.T
            if self.p_res:
                def res(v):
                    return self._residual_branch(v) @ rt
                value = value + res(u).detach()
                grad_u = grad_u + torch.autograd.functional.jacobian(res, u).detach()
                hess_u = hess_u + torch.autograd.functional.hessian(res, u).detach()
        return (float(value), grad_u.detach() / sc,
                hess_u.detach() / (sc[:, None] * sc[None, :]))
