"""Latent transformations T_eta and the ambiguity set they induce.

Proposition 1: for a diffeomorphic generator, D_f(Q_eta || p_theta) = D_f(T_eta # p_0 || p_0).
So the ambiguity constraint is imposed here, in latent space, where p_0 = N(0, I) and the
KL is closed form for the affine families.
"""
import torch
import torch.nn as nn


class ShiftTransform(nn.Module):
    """T_eta(z) = z + mu.  KL(T#p0 || p0) = 0.5 ||mu||^2, so the ambiguity set is a ball."""

    n_groups = 1

    def __init__(self, dim, dtype=None):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim, dtype=dtype))

    def forward(self, z):
        return z + self.mu

    def kl(self):
        return 0.5 * (self.mu ** 2).sum()

    def flat(self):
        return self.mu

    @torch.no_grad()
    def set_flat(self, v):
        self.mu.copy_(v.reshape(self.mu.shape))

    @torch.no_grad()
    def zero(self):
        self.mu.zero_()


class DiagAffineTransform(nn.Module):
    """T_eta(z) = mu + exp(s) * z.  KL = 0.5 sum_i (e^{2 s_i} - 1 - 2 s_i + mu_i^2).

    Expresses both a location shift and a change of dispersion of the latent, i.e. both a
    move of the input distribution and a change of how heavy its tails are.
    """

    n_groups = 2

    def __init__(self, dim, dtype=None):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim, dtype=dtype))
        self.s = nn.Parameter(torch.zeros(dim, dtype=dtype))

    def forward(self, z):
        return self.mu + torch.exp(self.s) * z

    def kl(self):
        return 0.5 * (torch.exp(2 * self.s) - 1 - 2 * self.s + self.mu ** 2).sum()

    def flat(self):
        return torch.cat([self.mu, self.s])

    @torch.no_grad()
    def set_flat(self, v):
        self.mu.copy_(v[:self.dim])
        self.s.copy_(v[self.dim:])

    @torch.no_grad()
    def zero(self):
        self.mu.zero_()
        self.s.zero_()


class FullAffineTransform(nn.Module):
    """T_eta(z) = mu + (I + L) z, with L a full matrix.

    The pushforward is N(mu, S S') with S = I + L, so

        KL = 0.5 (tr(S S') - d - logdet(S S') + ||mu||^2),

    still closed form. Unlike the diagonal family this can rotate as well as rescale, so
    it can express a change in the *dependence* between inputs -- a correlation breakdown
    -- whichever way the generator happened to orient its latent coordinates.
    """

    n_groups = 2

    def __init__(self, dim, dtype=None):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim, dtype=dtype))
        self.L = nn.Parameter(torch.zeros(dim, dim, dtype=dtype))
        self.register_buffer('eye', torch.eye(dim, dtype=dtype))

    def S(self):
        return self.eye + self.L

    def forward(self, z):
        return self.mu + z @ self.S().T

    def kl(self):
        S = self.S()
        C = S @ S.T
        sign, logdet = torch.linalg.slogdet(C)
        logdet = torch.where(sign > 0, logdet, torch.full_like(logdet, -1e6))
        return 0.5 * (torch.diagonal(C).sum() - self.dim - logdet + (self.mu ** 2).sum())

    def flat(self):
        return torch.cat([self.mu, self.L.reshape(-1)])

    @torch.no_grad()
    def set_flat(self, v):
        self.mu.copy_(v[:self.dim])
        self.L.copy_(v[self.dim:].reshape(self.dim, self.dim))

    @torch.no_grad()
    def zero(self):
        self.mu.zero_()
        self.L.zero_()


class CouplingTransform(nn.Module):
    """One affine coupling layer on the latent, eta = its weights.

    A richer ambiguity set: it can reweight regions of the latent (and hence rebalance
    mixture components of the input distribution) rather than only translate or rescale
    them. The price is that KL(T#p0 || p0) has no closed form and is estimated by Monte
    Carlo using the layer's own log-determinant:

        KL = E_z[ log p_0(z) - log|det J_T(z)| - log p_0(T(z)) ].
    """

    n_groups = 1

    def __init__(self, dim, hidden=32, scale_cap=1.0, dtype=None, seed=0):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        torch.manual_seed(seed)
        self.dim = dim
        mask = torch.zeros(dim, dtype=dtype)
        mask[0::2] = 1.0
        self.register_buffer('mask', mask)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * dim)).to(dtype)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale_cap = scale_cap
        self._z_kl = None

    def _st(self, z):
        s, t = self.net(z * self.mask).chunk(2, dim=-1)
        s = torch.tanh(s) * self.scale_cap
        return s * (1 - self.mask), t * (1 - self.mask)

    def forward(self, z):
        s, t = self._st(z)
        return z * torch.exp(s) + t

    def set_kl_samples(self, z):
        self._z_kl = z

    def kl(self):
        z = self._z_kl
        s, _ = self._st(z)
        logdet = s.sum(-1)
        tz = self.forward(z)
        lp0_z = -0.5 * (z ** 2).sum(-1)
        lp0_tz = -0.5 * (tz ** 2).sum(-1)
        return (lp0_z - logdet - lp0_tz).mean()

    def flat(self):
        return torch.cat([p.reshape(-1) for p in self.net.parameters()])

    @torch.no_grad()
    def zero(self):
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)


def project_to_ball(transform, rho, tol=1e-10, max_iter=60):
    """Scale eta toward 0 until KL(eta) <= rho.

    KL(t * eta) is increasing in t >= 0 for every family here (each term is convex with
    its minimum at eta = 0), so a bisection on the scale is an exact projection along the
    ray -- which is the true Euclidean projection for the shift family, whose feasible set
    is a ball, and a valid feasible retraction otherwise.
    """
    with torch.no_grad():
        if float(transform.kl()) <= rho:
            return 1.0
        v = transform.flat().clone()
        lo, hi = 0.0, 1.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            transform.set_flat(mid * v)
            if float(transform.kl()) > rho:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        transform.set_flat(lo * v)
        return lo


TRANSFORMS = {'shift': ShiftTransform, 'affine': DiagAffineTransform,
              'full-affine': FullAffineTransform, 'coupling': CouplingTransform}
