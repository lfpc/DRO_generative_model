"""Latent transformations T_eta and their KL to the base p_0 = N(0, I).

For an invertible generator, KL(Q_eta || p_theta) = KL(T_eta # p_0 || p_0), so the
ambiguity set is imposed in latent space, where the KL is closed form for these families.
"""
import torch
import torch.nn as nn


class ShiftTransform(nn.Module):
    """T(z) = z + mu.  KL = 0.5 ||mu||^2."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        return z + self.mu

    def kl(self):
        return 0.5 * (self.mu ** 2).sum()

    def flat(self):
        return self.mu

    @torch.no_grad()
    def set_flat(self, v):
        self.mu.copy_(v.reshape(self.mu.shape))


class DiagAffineTransform(nn.Module):
    """T(z) = mu + exp(s) * z.  KL = 0.5 sum_i (e^{2 s_i} - 1 - 2 s_i + mu_i^2)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim))
        self.s = nn.Parameter(torch.zeros(dim))

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


class FullAffineTransform(nn.Module):
    """T(z) = mu + (I + L) z.  With S = I + L: KL = 0.5 (tr(SS') - d - logdet(SS') + ||mu||^2).

    Can rotate as well as rescale, i.e. change the correlations between inputs.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim))
        self.L = nn.Parameter(torch.zeros(dim, dim))
        self.register_buffer('eye', torch.eye(dim))

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


def project_to_ball(transform, rho, tol=1e-10, max_iter=60):
    """Scale eta toward 0 until KL(eta) <= rho, by bisection on the scale.

    KL(t eta) increases with t >= 0 for every family here, so this is exact along the ray
    (the Euclidean projection for the shift family).
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
              'full-affine': FullAffineTransform}
