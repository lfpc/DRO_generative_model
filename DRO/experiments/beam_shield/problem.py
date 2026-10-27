"""Magnetic shield toy: sweep a heavy-tailed particle flux away from a detector.

    min_B  weight(B) + lambda * P( |y_det(B, x)| < R )

A small stand-in for the muon shield: sixteen magnet sections, each with its own field,
bend particles clear of a detector, and field costs weight. Each section gives a kick
0.3 B L / p (GeV, T, m), so the fast tail of the momentum spectrum sets the design.

Random variable: x = (p, p_t, y0, theta0) in R^4; (log p, log p_t) jointly normal, so p and
p_t are lognormal. Exact density by change of variables.
Decision variable: field of each section, design_dim = 16, boxed by |B| <= b_max.
"""
import math

import torch


class BeamShield:
    def __init__(self, n_sec=16, sec_len=0.5, drift=10.0, radius=1.5, lam=10.0,
                 smooth=0.1, b_max=2.0, dtype=torch.float32):
        self.n_sec, self.sec_len, self.drift, self.radius = n_sec, sec_len, drift, radius
        self.lam, self.smooth, self.b_max, self.dtype = lam, smooth, b_max, dtype
        self.input_dim, self.design_dim = 4, n_sec
        self.weight0 = n_sec * sec_len                    # weight of a uniform 1 T shield
        # (log p, log p_t, y0, theta0): 30 GeV median spectrum, entering near the axis
        self.mean = torch.tensor([math.log(30.0), math.log(0.5), 0.0, 0.0], dtype=dtype)
        sd = torch.tensor([0.80, 0.55, 0.05, 0.004], dtype=dtype)
        corr = torch.eye(4, dtype=dtype)
        corr[0, 1] = corr[1, 0] = 0.65                    # faster particles carry more p_t
        corr[2, 3] = corr[3, 2] = -0.30                   # entering high means angled back
        self.cov = corr * sd.unsqueeze(0) * sd.unsqueeze(1)
        self._log = torch.distributions.MultivariateNormal(self.mean, self.cov)

    def sample(self, n, generator=None):
        e = torch.randn(n, 4, generator=generator, dtype=self.dtype)
        z = self.mean + e @ torch.linalg.cholesky(self.cov).T
        return torch.cat([z[:, :2].exp(), z[:, 2:]], 1)

    def true_log_prob(self, x):
        """Normal in the log coordinates, minus the change-of-variables Jacobian."""
        z = torch.cat([x[:, :2].clamp_min(1e-12).log(), x[:, 2:]], 1)
        return self._log.log_prob(z) - z[:, :2].sum(1)

    def trajectory(self, phi, x):
        """Transverse position at the detector plane, shape (n,): paraxial tracking + drift."""
        p = x[:, 0].clamp_min(0.5)
        y, th = x[:, 2], x[:, 3] + x[:, 1] / p
        for j in range(self.n_sec):
            th = th + 0.3 * phi[j] * self.sec_len / p
            y = y + th * self.sec_len
        return y + th * self.drift

    def loss(self, phi, x):
        """Per-sample cost: stored field energy plus a sigmoid-smoothed detector hit."""
        weight = (phi ** 2).sum() * self.sec_len / self.weight0
        y = self.trajectory(phi, x)
        return weight + self.lam * torch.sigmoid((self.radius - y.abs()) / self.smooth)

    def risk(self, phi, x):
        return self.loss(phi, x).mean()

    def project(self, phi):
        return phi.clamp(-self.b_max, self.b_max)

    def init_design(self):
        return torch.full((self.n_sec,), 0.3, dtype=self.dtype)

    def solve(self, x, steps=3000, lr=0.01):
        """Nominal optimum on a sample, by projected momentum SGD."""
        phi = self.init_design().requires_grad_(True)
        opt = torch.optim.SGD([phi], lr=lr, momentum=0.9)
        for _ in range(steps):
            opt.zero_grad()
            self.risk(phi, x).backward()
            opt.step()
            with torch.no_grad():
                phi.copy_(self.project(phi.detach()))
        return phi.detach()
