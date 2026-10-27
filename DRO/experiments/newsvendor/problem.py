"""Multi-item newsvendor with multimodal, correlated demand.

    min_{q >= 0, tau}  E[C(q, d)] + rho * CVaR_alpha(C(q, d)),
    C(q, d) = sum_i  b_i (d_i - q_i)_+ + h_i (q_i - d_i)_+

As in Hanasusanto, Kuhn, Wallace & Zymler (Math. Prog. 2015). The per-item cost is
separable; only the CVaR term makes the joint demand distribution matter.

Random variable: demand d in R^8, a mixture of well-separated correlated Gaussians (one per
market regime). Exact density.
Decision variable: order quantities q >= 0 and the CVaR level tau (Rockafellar-Uryasev),
design_dim = n_items + 1.
"""
import torch


class Newsvendor:
    def __init__(self, n_items=8, n_modes=3, rho=1.0, alpha=0.1, spread=0.35, seed=0,
                 dtype=torch.float32):
        self.n_items, self.n_modes, self.rho, self.alpha, self.dtype = \
            n_items, n_modes, rho, alpha, dtype
        self.input_dim, self.design_dim = n_items, n_items + 1
        g = torch.Generator().manual_seed(seed)
        i = torch.arange(n_items, dtype=dtype)
        # one cluster per regime, plus a shared low-rank correlation within a regime
        self.centres = 7.0 + 3.0 * torch.stack(
            [torch.sin(0.7 * i + 2.1 * k) + 1.4 * k for k in range(n_modes)])
        self.log_w = torch.log(torch.full((n_modes,), 1.0 / n_modes, dtype=dtype))
        a = 0.6 * torch.randn(n_items, 2, generator=g, dtype=dtype)
        self.cov = a @ a.T + (spread ** 2) * torch.eye(n_items, dtype=dtype)
        self.b = 3.0 + 0.5 * i                       # shortage cost, rising with the index
        self.h = 1.0 + 0.0 * i                       # holding cost

    def _mix(self):
        return torch.distributions.MultivariateNormal(self.centres, self.cov)

    def sample(self, n, generator=None):
        k = torch.multinomial(self.log_w.exp(), n, replacement=True, generator=generator)
        e = torch.randn(n, self.n_items, generator=generator, dtype=self.dtype)
        L = torch.linalg.cholesky(self.cov)
        return self.centres[k] + e @ L.T

    def true_log_prob(self, x):
        return torch.logsumexp(self._mix().log_prob(x.unsqueeze(1)) + self.log_w, dim=1)

    def cost(self, q, x):
        """Total cost per sample, before the risk measure."""
        short = (x - q).clamp_min(0.0)
        over = (q - x).clamp_min(0.0)
        return (self.b * short + self.h * over).sum(-1)

    def loss(self, phi, x):
        """Per-sample mean-CVaR loss. phi = (q, tau)."""
        q, tau = phi[:-1], phi[-1]
        c = self.cost(q, x)
        return c + self.rho * (tau + (c - tau).clamp_min(0.0) / self.alpha)

    def risk(self, phi, x):
        return self.loss(phi, x).mean()

    def project(self, phi):
        return torch.cat([phi[:-1].clamp_min(0.0), phi[-1:]])

    def init_design(self):
        return torch.cat([self.centres.mean(0), torch.zeros(1, dtype=self.dtype)])

    def solve(self, x, steps=3000, lr=0.05):
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
