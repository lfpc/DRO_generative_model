"""Mean-risk portfolio selection: the standard numerical benchmark of the DRO literature
(Delage & Ye 2010; Mohajerin Esfahani & Kuhn 2018, Sec. 7.2).

    min_w  E[-w'xi] + rho * CVaR_alpha(-w'xi),      w in the simplex

Returns follow a one-factor model, xi_i = psi + zeta_i, with psi a systematic factor shared
by every asset and zeta_i idiosyncratic. Higher-index assets pay more and carry more risk,
so the optimal portfolio is a real trade-off rather than a corner.

CVaR is written in Rockafellar-Uryasev form, CVaR_alpha(L) = min_tau tau + E[(L-tau)_+]/alpha,
so the objective becomes the expectation of a per-sample loss over the decision (w, tau) and
stays convex in it.
"""
import torch


def project_simplex(v):
    """Euclidean projection onto {w >= 0, sum w = 1}."""
    u, _ = torch.sort(v, descending=True)
    css = torch.cumsum(u, 0) - 1.0
    idx = torch.arange(1, v.numel() + 1, dtype=v.dtype)
    k = int((u - css / idx > 0).nonzero().max()) + 1
    return (v - css[k - 1] / k).clamp_min(0.0)


class Portfolio:
    """n assets; the decision is (w, tau) with w the weights and tau the CVaR level."""

    def __init__(self, n_assets=10, rho=10.0, alpha=0.2, dtype=torch.float32):
        self.n_assets, self.rho, self.alpha, self.dtype = n_assets, rho, alpha, dtype
        i = torch.arange(1, n_assets + 1, dtype=dtype)
        self.sys_sd = 0.02                  # systematic factor
        self.mu, self.sd = i * 0.03, i * 0.025      # idiosyncratic mean and spread

    def sample(self, n, generator=None):
        psi = self.sys_sd * torch.randn(n, 1, generator=generator, dtype=self.dtype)
        zeta = self.mu + self.sd * torch.randn(n, self.n_assets, generator=generator,
                                               dtype=self.dtype)
        return psi + zeta

    def loss(self, phi, x):
        """Per-sample mean-CVaR loss. phi = (w, tau); x = (N, n_assets) returns."""
        w, tau = phi[:-1], phi[-1]
        r = x @ w
        return -r + self.rho * (tau + (-r - tau).clamp_min(0.0) / self.alpha)

    def risk(self, phi, x):
        return self.loss(phi, x).mean()

    def project(self, phi):
        return torch.cat([project_simplex(phi[:-1]), phi[-1:]])

    def init_design(self):
        w = torch.full((self.n_assets,), 1.0 / self.n_assets, dtype=self.dtype)
        return torch.cat([w, torch.zeros(1, dtype=self.dtype)])

    def solve(self, x, steps=2000, lr=0.02):
        """Nominal optimum on a sample, by projected gradient descent."""
        phi = self.init_design().requires_grad_(True)
        opt = torch.optim.Adam([phi], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            self.risk(phi, x).backward()
            opt.step()
            with torch.no_grad():
                phi.copy_(self.project(phi.detach()))
        return phi.detach()

    def true_log_prob(self, x):
        """Exact log density of the return model: N(mu, sigma_sys^2 11' + diag(sd^2))."""
        cov = self.sys_sd ** 2 * torch.ones(self.n_assets, self.n_assets, dtype=self.dtype)
        cov = cov + torch.diag(self.sd ** 2)
        return torch.distributions.MultivariateNormal(self.mu, cov).log_prob(x)
