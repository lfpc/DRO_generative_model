"""Series system of short columns, with the loads observed on a low-dimensional manifold.

    min_{b, h}  area(b, h) + lambda * P( min_j g_j < 0 )

    g_j = 1 - 4 M1 s1_j / (b_j h_j^2 sy) - 4 M2 s2_j / (b_j^2 h_j sy)
            - ( F s3_j / (b_j h_j sy) )^2

The short column under an axial load and biaxial bending is the standard reliability-based
design optimisation benchmark, and that limit state is the textbook one (Kuschel & Rackwitz
1997; reused throughout the RBDO and surrogate-assisted RBDO literature, e.g. Moustapha &
Sudret 2019). Five columns in series: the structure fails if any one of them does, so the
system limit state is the minimum, which is what makes the failure region a union of five
tails rather than one smooth set.

Random variable: twelve log sensor readings in R^12 -- three redundant sensors each for the
two bending moments, the axial load and the yield stress. All twelve are driven by only
THREE latent factors (load intensity, load direction, material batch) plus small independent
sensor noise, so the data concentrate near a 3-manifold embedded in 12 dimensions. The exact
density is available by quadrature over those three latents, so KL against the generator is
measurable.

Decision variable: the width and depth of each of the five columns, design_dim = 10.

Why this problem earns its place. It is the case the whole method is aimed at and the other
benchmarks do not cover: the ambient dimension is four times the intrinsic one, and the cost
is driven by a tail event. A transport ball in R^12 can move a sample straight off the
manifold -- three sensors reading the same load disagreeing by more than their noise, which
is not a load case but a broken instrument -- while a perturbation of the generator's latent
stays on it by construction.
"""
import torch


class ShortColumn:
    def __init__(self, n_cols=5, sensor_sd=0.05, lam=10.0, smooth=0.05, seed=0,
                 dtype=torch.float32):
        self.n_cols, self.sensor_sd, self.lam, self.smooth, self.dtype = \
            n_cols, sensor_sd, lam, smooth, dtype
        self.n_latent, self.input_dim, self.design_dim = 3, 12, 2 * n_cols
        # kN, kNm, kPa -- the benchmark's nominal load case, which sits close to the limit
        self.mu = torch.tensor([250.0, 125.0, 2500.0, 40000.0], dtype=dtype)
        g = torch.Generator().manual_seed(seed)
        self.share = 0.85 + 0.3 * torch.rand(3, n_cols, generator=g, dtype=dtype)
        self.bounds = (0.15, 0.80)
        self.area0 = n_cols * 0.25 * 0.50

    # -- the data ---------------------------------------------------------
    # Four physical quantities driven by three latent factors, so they are genuinely
    # dependent: the two moments share a load-intensity factor, and the yield stress moves
    # against the direction factor because a batch that resists bending one way is weaker
    # the other. Three sensors read each quantity, giving twelve observations of a
    # three-dimensional state -- the manifold this problem is here to test.
    LOADINGS = torch.tensor([[0.35, 0.00, 0.00],          # log M1
                             [0.21, 0.28, 0.00],          # log M2, shares the intensity
                             [0.00, 0.00, 0.25],          # log F
                             [0.00, -0.12, 0.00]])        # log sigma_y, against direction

    def _mean_cov(self):
        """x = mean + A u + sd * eps is exactly Gaussian, with a rank-3 covariance lifted
        off singularity only by the sensor noise. That near-degeneracy IS the manifold:
        the twelve-dimensional law has three directions of real variation and nine of pure
        instrument error, and the flow has to find them."""
        A = self.LOADINGS.to(self.dtype).repeat_interleave(3, dim=0)      # (12, 3)
        cov = A @ A.T + (self.sensor_sd ** 2) * torch.eye(self.input_dim, dtype=self.dtype)
        return self.mu.log().repeat_interleave(3), cov, A

    def sample(self, n, generator=None):
        mean, _, A = self._mean_cov()
        u = torch.randn(n, self.n_latent, generator=generator, dtype=self.dtype)
        return mean + u @ A.T + self.sensor_sd * torch.randn(
            n, self.input_dim, generator=generator, dtype=self.dtype)

    def true_log_prob(self, x):
        """Exact, by the linearity above -- no quadrature and no Monte-Carlo noise in the
        number the generator is validated against."""
        mean, cov, _ = self._mean_cov()
        return torch.distributions.MultivariateNormal(mean, cov).log_prob(x)

    # -- the structure ----------------------------------------------------
    def limit_state(self, phi, x):
        """g for every column, shape (n, n_cols). Positive is safe."""
        q = x.reshape(*x.shape[:-1], 4, 3).mean(-1).exp()        # sensor groups -> physics
        M1, M2, F, sy = (q[..., k:k + 1] for k in range(4))
        b, h = phi[:self.n_cols].unsqueeze(0), phi[self.n_cols:].unsqueeze(0)
        s1, s2, s3 = self.share
        return (1.0 - 4 * M1 * s1 / (b * h ** 2 * sy) - 4 * M2 * s2 / (b ** 2 * h * sy)
                - (F * s3 / (b * h * sy)) ** 2)

    def loss(self, phi, x):
        """Per-sample cost: material plus a smoothed system failure indicator.

        The indicator is smoothed rather than counted so that the objective has a gradient
        at all -- a hard indicator gives zero gradient almost everywhere and makes both the
        surrogate fit and the adversary's ascent meaningless.
        """
        b, h = phi[:self.n_cols], phi[self.n_cols:]
        area = (b * h).sum() / self.area0
        g = self.limit_state(phi, x).min(-1).values
        return area + self.lam * torch.sigmoid(-g / self.smooth)

    def risk(self, phi, x):
        return self.loss(phi, x).mean()

    def project(self, phi):
        return phi.clamp(*self.bounds)

    def init_design(self):
        return torch.cat([torch.full((self.n_cols,), 0.30, dtype=self.dtype),
                          torch.full((self.n_cols,), 0.55, dtype=self.dtype)])

    def solve(self, x, steps=2000, lr=0.002):
        phi = self.init_design().requires_grad_(True)
        opt = torch.optim.SGD([phi], lr=lr, momentum=0.9)
        for _ in range(steps):
            opt.zero_grad()
            self.risk(phi, x).backward()
            opt.step()
            with torch.no_grad():
                phi.copy_(self.project(phi.detach()))
        return phi.detach()
