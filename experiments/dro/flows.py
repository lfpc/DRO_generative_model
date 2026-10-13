"""Normalizing flow generator G_theta: R^d -> R^d with base N(0, I) (RealNVP-style).

forward is latent -> data (the differentiable sampling path the inner problem uses),
inverse is data -> latent (for the likelihood).
"""
import torch
import torch.nn as nn


def _mlp(d_in, hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, d_out))


class AffineCoupling(nn.Module):
    """x_free = z_free * exp(s(z_masked)) + t(z_masked), identity on the masked half."""

    def __init__(self, dim, hidden, mask, scale_cap=2.0):
        super().__init__()
        self.net = _mlp(dim, hidden, 2 * dim)
        self.register_buffer('mask', mask)
        self.scale_cap = scale_cap
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, y):
        s, t = self.net(y * self.mask).chunk(2, dim=-1)
        s = torch.tanh(s) * self.scale_cap
        return s * (1 - self.mask), t * (1 - self.mask)

    def forward(self, z):
        """z -> x with log|det dx/dz|."""
        s, t = self._st(z)
        return z * torch.exp(s) + t, s.sum(-1)

    def inverse(self, x):
        """x -> z with log|det dz/dx|."""
        s, t = self._st(x)
        return (x - t) * torch.exp(-s), -s.sum(-1)


class ActNorm(nn.Module):
    """Per-dimension affine layer, data-initialised on the first inverse pass in training."""

    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _data_init(self, x):
        self.shift.copy_(x.mean(0))
        self.log_scale.copy_(x.std(0).clamp_min(1e-6).log())
        self.initialized.fill_(1)

    def forward(self, z):
        return z * torch.exp(self.log_scale) + self.shift, self.log_scale.sum().expand(z.shape[0])

    def inverse(self, x):
        if self.initialized.item() == 0 and self.training:
            self._data_init(x)
        return (x - self.shift) * torch.exp(-self.log_scale), -self.log_scale.sum().expand(x.shape[0])


class Flow(nn.Module):
    """RealNVP-style flow. forward: latent -> data. inverse: data -> latent.

    split='random' conditions each coupling on a random half of the coordinates instead of
    alternating ones; interleave_actnorm puts an ActNorm after every coupling. Both help in
    high dimension (MNIST). scale_cap bounds each coupling's log-scale: a loose cap lets a
    deep flow fit the density well but sample far outside the data, so check sample ranges.
    """

    def __init__(self, dim, n_layers=6, hidden=64, seed=0, split='alternating',
                 interleave_actnorm=False, scale_cap=2.0):
        super().__init__()
        torch.manual_seed(seed)
        self.dim, self.split, self.scale_cap = dim, split, scale_cap
        g = torch.Generator().manual_seed(seed)
        layers, norms = [], []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            if split == 'random':
                mask[torch.randperm(dim, generator=g)[:dim // 2]] = 1.0
            else:
                mask[i % 2::2] = 1.0
            layers.append(AffineCoupling(dim, hidden, mask, scale_cap=scale_cap))
            norms.append(ActNorm(dim))
        self.couplings = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms) if interleave_actnorm else None
        self.actnorm = ActNorm(dim)

    def forward(self, z):
        """Latent -> data, with log|det|."""
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for i, c in enumerate(self.couplings):
            z, ld = c(z)
            logdet = logdet + ld
            if self.norms is not None:
                z, ld = self.norms[i](z)
                logdet = logdet + ld
        x, ld = self.actnorm(z)
        return x, logdet + ld

    def inverse(self, x):
        z, logdet = self.actnorm.inverse(x)
        for i in reversed(range(len(self.couplings))):
            if self.norms is not None:
                z, ld = self.norms[i].inverse(z)
                logdet = logdet + ld
            z, ld = self.couplings[i].inverse(z)
            logdet = logdet + ld
        return z, logdet

    def log_prob(self, x):
        z, logdet = self.inverse(x)
        base = -0.5 * (z ** 2).sum(-1) - 0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi, dtype=x.dtype))
        return base + logdet

    def config(self):
        """Everything needed to rebuild this flow, saved next to its weights."""
        return dict(dim=self.dim, n_layers=len(self.couplings),
                    hidden=self.couplings[0].net[0].out_features,
                    split=self.split, interleave_actnorm=self.norms is not None,
                    scale_cap=self.scale_cap)

    @torch.no_grad()
    def sample(self, n, generator=None, temperature=1.0):
        """Detached samples. The differentiable path is flow(T(z))."""
        z = torch.randn(n, self.dim, generator=generator,
                        dtype=self.actnorm.shift.dtype, device=self.actnorm.shift.device)
        return self.forward(z * temperature)[0]


def load_flow(ckpt, key='flow'):
    """Rebuild a saved flow from ckpt[key] and its ckpt[key + '_config']."""
    flow = Flow(**ckpt[key + '_config'])
    flow.load_state_dict(ckpt[key])
    flow.eval()
    return flow
