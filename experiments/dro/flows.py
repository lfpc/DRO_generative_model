"""Normalizing flow generator G_theta: R^d -> R^d with base N(0, I).

The flow is the paper's G_theta. What the method needs from it is exactly two things:
a differentiable sampling path z -> x (for the pathwise gradient of the inner problem)
and invertibility (so Proposition 1 applies and the latent divergence equals the data
divergence).
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
    """Per-dimension affine layer, data-initialized on the first forward pass."""

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
    """RealNVP-style flow. forward: latent -> data. inverse: data -> latent."""

    def __init__(self, dim, n_layers=6, hidden=64, seed=0, split='alternating',
                 interleave_actnorm=False, scale_cap=2.0):
        """split='random' draws a fresh half of the coordinates to condition on at every
        coupling instead of taking every other one, and interleave_actnorm puts a
        normalisation before each coupling rather than one at the end. Both are free and
        both matter once the dimension is large: on 784 MNIST pixels they take the held-out
        negative log-likelihood from 512 to 486, and with twice the depth to 460. Neither
        changes anything in two or ten dimensions, so the defaults are the old behaviour and
        existing checkpoints load unchanged.

        `scale_cap` bounds each coupling's log-scale, and it matters for a reason that held-
        out likelihood alone will not reveal. The sampling direction multiplies by exp(s) at
        every layer while the density direction divides by it, so a deep flow can fit a
        density well and still generate points far outside the data. On these 784 pixels the
        real range is about +-4; at the old cap of 2.0 a twelve-layer flow samples out to 67,
        at 0.25 to 16.6 -- and the tighter cap also fits BETTER, 448 against 468. Anything
        that uses the sampling path, which is everything the inner problem does, should
        check the range of its samples and not only its likelihood.
        """
        super().__init__()
        torch.manual_seed(seed)
        self.dim, self.split, self.scale_cap = dim, split, scale_cap
        self.interleave_actnorm = interleave_actnorm
        g = torch.Generator().manual_seed(seed)
        layers, norms = [], []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            if split == 'random':
                mask[torch.randperm(dim, generator=g)[:dim // 2]] = 1.0
            else:
                mask[i % 2::2] = 1.0
            if dim == 1:            # coupling needs >= 2 dims; fall back to actnorm only
                mask = torch.ones(dim)
            layers.append(AffineCoupling(dim, hidden, mask, scale_cap=scale_cap))
            norms.append(ActNorm(dim))
        self.couplings = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms) if interleave_actnorm else None
        self.actnorm = ActNorm(dim)

    def forward(self, z):
        """Latent -> data. Differentiable sampling path used by the inner problem."""
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
        """Data generation only -- detached on purpose. The differentiable sampling path
        the inner problem needs is `flow(T_eta(z))`, called explicitly."""
        z = torch.randn(n, self.dim, generator=generator,
                        dtype=self.actnorm.shift.dtype, device=self.actnorm.shift.device)
        return self.forward(z * temperature)[0]


def fit_flow(x, dim=None, n_layers=6, hidden=64, epochs=600, lr=3e-3, batch=2048,
             seed=0, verbose=False, val_frac=0.2, patience=40, eval_every=10):
    """Maximum-likelihood fit of the generator, early-stopped on a held-out split.

    Early stopping is not a refinement here, it is load-bearing. Trained to convergence on
    2000 samples these flows overfit catastrophically rather than gracefully: on P1 the
    median held-out log-density falls from 7.55 (the true model gives 7.59) at 100 epochs
    to -0.39 at 1000, and the fraction of held-out points the model puts below -50 goes
    from 0% to 36%. A generator with holes like that is the wrong object to anchor an
    ambiguity set to -- the adversary is free to move mass into a region the generator
    thinks is ordinary and the true model says is impossible -- and nothing downstream of
    it can recover. The training NLL falls monotonically throughout, so the only way to
    see this is to hold data out.

    Set val_frac=0 to disable and train for the full schedule.
    """
    dim = dim or x.shape[1]
    flow = Flow(dim, n_layers=n_layers, hidden=hidden, seed=seed).to(x.dtype)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)

    n_val = int(round(val_frac * x.shape[0])) if val_frac else 0
    if n_val:
        perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))
        x_val, x_tr = x[perm[:n_val]], x[perm[n_val:]]
    else:
        x_val, x_tr = None, x
    n = x_tr.shape[0]

    flow.train()
    with torch.no_grad():                      # trigger actnorm data init
        flow.inverse(x_tr[:min(n, 4096)])
    best = (float('inf'), 0, None)
    for ep in range(epochs):
        idx = torch.randperm(n, generator=g)[:batch]
        loss = -flow.log_prob(x_tr[idx]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        sched.step()
        if x_val is not None and (ep % eval_every == 0 or ep == epochs - 1):
            with torch.no_grad():
                v = float(-flow.log_prob(x_val).mean())
            if v < best[0] - 1e-4:
                best = (v, ep, {k: t.detach().clone()
                                for k, t in flow.state_dict().items()})
            elif ep - best[1] >= patience:
                break
        if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
            print(f'    flow epoch {ep:4d}  nll {loss.item():+.4f}')
    if best[2] is not None:
        flow.load_state_dict(best[2])
        if verbose:
            print(f'    early stop: best val nll {best[0]:+.4f} at epoch {best[1]}')
    flow.eval()
    return flow


def load_flow(ckpt, key='flow'):
    """Rebuild a saved flow from its own recorded shape.

    Callers used to hardcode `Flow(dim, n_layers=6, hidden=64)` next to every
    `load_state_dict`, which silently rots the moment the fitting code changes width or
    depth: the load raises, and a study script that redirects stderr to a log file loses
    the whole stage while the surrounding shell still reports success. Read the shape from
    the checkpoint instead.
    """
    sd = ckpt[key] if isinstance(ckpt, dict) and key in ckpt else ckpt
    cfg = ckpt.get(key + '_config') if isinstance(ckpt, dict) else None
    if cfg is None:                                   # infer it from the tensor shapes
        n_layers = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('couplings.'))
        w = sd['couplings.0.net.0.weight']
        cfg = dict(dim=w.shape[1], n_layers=n_layers, hidden=w.shape[0])
    flow = Flow(cfg['dim'], n_layers=cfg['n_layers'], hidden=cfg['hidden'], seed=0,
                split=cfg.get('split', 'alternating'),
                interleave_actnorm=cfg.get('interleave_actnorm', False),
                scale_cap=cfg.get('scale_cap', 2.0))
    flow.load_state_dict(sd)
    flow.eval()
    return flow
