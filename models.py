from math import sqrt, log, exp
import torch
from scipy.stats import qmc

def _mlp(in_dim, hidden, out_dim, activation):
    act = {'relu': torch.nn.ReLU, 'gelu': torch.nn.GELU, 'silu': torch.nn.SiLU}[activation]
    return torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), act(),
                               torch.nn.Linear(hidden, hidden), act(),
                               torch.nn.Linear(hidden, out_dim))


class LCSONet(torch.nn.Module):
    """logit(phi, x) = <b(phi), t(x)> + bias, on PHYSICAL phi and RAW muons.

    The phi box and the muon statistics are buffers, so they follow .to() and are saved
    with the state_dict.
    """

    def __init__(self, hidden: int = 128, p: int = 128, phi_dim: int = 43, x_dim: int = 7,
                 sampling: str = 'sobol', delta: float = 0.1, phi_0: torch.Tensor = None):
        super().__init__()
        if phi_0 is None:
            raise ValueError('phi_0 is required: it fixes the +-delta box for phi')
        self.hp = {'hidden': hidden, 'p': p}
        self.dim, self.x_dim, self.p = phi_dim, x_dim, p
        self.sampling, self.delta = sampling, delta

        self.branch_net = _mlp(phi_dim, hidden, p, 'silu')
        self.trunk_net = _mlp(x_dim, hidden, p, 'relu')
        self.bias = torch.nn.Parameter(torch.zeros(1))
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.branch_net[-1].weight.div_(p ** 0.5)
            self.trunk_net[-1].weight.div_(p ** 0.5)

        phi_0 = torch.as_tensor(phi_0, dtype=torch.float32).view(-1)
        self.register_buffer('lower_bound', phi_0 - delta * phi_0)
        self.register_buffer('upper_bound', phi_0 + delta * phi_0)
        self.register_buffer('muon_mean', torch.zeros(x_dim - 1))    # kinematics, no pdg
        self.register_buffer('muon_std', torch.ones(x_dim - 1))

    # ---- normalization -----------------------------------------------------
    def set_muon_norm(self, mean, std):
        self.muon_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).view(-1))
        self.muon_std.copy_(torch.as_tensor(std, dtype=torch.float32).view(-1))
        return self

    def normalize_muons(self, x):
        """Raw muons (..., >=x_dim) -> standardized kinematics, pdg -> -sign(pdg)."""
        k = self.muon_mean.numel()
        return torch.cat([(x[..., :k] - self.muon_mean) / self.muon_std,
                          -torch.sign(x[..., k:k + 1])], dim=-1)

    def normalize_phi(self, phi):
        """Physical phi -> the [-1, 1] box (the nominal design maps to 0)."""
        return (phi - self.lower_bound) / (self.upper_bound - self.lower_bound) * 2 - 1

    def denormalize_phi(self, u):
        return (u + 1) / 2 * (self.upper_bound - self.lower_bound) + self.lower_bound

    def sample_phi(self, n_samples=100, seed=None):
        """Designs in the NORMALIZED box; pass through denormalize_phi to use them."""
        s = self.sampling.lower()
        gen = None if seed is None else torch.Generator().manual_seed(int(seed))
        if s == 'normal':
            u = torch.randn(n_samples, self.dim, generator=gen)
        elif s == 'uniform':
            u = torch.rand(n_samples, self.dim, generator=gen) * 2 - 1
        elif s == 'lhs':
            u = qmc.LatinHypercube(d=self.dim, seed=seed).random(n_samples) * 2 - 1
        elif s == 'sobol':
            u = qmc.Sobol(d=self.dim, scramble=True, seed=seed).random(n_samples) * 2 - 1
        else:
            raise ValueError(f'unknown sampling method: {self.sampling}')
        return torch.as_tensor(u, dtype=torch.float32)

    # ---- evaluation --------------------------------------------------------
    def forward(self, phi, x):
        """phi: PHYSICAL (B, d) or (d,). x: RAW muons (B, N, x_dim) or (N, x_dim)."""
        u = self.normalize_phi(phi)
        b = self.branch_net(u if u.dim() > 1 else u.unsqueeze(0))
        t = self.trunk_net(self.normalize_muons(x))
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and b.shape[0] > 1:
            t = t.expand(b.shape[0], -1, -1)
        return torch.einsum('bp,bnp->bn', b, t) + self.bias

    def predict_proba(self, phi, x):
        return torch.sigmoid(self.forward(phi, x))

    def predict_hits(self, phi, x, batch_size: int = 2 ** 20):
        """(sum of hit probabilities, its Bernoulli spread sqrt(sum p(1-p)))."""
        single_phi = phi.dim() == 1
        if single_phi:
            phi = phi.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        total = torch.zeros(phi.shape[0], device=phi.device, dtype=phi.dtype)
        var = torch.zeros_like(total)
        for s in range(0, x.shape[1], batch_size):
            pr = self.predict_proba(phi, x[:, s:s + batch_size])
            total = total + pr.sum(dim=1)
            var = var + (pr * (1 - pr)).sum(dim=1)
        std = var.sqrt()
        return (total.squeeze(0), std.squeeze(0)) if single_phi else (total, std)

    # ---- derivatives of the weighted expected hits, w.r.t. PHYSICAL phi -----
    def _weighted_hits(self, phi, x, weights):
        pr = self.predict_proba(phi, x)
        return pr.sum() if weights is None else (pr * weights.reshape(1, -1).to(pr)).sum()

    def grad_phi(self, phi, x, weights=None, chunk: int = 2 ** 18):
        """d/dphi of sum_x w_x p(hit | phi, x). Chunked: the graph over a few million
        muons does not fit."""
        phi = phi.detach().clone().requires_grad_(True)
        pb = phi if phi.dim() > 1 else phi.unsqueeze(0)
        out = torch.zeros_like(phi)
        for s in range(0, x.shape[-2], chunk):
            w = None if weights is None else weights[s:s + chunk]
            out = out + torch.autograd.grad(
                self._weighted_hits(pb, x[..., s:s + chunk, :], w), phi)[0]
        return out.detach()

    def hess_phi(self, phi, x, weights=None, chunk: int = 2 ** 18):
        """d2/dphi2 of the same. One backward per parameter per chunk, so it does not
        scale: see TaylorLCSONet for the closed form."""
        phi = phi.detach().clone().requires_grad_(True)
        pb = phi if phi.dim() > 1 else phi.unsqueeze(0)
        d = phi.numel()
        out = torch.zeros(d, d, device=phi.device, dtype=phi.dtype)
        for s in range(0, x.shape[-2], chunk):
            w = None if weights is None else weights[s:s + chunk]
            g = torch.autograd.grad(self._weighted_hits(pb, x[..., s:s + chunk, :], w),
                                    phi, create_graph=True)[0]
            for i in range(d):
                out[i] += torch.autograd.grad(g.reshape(-1)[i], phi,
                                              retain_graph=i < d - 1)[0].reshape(-1)
        return out.detach()


# --------------------------------------------------------------------------- #
# Quadratic (Taylor) surrogate
# --------------------------------------------------------------------------- #
class _TaylorTrunk(torch.nn.Module):
    """Muon -> its own Taylor coefficients [a(x), g(x), lambda(x), (residual)]."""

    def __init__(self, x_dim, phi_dim, hidden, rank, p_res):
        super().__init__()
        self.base = torch.nn.Sequential(
            torch.nn.Linear(x_dim, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU())
        self.head_a = torch.nn.Linear(hidden, 1)
        self.head_g = torch.nn.Linear(hidden, phi_dim)
        self.head_lambda = torch.nn.Linear(hidden, rank)
        self.head_res = torch.nn.Linear(hidden, p_res) if p_res else None
        with torch.no_grad():       # start near-linear: a large initial curvature blows
            self.head_lambda.weight.mul_(0.1)    # up at the box faces, where |u| ~ sqrt(D)
            self.head_lambda.bias.zero_()

    def forward(self, x):
        h = self.base(x)
        feats = [self.head_a(h), self.head_g(h), self.head_lambda(h)]
        if self.head_res is not None:
            feats.append(self.head_res(h))
        return torch.cat(feats, dim=-1)


class _TaylorBranch(torch.nn.Module):
    """phi -> [1, u, 0.5*(Q^T u)^2, (||u||^3 * residual)], u = normalized phi. Only Q and
    the residual are learned; the polynomial structure is imposed."""

    def __init__(self, phi_dim, hidden, rank, p_res):
        super().__init__()
        self.Q_raw = torch.nn.Parameter(torch.randn(phi_dim, rank))
        self.res = _mlp(phi_dim, hidden, p_res, 'gelu') if p_res else None

    @property
    def Q(self):
        return torch.linalg.qr(self.Q_raw, mode='reduced')[0]

    def forward(self, u):
        feats = [torch.ones_like(u[..., :1]), u, 0.5 * (u @ self.Q) ** 2]
        if self.res is not None:
            # The gate makes the residual and its first two derivatives vanish at u = 0,
            # so a, g and C stay the true value, gradient and Hessian of the logit there.
            # Written as (|u|^2+eps)^1.5 because d|u|/du is 0/0 at the nominal design.
            gate = (u.pow(2).sum(-1, keepdim=True) + 1e-12).pow(1.5)
            feats.append(self.res(u) * gate)
        return torch.cat(feats, dim=-1)


class TaylorLCSONet(LCSONet):
    r"""LCSONet whose logit is an explicit per-muon quadratic in phi:

        z(phi, x) = a(x) + g(x).u + 0.5 u^T C(x) u  [+ ||u||^3 <r_b(x), r_t(u)>]
        C(x) = Q diag(lambda(x)) Q^T,   u = normalized phi

    Only the two networks differ from the parent, so forward, the normalization and
    predict_hits are inherited. With H = sum_x w_x sigma(z), m_x = g_x + C_x u, p =
    sigma(z), the derivatives of H are closed form:

        grad H = sum_x w_x p(1-p) m_x
        hess H = sum_x w_x p(1-p)(1-2p) m_x m_x^T + sum_x w_x p(1-p) C_x
    """

    def __init__(self, hidden: int = 128, p: int = 64, phi_dim: int = 43, x_dim: int = 7,
                 sampling: str = 'sobol', delta: float = 0.1, phi_0: torch.Tensor = None,
                 rank: int = 8, use_residual: bool = False):
        super().__init__(hidden=hidden, p=p, phi_dim=phi_dim, x_dim=x_dim,
                         sampling=sampling, delta=delta, phi_0=phi_0)
        p_res = p if use_residual else 0
        self.branch_net = _TaylorBranch(phi_dim, hidden, rank, p_res)
        self.trunk_net = _TaylorTrunk(x_dim, phi_dim, hidden, rank, p_res)
        self.rank = rank
        self.p = 1 + phi_dim + rank + p_res          # feature dim
        self.hp = {'hidden': hidden, 'p': p, 'rank': rank, 'use_residual': use_residual}

    @torch.no_grad()
    def taylor_coefficients(self, x):
        """a(x), g(x), lambda(x): value, gradient and curvature eigenvalues of the LOGIT
        at the nominal design."""
        tau = self.trunk_net(self.normalize_muons(x))
        D, r = self.dim, self.rank
        return tau[..., :1], tau[..., 1:1 + D], tau[..., 1 + D:1 + D + r]

    @torch.no_grad()
    def hits_derivatives(self, x, weights=None, phi=None, want_hess=True,
                         chunk: int = 2 ** 18):
        """(H, grad, hess) of H(phi) = sum_x w_x sigma(z(phi, x)), w.r.t. PHYSICAL phi.

        x is RAW, phi is PHYSICAL (None = the nominal design). Exact for the quadratic
        core at any phi; the residual is not differentiated, and it vanishes to third
        order at the nominal design.
        """
        ref = next(self.parameters())
        dev, dt, D = ref.device, ref.dtype, self.dim
        u = (torch.zeros(D, device=dev, dtype=dt) if phi is None
             else self.normalize_phi(torch.as_tensor(phi, dtype=dt, device=dev).view(-1)))
        Q = self.branch_net.Q
        H = torch.zeros((), dtype=torch.float64, device=dev)
        grad = torch.zeros(D, dtype=torch.float64, device=dev)
        hess = torch.zeros(D, D, dtype=torch.float64, device=dev) if want_hess else None
        lam_sum = torch.zeros(self.rank, dtype=torch.float64, device=dev)

        for s in range(0, x.shape[0], chunk):
            xb = x[s:s + chunk].to(dev, dt)
            a, g, lam = self.taylor_coefficients(xb)
            w = (torch.ones(xb.shape[0], device=dev, dtype=dt) if weights is None
                 else weights[s:s + chunk].to(dev, dt))
            Cu = (lam * (u @ Q)) @ Q.T                  # C(x) u, without forming C
            z = a.squeeze(-1) + g @ u + 0.5 * (Cu * u).sum(-1) + self.bias
            pr = torch.sigmoid(z)
            dp = pr * (1 - pr) * w
            m = g + Cu                                  # per-muon logit gradient
            H += (pr * w).sum().double()
            grad += (dp.unsqueeze(-1) * m).sum(0).double()
            if want_hess:
                hess += (m * (dp * (1 - 2 * pr)).unsqueeze(-1)).T.double() @ m.double()
                lam_sum += (dp.unsqueeze(-1) * lam).sum(0).double()
        if want_hess:
            hess += ((Q * lam_sum.to(dt)) @ Q.T).double()

        jac = (2.0 / (self.upper_bound - self.lower_bound)).double()   # normalized -> physical
        grad = grad * jac
        if want_hess:
            hess = hess * jac[:, None] * jac[None, :]
        return float(H), grad.cpu(), (hess.cpu() if want_hess else None)

    def grad_phi(self, phi, x, weights=None, chunk: int = 2 ** 18):
        return self.hits_derivatives(x, weights, phi, want_hess=False, chunk=chunk)[1]

    def hess_phi(self, phi, x, weights=None, chunk: int = 2 ** 18):
        return self.hits_derivatives(x, weights, phi, want_hess=True, chunk=chunk)[2]


# --------------------------------------------------------------------------- #
# build / save / load
# --------------------------------------------------------------------------- #
SURROGATES = {'deeponet': LCSONet, 'taylor': TaylorLCSONet}


def build_surrogate(model_type='deeponet', phi_dim=43, x_dim=7, phi_0=None, hidden=128,
                    p=128, sampling='sobol', delta=0.1, **taylor_kw):
    """`taylor_kw` (rank, use_residual) is ignored for 'deeponet'."""
    if model_type not in SURROGATES:
        raise ValueError(f'unknown model_type {model_type!r}; pick one of {list(SURROGATES)}')
    common = dict(hidden=hidden, p=p, phi_dim=phi_dim, x_dim=x_dim,
                  sampling=sampling, delta=delta, phi_0=phi_0)
    return (LCSONet(**common) if model_type == 'deeponet'
            else TaylorLCSONet(**common, **taylor_kw))


def save_surrogate(model, path):
    cfg = dict(model_type='taylor' if isinstance(model, TaylorLCSONet) else 'deeponet',
               phi_dim=model.dim, x_dim=model.x_dim, sampling=model.sampling,
               delta=model.delta, **model.hp)
    torch.save({'config': cfg, 'state_dict': model.state_dict()}, path)


def load_surrogate(path, phi_0, map_location='cpu', delta=None):
    """Rebuild a surrogate saved by save_surrogate. Returns (model, config)."""
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if not (isinstance(obj, dict) and 'state_dict' in obj and 'config' in obj):
        raise RuntimeError(f'{path} is not a self-describing checkpoint; retrain with the '
                           f'current lcso.py.')
    cfg = dict(obj['config'])
    if delta is not None:
        cfg['delta'] = delta
    model = build_surrogate(phi_0=phi_0, **cfg)
    sd = {**model.state_dict(), **obj['state_dict']}
    if 'muon_mean' in obj:              # written when the statistics were plain attributes
        sd['muon_mean'] = torch.as_tensor(obj['muon_mean'], dtype=torch.float32).view(-1)
        sd['muon_std'] = torch.as_tensor(obj['muon_std'], dtype=torch.float32).view(-1)
    elif 'muon_mean' not in obj['state_dict']:
        raise RuntimeError(f'{path} carries no muon normalization, so it cannot be '
                           f'evaluated. Retrain with the current lcso.py.')
    model.load_state_dict(sd)
    return model, cfg


# --------------------------------------------------------------------------- #
# Gaussian process (gp.py)
# --------------------------------------------------------------------------- #
def GaussianProcess(train_X, train_Y, bounds=None):
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.means import ConstantMean
    from gpytorch.priors import LogNormalPrior

    d = train_X.shape[-1]
    loc = sqrt(2.0) + 0.5 * log(d)
    base = MaternKernel(nu=2.5, ard_num_dims=d,
                        lengthscale_prior=LogNormalPrior(loc, sqrt(3.0)))
    base.lengthscale = exp(loc)
    covar = ScaleKernel(base, outputscale_prior=LogNormalPrior(0.0, 1.0))
    covar.outputscale = 1.0
    return SingleTaskGP(train_X=train_X, train_Y=train_Y, mean_module=ConstantMean(),
                        covar_module=covar, input_transform=Normalize(d=d, bounds=bounds),
                        outcome_transform=Standardize(m=1)).to(torch.float64)
