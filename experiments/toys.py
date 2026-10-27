"""Toy problems with the muon shield's structure and an exact answer.

Each has a design phi = phi_0 + half * u (u = +-1: the tolerance box), samples x ~ p(x), and a
binary outcome per sample ("hit") with a KNOWN probability p(u, x) = sigmoid(f(u, x)). So the
expected hit rate H(u) = E_x p(u, x), its gradient and its Hessian are exact (autograd on a
large fixed sample), while a surrogate only ever sees (u, x, hit) rows, as with the simulator.

  synthetic     D = 43, x in R^6, rare hits (2e-3); the design acts mostly through a 4-d
                subspace (quadratic + tanh) plus a weak full-dimensional linear part
  short_column  D = 10: failure (1e-2) of a series system of five short columns (Kuschel &
                Rackwitz), box +-5%
  beam_shield   D = 16: a particle hits the detector (0.13) behind sixteen magnet sections; the
                fields act through one direction only, box +-50%
  two_moons     D = 21: misclassification (0.06) by a degree-5 polynomial logistic classifier at
                its optimum: small gradient, the Hessian dominates; box +-30%

    python toys.py        # print each problem's truth at u = 0
"""
import math
import os

import torch

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_designs.pt')


class Toy:
    """Subclasses set D, dx, phi_0, half and define sample(n, g) and logit_phi(phi (B, D), x (N, dx))."""
    dev = 'cpu'

    def logit(self, U, x):
        return self.logit_phi(self.phi_0 + self.half * U, x)

    def prob(self, U, x):
        return torch.sigmoid(self.logit(U, x))

    def to(self, dev):
        for k, v in vars(self).items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(dev))
        self.dev = dev
        return self


class Synthetic(Toy):
    def __init__(self, D=43, rank=4, eps=0.15, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.D, self.dx, self.r = D, 6, rank
        self.V = torch.linalg.qr(torch.randn(D, rank, generator=g))[0] * 0.5
        self.w = torch.randn(D, generator=g) / math.sqrt(D) * eps
        self.b = torch.randn(rank, generator=g) * 0.6
        self.Bx = torch.randn(rank, self.dx, generator=g) * 0.3
        self.k = torch.linspace(0.8, -0.5, rank)
        self.phi_0, self.half = torch.zeros(D), torch.ones(D)

    def sample(self, n, g=None):
        return torch.randn(n, self.dx, generator=g, device=self.dev)

    def logit_phi(self, U, x):
        z = U @ self.V                                                  # (B, r)
        lin = z @ (self.b[:, None] + self.Bx @ x.T)                     # (B, N)
        quad = 0.5 * (z ** 2) @ (self.k[:, None] + 0.3 * x[:, 2:2 + self.r].T)
        tail = 0.8 * torch.tanh(z[:, :1] - x[None, :, 1])
        full = (U @ self.w)[:, None] * (1 + 0.5 * x[None, :, 0])
        return -7.0 + 1.3 * x[None, :, 0] + 0.5 * x[None, :, 1] + lin + quad + tail + full


class ShortColumn(Toy):
    LOADINGS = torch.tensor([[0.35, 0.00, 0.00], [0.21, 0.28, 0.00], [0.00, 0.00, 0.25], [0.00, -0.12, 0.00]])

    def __init__(self, n_cols=5, sensor_sd=0.05, smooth=0.05):
        self.n, self.sd, self.smooth = n_cols, sensor_sd, smooth
        self.D, self.dx = 2 * n_cols, 12
        g = torch.Generator().manual_seed(0)
        self.share = 0.85 + 0.3 * torch.rand(3, n_cols, generator=g)
        self.mean = torch.tensor([250.0, 125.0, 2500.0, 40000.0]).log().repeat_interleave(3)
        self.A = self.LOADINGS.repeat_interleave(3, dim=0)
        # the benchmark's initial design moved off its kink (all columns equal), see DRO/experiments/optb
        u = torch.rand(self.D, generator=torch.Generator().manual_seed(0)) * 2 - 1
        init = torch.cat([torch.full((n_cols,), 0.30), torch.full((n_cols,), 0.55)])
        # 1.4x stronger columns: failure probability ~1e-2 instead of 0.7
        self.phi_0 = 1.4 * (init + 0.4 * u * 0.325)
        self.half = 0.05 * self.phi_0

    def sample(self, n, g=None):
        z = torch.randn(n, 3, generator=g, device=self.dev)
        return self.mean + z @ self.A.T + self.sd * torch.randn(n, self.dx, generator=g, device=self.dev)

    def logit_phi(self, phi, x):
        q = x.reshape(-1, 4, 3).mean(-1).exp()
        M1, M2, F, sy = (q[None, :, k:k + 1] for k in range(4))      # (1, N, 1)
        b, h = phi[:, None, :self.n], phi[:, None, self.n:]             # (B, 1, n)
        s1, s2, s3 = self.share
        g = (1 - 4 * M1 * s1 / (b * h ** 2 * sy) - 4 * M2 * s2 / (b ** 2 * h * sy)
             - (F * s3 / (b * h * sy)) ** 2)
        return -g.min(-1).values / self.smooth


class BeamShield(Toy):
    def __init__(self, n_sec=16, sec_len=0.5, drift=10.0, radius=1.5, smooth=0.1):
        self.D, self.dx = n_sec, 4
        self.L, self.drift, self.R, self.smooth = sec_len, drift, radius, smooth
        self.mu = torch.tensor([math.log(30.0), math.log(0.5), 0.0, 0.0])
        sd = torch.tensor([0.80, 0.55, 0.05, 0.004])
        corr = torch.eye(4)
        corr[0, 1] = corr[1, 0] = 0.65
        corr[2, 3] = corr[3, 2] = -0.30
        self.chol = torch.linalg.cholesky(corr * sd[None] * sd[:, None])
        k = torch.arange(1, n_sec + 1, dtype=torch.float32)
        self.coef = 0.3 * sec_len * (sec_len * (n_sec - k + 1) + drift)   # y_det is linear in the fields
        # twice the optimal fields (hit rate 0.13); the box is +-50%: the hits barely respond to +-10%
        self.phi_0 = 2 * _cached('beam_shield', self._solve)
        self.half = 0.5 * self.phi_0.abs()

    def sample(self, n, g=None):
        z = self.mu + torch.randn(n, 4, generator=g, device=self.dev) @ self.chol.T
        return torch.cat([z[:, :2].exp(), z[:, 2:]], 1)

    def y_det(self, phi, x):
        p = x[:, 0].clamp_min(0.5)
        th = x[:, 3] + x[:, 1] / p
        return x[None, :, 2] + th[None] * (self.D * self.L + self.drift) + (phi @ self.coef)[:, None] / p[None]

    def logit_phi(self, phi, x):
        return (self.R - self.y_det(phi, x).abs()) / self.smooth

    def _solve(self, lam=10.0, steps=3000):
        """The nominal optimum of field energy + lam * P(hit), as in DRO/experiments/beam_shield."""
        x = self.sample(200_000, torch.Generator().manual_seed(1))
        phi = torch.full((1, self.D), 0.3, requires_grad=True)
        opt = torch.optim.SGD([phi], lr=0.01, momentum=0.9)
        for _ in range(steps):
            opt.zero_grad()
            ((phi ** 2).sum() * self.L / (self.D * self.L) + lam * torch.sigmoid(self.logit_phi(phi, x)).mean()).backward()
            opt.step()
            with torch.no_grad():
                phi.clamp_(-2, 2)
        return phi.detach()[0]


class TwoMoons(Toy):
    def __init__(self, noise=0.12, degree=5, ridge=1e-3):
        self.noise, self.deg, self.ridge = noise, degree, ridge
        self.D, self.dx = (degree + 1) * (degree + 2) // 2, 3
        self.phi_0 = _cached('two_moons', self._fit)
        self.half = 0.3 * self.phi_0.abs().clamp_min(0.1 * self.phi_0.abs().mean())     # +-30%: a flat optimum

    def sample(self, n, g=None):
        y = torch.where(torch.rand(n, generator=g, device=self.dev) < 0.5, -1.0, 1.0)
        t = math.pi * torch.rand(n, generator=g, device=self.dev)
        up = torch.stack([torch.cos(t), torch.sin(t)], -1)
        lo = torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], -1)
        x = torch.where(y[:, None] > 0, up, lo) + self.noise * torch.randn(n, 2, generator=g, device=self.dev)
        return torch.cat([x, y[:, None]], 1)

    def feats(self, x):
        return torch.stack([x[:, 0] ** (d - k) * x[:, 1] ** k for d in range(self.deg + 1) for k in range(d + 1)], 1)

    def logit_phi(self, phi, x):
        """Misclassification probability sigmoid(-y f(x)), the logistic loss's 'error'."""
        return -x[None, :, 2] * (phi @ self.feats(x).T)

    def _fit(self, steps=2000):
        x = self.sample(200_000, torch.Generator().manual_seed(1))
        phi = torch.zeros(1, self.D, requires_grad=True)
        opt = torch.optim.Adam([phi], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            (torch.nn.functional.softplus(self.logit_phi(phi, x)).mean() + self.ridge * (phi ** 2).sum()).backward()
            opt.step()
        return phi.detach()[0]


def _cached(name, solve):
    """Nominal designs that need an optimisation, solved once and kept in toy_designs.pt."""
    store = torch.load(CACHE) if os.path.exists(CACHE) else {}
    if name not in store:
        store[name] = solve()
        torch.save(store, CACHE)
    return store[name]


PROBLEMS = {'synthetic': Synthetic, 'short_column': ShortColumn, 'beam_shield': BeamShield, 'two_moons': TwoMoons}


def load(name, device='cpu'):
    return PROBLEMS[name]().to(device)


# ---------------------------------------------------------------------------
# The truth
# ---------------------------------------------------------------------------
def true_H(prob, X, U, chunk=2 ** 16):
    """H at designs U (B, D) over X: (B,), float64."""
    with torch.no_grad():
        return sum(prob.prob(U, X[c:c + chunk]).double().sum(1) for c in range(0, len(X), chunk)) / len(X)


def true_derivatives(prob, X, u0=None, chunk=2 ** 15):
    """(H, grad, Hessian) at u0 (default 0) over X, by autograd in float64."""
    from torch.func import hessian, jacrev
    u0 = torch.zeros(prob.D, device=X.device, dtype=torch.float64) if u0 is None else u0.double()
    saved = {k: v for k, v in vars(prob).items() if torch.is_tensor(v)}
    for k, v in saved.items():                     # the whole problem in float64
        setattr(prob, k, v.double())
    try:
        H, g, h = 0.0, 0.0, 0.0
        for c in range(0, len(X), chunk):
            xb = X[c:c + chunk].double()

            def f(u):
                return prob.prob(u[None], xb)[0].sum()

            H += float(f(u0))
            g = g + jacrev(f)(u0)
            h = h + hessian(f)(u0)
    finally:
        for k, v in saved.items():
            setattr(prob, k, v)
    n = len(X)
    return H / n, (g / n).cpu(), (h / n).cpu()


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for name in PROBLEMS:
        p = load(name, dev)
        X = p.sample(1_000_000, torch.Generator(dev).manual_seed(7))
        H, g, h = true_derivatives(p, X)
        ev = torch.linalg.eigvalsh(h)
        print(f'{name:13s} D={p.D:2d} H={H:.3e}  |g|/H={float(g.norm()) / H:.3f}  '
              f'eig(H)/H in [{float(ev[0]) / H:+.3f}, {float(ev[-1]) / H:+.3f}]  |hess|_F/H={float(h.norm()) / H:.3f}')
