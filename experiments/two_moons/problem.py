"""Two moons as a distributionally robust classification problem.

    min_theta  sup_{Q in U(P)}  E_{(x,y) ~ Q} [ log(1 + exp(-y f_theta(x))) ]

(Sinha, Namkoong & Duchi 2018, Sec. 5.) y in {-1, +1} is drawn first, then x | y on a noisy
half-circle, so an adversary that moves x leaves y intact. The classifier is either a small
tanh network ('nn') or logistic regression on polynomial features ('poly'). log p(x | y) is
exact up to a 1-d quadrature over the arc parameter.
"""
import math

import torch


class TwoMoons:
    def __init__(self, noise=0.12, hidden=16, kind='nn', degree=3, ridge=1e-3,
                 dtype=torch.float32):
        self.noise, self.hidden, self.dtype = noise, hidden, dtype
        self.kind, self.degree, self.ridge = kind, degree, ridge
        self.input_dim = 2
        self.design_dim = (degree + 1) * (degree + 2) // 2 if kind == 'poly' \
            else 4 * hidden + 1

    def _arc(self, t, y):
        """Centre of the noise kernel at arc parameter t, for class y in {-1, +1}."""
        upper = torch.stack([torch.cos(t), torch.sin(t)], -1)
        lower = torch.stack([1.0 - torch.cos(t), 0.5 - torch.sin(t)], -1)
        return torch.where((y > 0).unsqueeze(-1), upper, lower)

    def sample(self, n, generator=None):
        y = torch.where(torch.rand(n, generator=generator, dtype=self.dtype) < 0.5,
                        -1.0, 1.0).to(self.dtype)
        t = math.pi * torch.rand(n, generator=generator, dtype=self.dtype)
        x = self._arc(t, y) + self.noise * torch.randn(n, 2, generator=generator,
                                                       dtype=self.dtype)
        return x, y

    def log_prob(self, x, y, n_quad=512, chunk=20000):
        """log p(x | y) = log (1/pi) int_0^pi N(x; c_y(t), sigma^2 I) dt, by midpoint rule."""
        t = math.pi * (torch.arange(n_quad, dtype=self.dtype) + 0.5) / n_quad
        cent = {s: self._arc(t, torch.full((n_quad,), s, dtype=self.dtype))
                for s in (-1.0, 1.0)}
        norm = -math.log(2 * math.pi * self.noise ** 2) - math.log(n_quad)
        out = []
        for i in range(0, x.shape[0], chunk):
            xb, yb = x[i:i + chunk], y[i:i + chunk]
            per = {}
            for s, c in cent.items():
                d2 = ((xb.unsqueeze(1) - c.unsqueeze(0)) ** 2).sum(-1)   # (B, n_quad)
                per[s] = torch.logsumexp(-d2 / (2 * self.noise ** 2), dim=1) + norm
            out.append(torch.where(yb > 0, per[1.0], per[-1.0]))
        return torch.cat(out)

    # 'nn': score nonlinear and permutation-symmetric in the weights.
    # 'poly': score theta' phi(x) is linear in the design (easy for a branch-trunk surrogate).
    def _features(self, x):
        """Polynomial features up to `degree`, including the constant."""
        cols = [torch.ones_like(x[:, :1])]
        for d in range(1, self.degree + 1):
            for k in range(d + 1):
                cols.append(x[:, :1] ** (d - k) * x[:, 1:2] ** k)
        return torch.cat(cols, dim=1)

    def unpack(self, phi):
        h = self.hidden
        i = 0
        W1 = phi[i:i + 2 * h].reshape(2, h); i += 2 * h
        b1 = phi[i:i + h]; i += h
        W2 = phi[i:i + h].reshape(h, 1); i += h
        b2 = phi[i:i + 1]
        return W1, b1, W2, b2

    def logits(self, phi, x):
        if self.kind == 'poly':
            return self._features(x) @ phi
        W1, b1, W2, b2 = self.unpack(phi)
        return (torch.tanh(x @ W1 + b1) @ W2 + b2).squeeze(-1)

    def loss(self, phi, x, y):
        """Per-sample logistic loss plus a ridge penalty.

        The ridge keeps the optimum interior: without it the data are nearly separable and
        the risk keeps falling as theta is scaled up.
        """
        pen = self.ridge * (phi ** 2).sum()
        return torch.nn.functional.softplus(-y * self.logits(phi, x)) + pen

    def risk(self, phi, x, y):
        return self.loss(phi, x, y).mean()

    def accuracy(self, phi, x, y):
        return ((self.logits(phi, x) > 0).to(self.dtype) * 2 - 1 == y).to(self.dtype).mean()

    def init_design(self, seed=0):
        if self.kind == 'poly':
            return torch.zeros(self.design_dim, dtype=self.dtype)
        g = torch.Generator().manual_seed(seed)
        h = self.hidden
        return torch.cat([
            torch.randn(2 * h, generator=g, dtype=self.dtype) * (2.0 / 2) ** 0.5,
            torch.zeros(h, dtype=self.dtype),
            torch.randn(h, generator=g, dtype=self.dtype) * (1.0 / h) ** 0.5,
            torch.zeros(1, dtype=self.dtype)])

    def fit(self, x, y, steps=1500, lr=0.05, seed=0):
        """Nominal (non-robust) classifier, by gradient descent."""
        phi = self.init_design(seed).requires_grad_(True)
        opt = torch.optim.Adam([phi], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            self.risk(phi, x, y).backward()
            opt.step()
        return phi.detach()
