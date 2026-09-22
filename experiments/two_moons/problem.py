"""Two moons as a distributionally robust classification problem.

    min_theta  sup_{Q in U(P)}  E_{(x,y) ~ Q} [ log(1 + exp(-y f_theta(x))) ]

the synthetic binary-classification setting used for adversarially robust training and
Wasserstein DRO (Sinha, Namkoong & Duchi 2018, Sec. 5). Two interleaving half-circles with
Gaussian noise: not linearly separable, so the classifier has to be nonlinear, and the
region between the arcs is where a shifted distribution does the damage.

Random variable: the pair (x, y) with x in R^2 and y in {-1, +1}. The label is drawn first
and the input conditionally, so a generative model of the inputs is class-conditional and
an adversary that moves x leaves y intact -- the usual adversarial-example semantics.

Decision variable: the weights of a small tanh network, flattened. `design_dim` is
4 * hidden + 1.

The density has no closed form, but each class-conditional is a Gaussian mixture along one
arc parameter, so integrating that single parameter numerically gives p(x | y) to
quadrature accuracy. That makes an exact KL available for validating the generator, which
a purely geometric construction would not.
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

    # -- the data ---------------------------------------------------------
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
        """log p(x | y), by midpoint quadrature over the arc parameter.

        Each class-conditional is a continuous Gaussian mixture along one arc,
        p(x|y) = (1/pi) int_0^pi N(x; c_y(t), sigma^2 I) dt, so a single-variable
        quadrature gives it to essentially machine accuracy -- and with it an exact KL for
        validating the generator, which a purely geometric construction would not provide.
        """
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

    # -- the classifier ---------------------------------------------------
    # Two parameterisations of the same decision boundary, because they put the design in
    # completely different places relative to a branch-trunk surrogate.
    #
    #   'nn'    a tanh network; the weights enter the score nonlinearly and are
    #           permutation-symmetric (swapping two hidden units changes nothing), so a
    #           surrogate fitted in the raw weight coordinates is fighting the geometry.
    #   'poly'  logistic regression on polynomial features; the score is theta' phi(x),
    #           LINEAR in the design. A branch-trunk surrogate <b(theta), t(x)> can
    #           represent that exactly with b = identity, which is the structure the
    #           architecture was built for -- and the structure a physical design problem
    #           usually has, where a handful of parameters enter a smooth response.
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
        """Per-sample logistic loss, with a ridge penalty on the weights.

        The penalty is not cosmetic. Without it the problem is separable, so the risk
        falls monotonically just by scaling theta up and the optimum is at infinity; boxed
        in, it sits in a corner of the design box, and the risk surface is a long flat
        plateau leading to it. That is the worst case for a trust-region method whose
        acceptance test is driven by sampling noise -- it stalls on the plateau while an
        exact gradient crawls to the corner, and the comparison measures that rather than
        the surrogate. With a ridge term the optimum is interior and the problem is
        strongly convex, which is also the structure the target application has: a bounded
        objective (area plus a penalised failure probability) with an interior optimum.
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
