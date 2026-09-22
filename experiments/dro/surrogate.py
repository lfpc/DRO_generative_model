"""Branch-trunk surrogate of the loss, Section 4 of the paper.

    s(phi, x) = <b_w(phi), t_w(x)> + beta,     f_w(phi, x) = h(s(phi, x))

The link h is the only place the architecture sees whether the target is a probability
(sigmoid + cross-entropy) or a continuous observable (identity + squared loss).
"""
import torch
import torch.nn as nn


def _mlp(d_in, hidden, d_out, act=nn.GELU):
    return nn.Sequential(nn.Linear(d_in, hidden), act(),
                         nn.Linear(hidden, hidden), act(),
                         nn.Linear(hidden, d_out))


class BranchTrunk(nn.Module):
    def __init__(self, design_dim, input_dim, p=48, hidden=64, link='sigmoid',
                 design_scale=None, input_scale=None, seed=0, dtype=None):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        torch.manual_seed(seed)
        self.branch = _mlp(design_dim, hidden, p).to(dtype)
        self.trunk = _mlp(input_dim, hidden, p).to(dtype)
        self.beta = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.link = link
        self.p = p
        ds = torch.ones(design_dim, dtype=dtype) if design_scale is None else design_scale.to(dtype)
        xs = torch.ones(input_dim, dtype=dtype) if input_scale is None else input_scale.to(dtype)
        self.register_buffer('design_loc', torch.zeros(design_dim, dtype=dtype))
        self.register_buffer('design_scale', ds)
        self.register_buffer('input_loc', torch.zeros(input_dim, dtype=dtype))
        self.register_buffer('input_scale', xs)
        with torch.no_grad():
            self.branch[-1].weight.div_(p ** 0.5)
            self.trunk[-1].weight.div_(p ** 0.5)

    def set_norm(self, phi_samples, x_samples):
        with torch.no_grad():
            self.design_loc.copy_(phi_samples.mean(0))
            self.design_scale.copy_(phi_samples.std(0).clamp_min(1e-8))
            self.input_loc.copy_(x_samples.mean(0))
            self.input_scale.copy_(x_samples.std(0).clamp_min(1e-8))
        return self

    def score(self, phi, x):
        """phi: (D,) or (B, D).  x: (N, dx).  Returns (N,) or (B, N)."""
        pn = (phi - self.design_loc) / self.design_scale
        xn = (x - self.input_loc) / self.input_scale
        single = pn.dim() == 1
        b = self.branch(pn.unsqueeze(0) if single else pn)      # (B, p)
        t = self.trunk(xn)                                      # (N, p)
        out = b @ t.T + self.beta                               # (B, N)
        return out.squeeze(0) if single else out

    def forward(self, phi, x):
        s = self.score(phi, x)
        return torch.sigmoid(s) if self.link == 'sigmoid' else s


class TaylorBranchTrunk(nn.Module):
    """Surrogate whose design dependence is an explicit low-rank quadratic.

        s(phi, x) = a(x) + c(x)'u + 0.5 sum_j nu_j(x) (q_j'u)^2,   u = normalized design

    Only the shared basis Q (D x r) is learned on the design side, so the number of
    parameters describing the design dependence is O(D r) rather than an MLP's. That is
    the point: a generic branch network has to learn an arbitrary function of a
    D-dimensional design from however many designs were simulated, which fails once D
    grows, whereas a low-order expansion around a nominal design is identifiable from far
    fewer. This is the architecture the muon-shield surrogate uses.
    """

    def __init__(self, design_dim, input_dim, rank=4, hidden=96, link='sigmoid',
                 seed=0, dtype=None):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        torch.manual_seed(seed)
        # reduced QR of a D x r matrix returns only min(D, r) columns
        self.D, self.r, self.link = design_dim, min(rank, design_dim), link
        self.trunk = _mlp(input_dim, hidden, 1 + design_dim + self.r).to(dtype)
        self.Q_raw = nn.Parameter(torch.randn(design_dim, self.r, dtype=dtype) * 0.5)
        self.beta = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.register_buffer('design_loc', torch.zeros(design_dim, dtype=dtype))
        self.register_buffer('design_scale', torch.ones(design_dim, dtype=dtype))
        self.register_buffer('input_loc', torch.zeros(input_dim, dtype=dtype))
        self.register_buffer('input_scale', torch.ones(input_dim, dtype=dtype))

    @property
    def Q(self):
        return torch.linalg.qr(self.Q_raw, mode='reduced')[0]

    def set_norm(self, phi_samples, x_samples):
        with torch.no_grad():
            self.design_loc.copy_(phi_samples.mean(0))
            self.design_scale.copy_(phi_samples.std(0).clamp_min(1e-8))
            self.input_loc.copy_(x_samples.mean(0))
            self.input_scale.copy_(x_samples.std(0).clamp_min(1e-8))
        return self

    def score(self, phi, x):
        u = (phi - self.design_loc) / self.design_scale
        single = u.dim() == 1
        if single:
            u = u.unsqueeze(0)
        tau = self.trunk((x - self.input_loc) / self.input_scale)       # (N, 1+D+r)
        a = tau[:, 0]
        c = tau[:, 1:1 + self.D]
        nu = tau[:, 1 + self.D:]
        proj = u @ self.Q                                               # (B, r)
        out = a.unsqueeze(0) + u @ c.T + 0.5 * (proj ** 2) @ nu.T + self.beta
        return out.squeeze(0) if single else out

    def forward(self, phi, x):
        s = self.score(phi, x)
        return torch.sigmoid(s) if self.link == 'sigmoid' else s

    def design_derivatives(self, phi, x):
        """(value, grad, Hessian) of mean_x sigmoid(s(phi,x)) w.r.t. phi, in closed form.

        With s quadratic in the normalized design u and p = sigmoid(s), writing
        m(x) = c(x) + C(x)u for the per-input gradient of the score,

            d/du   mean p  =  mean[ p(1-p) m ]
            d2/du2 mean p  =  mean[ p(1-p)(1-2p) m m' ]  +  mean[ p(1-p) C ]

        and the second term collapses because C(x) = V diag(nu(x)) V' shares V across
        inputs: mean[p(1-p) C] = V diag(mean[p(1-p) nu]) V'. So the Hessian costs one
        (D x N)(N x D) product and no autograd at all -- no per-parameter backward pass,
        which is what makes this affordable when the design is high-dimensional.
        """
        sc = self.design_scale
        u = (phi - self.design_loc) / sc
        tau = self.trunk((x - self.input_loc) / self.input_scale)
        a, c, nu = tau[:, 0], tau[:, 1:1 + self.D], tau[:, 1 + self.D:]
        Q = self.Q                                    # (D, r)
        proj = u @ Q                                  # (r,)
        Cu = (nu * proj) @ Q.T                        # (N, D) = C(x) u
        s = a + c @ u + 0.5 * (nu * proj ** 2).sum(-1) + self.beta
        m = c + Cu                                    # (N, D)
        n = x.shape[0]
        if self.link != 'sigmoid':
            # Identity link: the prediction IS the quadratic, so the chain rule terms
            # above drop out and the Hessian is just the mean curvature -- no dependence
            # on the predictions at all, which is why it stays exact however large the
            # residuals are.
            grad_u = m.mean(0)
            hess_u = Q @ torch.diag(nu.mean(0)) @ Q.T
            return float(s.mean().detach()), grad_u / sc, hess_u / (sc[:, None] * sc[None, :])
        p = torch.sigmoid(s)
        w = p * (1 - p)                               # (N,)
        grad_u = (w.unsqueeze(1) * m).mean(0)
        H1 = (m * (w * (1 - 2 * p)).unsqueeze(1)).T @ m / n
        H2 = Q @ torch.diag((w.unsqueeze(1) * nu).mean(0)) @ Q.T
        hess_u = H1 + H2
        return float(p.mean()), grad_u / sc, hess_u / (sc[:, None] * sc[None, :])


def train_surrogate(model, phi_train, x_train, y_train, epochs=400, lr=3e-3,
                    batch_x=4096, seed=0, verbose=False, weight_decay=0.0):
    """phi_train: (M, D) designs.  x_train: (N, dx) shared input pool.  y_train: (M, N).

    The pool is shared across designs, which is what makes the branch-trunk factorization
    pay off in training as well as at evaluation time: the trunk is evaluated once per
    batch for all M designs instead of M times.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    M, N = y_train.shape
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for ep in range(epochs):
        idx = torch.randperm(N, generator=g)[:batch_x]
        s = model.score(phi_train, x_train[idx])              # (M, batch)
        y = y_train[:, idx]
        loss = bce(s, y) if model.link == 'sigmoid' else ((s - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
            print(f'    surrogate epoch {ep:4d}  loss {loss.item():.5f}')
    model.eval()
    return model


def train_resample(problem, flow, model, budget_designs, n_inputs, center=None,
                   epochs=2000, refresh_every=200, patience=20, n_val=4, buffer=None,
                   min_epochs=600,
                   temperature=1.35, lr=3e-3, batch_x=8192, seed=0, verbose=False,
                   designs='box'):
    """Rotating-buffer training, as lcso_resample.py does it.

    A static fit sees one set of designs for its whole life, so it is accurate on average
    over the design box but can be poor anywhere in particular -- and a minimax optimizer
    goes looking for exactly those places. Here a buffer of designs is held, half of it is
    re-simulated with fresh designs every `refresh_every` epochs, and training stops on
    held-out designs judged by the error in the aggregate quantity that matters (the
    failure probability), not by the per-sample cross-entropy.

    `budget_designs` is the TOTAL number of designs handed to the true loss, buffer plus
    every refresh, so it is directly comparable with the static scheme's budget.
    """
    from .evaluate import radial_designs, sobol_designs
    draw = ((lambda c, l, h, m, sd: radial_designs(c, l, h, m, sd))
            if designs == 'radial' else
            (lambda c, l, h, m, sd: sobol_designs(l, h, m, sd)))
    lo, hi = problem.design_bounds()
    center = center if center is not None else (lo + hi) / 2
    # Spend most of the budget on the live buffer: at a fixed budget, holding fewer
    # designs at once is strictly less information per step, so the buffer should be
    # as large as the budget allows after reserving a few held-out designs.
    n_val = max(2, min(n_val, budget_designs // 6))
    W = buffer or max(4, budget_designs - n_val)
    g = torch.Generator().manual_seed(seed)

    def simulate(designs):
        xs = flow.sample(n_inputs, g, temperature=temperature)
        ys = torch.stack([problem.failure(designs[i], xs)
                          for i in range(designs.shape[0])])
        return xs, ys

    spent = 0
    designs = draw(center, lo, hi, W, seed)
    designs[0] = center                        # slot 0 pinned to the nominal design
    xs, ys = simulate(designs)
    spent += W

    val_designs = draw(center, lo, hi, n_val, seed + 991)
    val_x, val_y = simulate(val_designs)
    spent += n_val
    val_pf = val_y.mean(dim=1)

    model.set_norm(designs, xs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    best, bad, best_state, rot = float('inf'), 0, None, 0
    hist = []
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(xs.shape[0], generator=g)[:batch_x]
        loss = bce(model.score(designs, xs[idx]), ys[:, idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if (ep + 1) % 100 == 0 and ep + 1 >= min_epochs:
            model.eval()
            with torch.no_grad():
                pred = model(val_designs, val_x).mean(dim=1)
            err = float((pred - val_pf).abs().mean())       # the aggregate quantity
            hist.append((ep, float(loss), err))
            if err < best - 1e-12:
                best, bad = err, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if refresh_every and (ep + 1) % refresh_every == 0 and spent < budget_designs:
            k = min(max(1, (W - 1) // 2), budget_designs - spent)
            fresh = draw(center, lo, hi, k, seed + 7919 + ep)
            slots = [1 + (rot + j) % max(1, W - 1) for j in range(k)]
            rot += k
            designs = designs.clone()
            designs[slots] = fresh
            xs, ys = simulate(designs)      # new inputs too, as the reference does
            spent += k
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f'    resample: {spent} designs simulated, best val Pf err {best:.5f}, '
              f'stopped at epoch {hist[-1][0] if hist else 0}')
    return model, spent, best
