"""Shared machinery of the robustness scripts.

Frame: u = 0 is the nominal design, u_i = +-1 puts parameter i on the face of the +-10% box.
The logit is <b(u), t(x)> + bias, so the design enters only through the branch features b(u).
The surrogate's expected hits are integrated over the muons as weighted points standing for n_ref
muons (muons.py: for the flow a quadrature grid, no sampling noise). One pass at u = 0
accumulating, with p = p(hit), s = p(1-p) and w the weights,

    T = sum w s t,   A = sum w s (1 - 2p) t t^T,   S = sum w s^2 t t^T

gives the derivatives of the expected hits H(u) = sum w p:

    grad = J^T T,   hess = J^T A J + d2(b.T)/du2,   active subspace = J^T S J / sum w,

with J = db/du from autograd on the branch net alone. The trunk features of a coarser set of
points are cached, after which the hits of any design are one matmul. The simulator needs sampled
muons (muon_blocks, simulate). The surrogate's own uncertainty is the spread of the ensemble
members it averages (lcso.surrogate.load_members).
"""
import time
from types import SimpleNamespace

import numpy as np
import torch

from lcso.surrogate import load_surrogate
from shield import MuonShield

OK, BAD, GREY = '#0072B2', '#D55E00', '0.6'      # figure colours


def setup(args):
    """(shield, surrogate) of the --muons and --model of cli.add_common. The muon source is built
    here, before any seeding, because building the flow draws from the global RNG."""
    torch.backends.cuda.matmul.allow_tf32 = False   # too coarse for the curvature
    shield = MuonShield(args.muons, fields=args.fields)
    model = load_surrogate(args.model, shield.phi_0, args.device)
    shield.muons.load()
    return shield, model


def one_group(x):
    """Every muon in group 0."""
    return torch.zeros(len(x), dtype=torch.long)


def muon_blocks(shield, n, block, seed=0, temperature=1.0):
    """Yield (lo, muons[lo:lo + block]) of sampled muons, for the simulator. Each block is
    seeded by its offset, so every script with the same seed sees the same muons."""
    t0 = time.time()
    for lo in range(0, n, block):
        torch.manual_seed(seed + lo)
        yield lo, shield.sample(min(block, n - lo), temperature)
        print(f'  {min(lo + block, n):,}/{n:,} muons ({time.time() - t0:.0f}s)')


def nominal_pass(model, points, group, G):
    """One pass at the nominal design over weighted points (muons, w). Per group g
    (group(muons) -> ids in [0, G)): T, A, S as in the module docstring, H = sum w p, N = sum w."""
    dev = model.bias.device
    with torch.no_grad():
        b0 = model.branch_net(torch.zeros(1, model.dim, device=dev))[0]
    P = b0.numel()
    T = torch.zeros(G, P, dtype=torch.float64, device=dev)
    A = torch.zeros(G, P, P, dtype=torch.float64, device=dev)
    S = torch.zeros(G, P, P, dtype=torch.float64, device=dev)
    H = torch.zeros(G, dtype=torch.float64, device=dev)
    N = torch.zeros(G, dtype=torch.float64, device=dev)
    with torch.no_grad():
        for x, w in points:
            x, w = x.to(dev), w.to(dev)
            g = group(x).to(dev)
            t = model.features(x)
            p = torch.sigmoid(t @ b0 + model.bias)
            s = p * (1 - p)
            H.index_add_(0, g, p.double() * w)
            N.index_add_(0, g, w)
            T.index_add_(0, g, (s.double() * w)[:, None] * t.double())
            for k in range(G):             # float32 matmul per chunk, summed in float64
                m = g == k
                tk, wk = t[m], w[m].float()
                A[k] += ((tk * (wk * s[m] * (1 - 2 * p[m]))[:, None]).T @ tk).double()
                S[k] += ((tk * (wk * s[m] ** 2)[:, None]).T @ tk).double()
    return SimpleNamespace(T=T, A=A, S=S, H=H.cpu().numpy(), N=N.cpu().numpy())


def feature_cache(model, points, group):
    """Trunk features (float16), weights and groups of every point, for cached_hits."""
    dev = model.bias.device
    t, w, g = [], [], []
    with torch.no_grad():
        for x, wx in points:
            x = x.to(dev)
            t.append(model.features(x).half())
            w.append(wx.to(dev))
            g.append(group(x).to(dev))
    return SimpleNamespace(t=torch.cat(t), w=torch.cat(w), g=torch.cat(g))


def branch_jacobian(model):
    """db/du at the nominal design: (p, D), float64."""
    u0 = torch.zeros(model.dim, device=model.bias.device)
    return torch.autograd.functional.jacobian(lambda u: model.branch_net(u[None])[0], u0).double()


def derivatives(model, J, T, A):
    """(grad, hess) of H in the normalized frame, from one set of accumulators."""
    u0 = torch.zeros(model.dim, device=model.bias.device)
    Hb = torch.autograd.functional.hessian(lambda u: model.branch_net(u[None])[0] @ T.float(), u0)
    h = (J.T @ A @ J + Hb.double()).cpu().numpy()
    return (J.T @ T).cpu().numpy(), 0.5 * (h + h.T)


@torch.no_grad()
def cached_hits(model, cache, G, U, budget=2 ** 24):
    """Expected hits of normalized designs U (m, D) over the cached points, per group: (m, G)."""
    t, w, g = cache.t, cache.w, cache.g
    U = torch.as_tensor(np.asarray(U), dtype=torch.float32, device=t.device)
    out = torch.zeros(U.shape[0], G, dtype=torch.float64, device=t.device)
    m_batch = max(1, budget // t.shape[0])
    for i in range(0, U.shape[0], m_batch):
        b = model.branch_net(U[i:i + m_batch])
        step = max(1, budget // b.shape[0])
        for c in range(0, t.shape[0], step):
            p = torch.sigmoid(b @ t[c:c + step].float().T + model.bias)
            out[i:i + b.shape[0]].index_add_(1, g[c:c + step], p.double() * w[c:c + step])
    return out.cpu().numpy()


def simulate(model, shield, blocks, U, group, G):
    """Simulated hits of normalized designs U (m, D) on sampled muons, per group: (m, G).
    A new transport seed every call."""
    phis = model.denormalize_phi(torch.as_tensor(np.asarray(U), dtype=torch.float32,
                                                 device=model.bias.device)).cpu()
    sim = np.zeros((len(U), G))
    shield.prefetch(phis)                        # the field maps, while the first ones are simulated
    for _, raw in blocks:
        groups = group(raw)
        for i in range(len(U)):
            sim[i] += np.bincount(groups[shield(phis[i], raw).cpu()].numpy(), minlength=G)
    return sim


def trust_region_max(g, B, radius):
    """argmax of g.v + 0.5 v^T B v over ||v|| <= radius (More-Sorensen: bisection on lam in
    (lam I - B) v = g)."""
    ev, Q = np.linalg.eigh(B)
    gh, top = Q.T @ g, ev[-1]
    if top < 0:                                  # concave: the free maximum may be inside
        v = -np.linalg.solve(B, g)
        if np.linalg.norm(v) <= radius:
            return v

    def v_of(lam):
        return Q @ (gh / (lam - ev))

    lo = top + 1e-12 * max(1.0, abs(top))
    if np.linalg.norm(v_of(lo)) < radius:        # hard case: step out along the top eigenvector
        keep = ev < lo
        v = Q[:, keep] @ (gh[keep] / (top - ev[keep]))
        return v + np.sqrt(max(radius ** 2 - v @ v, 0.0)) * Q[:, -1] * (1.0 if gh[-1] >= 0 else -1.0)
    hi = top + max(1e-12, np.linalg.norm(g) / radius)
    while np.linalg.norm(v_of(hi)) > radius:
        hi = top + 2 * (hi - top)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.linalg.norm(v_of(mid)) > radius else (lo, mid)
    return v_of(0.5 * (lo + hi))


def seed_se(values):
    """Standard error of the ensemble mean from its members' values (K, ...): sd / sqrt(K)."""
    v = np.asarray(values, dtype=np.float64)
    return v.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.zeros(v.shape[1:])


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class Report:
    """print() that also keeps the lines, to save as a text report."""

    def __init__(self):
        self.lines = []

    def __call__(self, line=''):
        print(line)
        self.lines.append(line)

    def save(self, path):
        with open(path, 'w') as f:
            f.write('\n'.join(self.lines) + '\n')
