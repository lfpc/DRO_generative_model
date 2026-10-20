"""Can the surrogate's expected hits be integrated instead of sampled?

The muons enter the shield on axis (shield.sample overwrites the positions), so a muon is its
momentum (pz, pt), azimuth phi and charge q. In the flow these are independent: (pz, pt) is the
image of a 2-d standard normal latent z under the momentum flow, phi is uniform and q is +-1 with
the data's mu+ fraction. So, with the kinematic envelope of shield.sample as an indicator A,

    H(u) / N = E_z,phi,q[ A p(hit | u, x) ] / E_z[ A ],     x = x(z, phi, q),

and three estimators of it (and of its gradient and Hessian at u = 0) are compared at equal
numbers of points:
  mc     plain Monte Carlo: z ~ N(0, I), phi ~ U, q ~ Bernoulli (what the analysis does now)
  rqmc   scrambled Sobol points in 4-d, z by the inverse normal CDF, q by thresholding
  grid   a tensor grid in z on [-L, L]^2 with Gaussian weights, phi midpoints, both charges

    python quadrature.py            # -> outputs/surrogate_study/figs/quadrature.{png,txt}
"""
import argparse
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm, qmc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # after this folder: both have a plots.py
from analysis import branch_jacobian, derivatives  # noqa: E402
from shield import MuonShield  # noqa: E402
from surrogate import load_surrogate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'outputs/surrogate_study/figs')


class Muons:
    """The flow's momentum map and the shield's envelope, as functions of (z, phi, q)."""

    def __init__(self, shield, dev):
        shield._load_flow()
        gen = shield._flow
        self.flow, self.m, self.mu_plus = gen.flow.momentum, gen.m, gen.mu_plus
        self.mean = torch.as_tensor(gen.mean[:2], dtype=torch.float64, device=dev)
        self.std = torch.as_tensor(gen.std[:2], dtype=torch.float64, device=dev)
        self.env, self.dev = shield, dev

    @torch.no_grad()
    def momentum(self, z):
        """Latent (n, 2) -> pz, pt, and the envelope indicator."""
        y = z.float()
        for layer in reversed(self.flow.layers):
            y = layer.inverse(y, None)
        uv = y.double() * self.std + self.mean
        m = self.m
        p = m['p_min'] + (m['p_max'] - m['p_min']) / (1 + torch.exp(-uv[:, 0]))
        pz = p / torch.sqrt(1 + torch.exp(2 * uv[:, 1]))
        pt = pz * torch.exp(uv[:, 1])
        ok = (p <= self.env.P_MAX) & (pt <= self.env.PT_MAX) & (pz > self.env.PZ_MIN)
        return pz, pt, ok

    def muons(self, pz, pt, phi, q):
        """Raw muon rows (n, 7) as shield.sample makes them."""
        z0 = torch.zeros_like(pz)
        return torch.stack([pt * torch.cos(phi), pt * torch.sin(phi), pz, z0, z0, z0 - 1, -13 * q], 1).float()


@torch.no_grad()
def accumulate(model, x, w, acc):
    """Add the weighted H, T = sum w s t and A = sum w s (1 - 2s) t t' (s = p(1 - p)) of muons x."""
    t = model.features(x)
    b0 = model.branch_net(torch.zeros(1, model.dim, device=x.device))[0]
    p = torch.sigmoid(t @ b0 + model.bias)
    s = p * (1 - p) * w.float()
    acc['H'] += float((p.double() * w).sum())
    acc['T'] += (s[:, None] * t).double().sum(0)
    acc['A'] += ((t * (s * (1 - 2 * p))[:, None]).T @ t).double()
    acc['W'] += float(w.sum())


def estimate(model, J, points, n_muons=1e9):
    """(H, grad, hess) per n_muons accepted muons from batches of (x, weight, accepted-weight)."""
    P = model.branch_net[-1].out_features
    dev = model.bias.device
    acc = dict(H=0.0, W=0.0, T=torch.zeros(P, dtype=torch.float64, device=dev),
               A=torch.zeros(P, P, dtype=torch.float64, device=dev))
    for x, w in points:
        accumulate(model, x, w, acc)
    g, h = derivatives(model, J, acc['T'] / acc['W'], acc['A'] / acc['W'])
    return acc['H'] / acc['W'] * n_muons, g * n_muons, h * n_muons


def mc_points(M, n, seed, batch=2 ** 21):
    g = torch.Generator(M.dev).manual_seed(seed)
    for c in range(0, n, batch):
        k = min(batch, n - c)
        z = torch.randn(k, 2, device=M.dev, generator=g, dtype=torch.float64)
        phi = 2 * math.pi * torch.rand(k, device=M.dev, generator=g, dtype=torch.float64)
        q = torch.where(torch.rand(k, device=M.dev, generator=g) < M.mu_plus, 1.0, -1.0).double()
        pz, pt, ok = M.momentum(z)
        yield M.muons(pz, pt, phi, q), ok.double()


def rqmc_points(M, n, seed, batch=2 ** 21):
    s = qmc.Sobol(d=4, scramble=True, seed=seed).random(n)
    for c in range(0, n, batch):
        u = torch.as_tensor(s[c:c + batch], device=M.dev)
        z = torch.as_tensor(norm.ppf(np.clip(s[c:c + batch, :2], 1e-15, 1 - 1e-15)), device=M.dev)
        q = torch.where(u[:, 3] < M.mu_plus, 1.0, -1.0).double()
        pz, pt, ok = M.momentum(z)
        yield M.muons(pz, pt, 2 * math.pi * u[:, 2], q), ok.double()


def grid_points(M, n_z, n_phi, L=7.0, batch=2 ** 21):
    """Tensor grid: z on [-L, L]^2 (trapezoid, Gaussian weights), phi midpoints, both charges."""
    zs = torch.linspace(-L, L, n_z, dtype=torch.float64, device=M.dev)
    wz = torch.exp(-zs ** 2 / 2) * (zs[1] - zs[0]) / math.sqrt(2 * math.pi)
    Z = torch.stack(torch.meshgrid(zs, zs, indexing='ij'), -1).view(-1, 2)
    WZ = (wz[:, None] * wz[None, :]).reshape(-1)
    pz, pt, ok = M.momentum(Z)
    keep = WZ > 1e-18 * WZ.max()
    pz, pt, ok, WZ = pz[keep], pt[keep], ok[keep], WZ[keep]
    phis = (torch.arange(n_phi, dtype=torch.float64, device=M.dev) + 0.5) * 2 * math.pi / n_phi
    per = max(1, batch // (2 * n_phi))
    for c in range(0, len(pz), per):
        a, b, o, w = pz[c:c + per], pt[c:c + per], ok[c:c + per], WZ[c:c + per]
        k = len(a)
        for q, pq in ((1.0, M.mu_plus), (-1.0, 1 - M.mu_plus)):
            A, B, O, W = (v.repeat_interleave(n_phi) for v in (a, b, o, w))
            PH = phis.repeat(k)
            yield M.muons(A, B, PH, torch.full_like(A, q)), O * W * pq / n_phi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=type(
        'F', (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter), {}))
    parser.add_argument('--model', default=os.path.join(ROOT, 'outputs/lcso.pt'))
    parser.add_argument('--sizes', default='1e5,1e6,1e7,1e8', help='points per estimate (mc, rqmc)')
    parser.add_argument('--reps', type=int, default=8, help='independent repeats per size (mc, rqmc)')
    parser.add_argument('--grids', default='101x8,201x16,401x32,801x32,801x64,1601x64', help='n_z x n_phi')
    args = parser.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    shield = MuonShield()
    model = load_surrogate(args.model, shield.phi_0, dev)
    J = branch_jacobian(model)
    M = Muons(shield, dev)
    out = dict(model=args.model, mu_plus=float(M.mu_plus), mc={}, rqmc={}, grid={})

    def run(points):
        t0 = time.time()
        H, g, h = estimate(model, J, points)
        return dict(H=H, g=g.tolist(), h=h.tolist(), seconds=time.time() - t0)

    for gspec in args.grids.split(','):
        nz, nphi = map(int, gspec.split('x'))
        out['grid'][gspec] = r = run(grid_points(M, nz, nphi))
        r['points'] = nz * nz * nphi * 2
        print(f'grid {gspec:9s} ({r["points"]:.1e} points): H = {r["H"]:.3f} ({r["seconds"]:.0f} s)', flush=True)
    for kind, gen in (('rqmc', rqmc_points), ('mc', mc_points)):
        for size in args.sizes.split(','):
            n = int(float(size))
            out[kind][size] = [run(gen(M, n, seed)) for seed in range(args.reps)]
            Hs = [r['H'] for r in out[kind][size]]
            print(f'{kind:4s} n = {n:.0e}: H = {np.mean(Hs):.3f} +- {np.std(Hs, ddof=1):.3f} (sd over {args.reps})', flush=True)
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, 'quadrature.json'), 'w') as f:
        json.dump(out, f)

    # errors against the finest grid
    ref = out['grid'][args.grids.split(',')[-1]]
    H0, g0, h0 = ref['H'], np.array(ref['g']), np.array(ref['h'])
    lines = [f'reference: grid {args.grids.split(",")[-1]}, H0 = {H0:.4f} hits per 1e9 muons',
             f'{"estimator":22s} {"points":>9s} {"H rel. err":>11s} {"grad rel. err":>14s} {"Hess rel. err":>14s}']
    curves = {}
    for kind in ('mc', 'rqmc'):
        for size, rs in out[kind].items():
            eH = np.sqrt(np.mean([(r['H'] / H0 - 1) ** 2 for r in rs]))
            eg = np.sqrt(np.mean([np.sum((np.array(r['g']) - g0) ** 2) for r in rs])) / np.linalg.norm(g0)
            eh = np.sqrt(np.mean([np.sum((np.array(r['h']) - h0) ** 2) for r in rs])) / np.linalg.norm(h0)
            curves.setdefault(kind, []).append((float(size), eH, eg, eh))
            lines.append(f'{kind + " (rms of " + str(len(rs)) + ")":22s} {float(size):9.0e} {eH:11.2e} {eg:14.2e} {eh:14.2e}')
    for gspec, r in out['grid'].items():
        if gspec == args.grids.split(',')[-1]:
            continue
        eH, eg = abs(r['H'] / H0 - 1), np.linalg.norm(np.array(r['g']) - g0) / np.linalg.norm(g0)
        eh = np.linalg.norm(np.array(r['h']) - h0) / np.linalg.norm(h0)
        curves.setdefault('grid', []).append((r['points'], eH, eg, eh))
        lines.append(f'{"grid " + gspec:22s} {r["points"]:9.1e} {eH:11.2e} {eg:14.2e} {eh:14.2e}')
    with open(os.path.join(FIGS, 'quadrature.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    style = {'mc': ('#2a78d6', 'Monte Carlo (now)'), 'rqmc': ('#eb6834', 'randomized QMC (Sobol)'),
             'grid': ('#1baf7a', 'grid in the latent x azimuth x charge')}
    for i, title in enumerate(('expected hits H', 'gradient at u = 0', 'Hessian at u = 0')):
        for kind, pts in curves.items():
            pts = sorted(pts)
            axes[i].plot([p[0] for p in pts], [p[i + 1] for p in pts], '-o', color=style[kind][0], lw=2, ms=4,
                         label=style[kind][1])
        axes[i].set(xscale='log', yscale='log', xlabel='points (muons) per estimate', title=title)
    axes[0].set_ylabel('relative error vs. the finest grid')
    axes[0].legend(fontsize=8)
    fig.suptitle('Integrating the surrogate over the muon flow: sampling vs. quadrature (final surrogate, T = 1)')
    fig.savefig(os.path.join(FIGS, 'quadrature.png'), dpi=150)
    print('saved quadrature.png, quadrature.txt, quadrature.json')
