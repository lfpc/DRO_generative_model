"""Would a targeted latent proposal give the surrogate's training many more useful hits than the
momentum temperature? Checked without simulating, with the surrogate itself.

On the momentum flow's 2-d latent grid, r(z) is the surrogate's hit probability averaged over
the azimuth, the charge and --n_designs random designs in the +-10% box (plus the nominal one).
A training proposal q(z) then yields, per simulated muon, E_q[r] hits, of which E_q[r 1_band]
in a momentum band (default 30-75 GeV, where the nominal hits are and the surrogate is weakest).
Compared, per simulated muon and relative to T = 2 (the hot pool):
  T = t        the latent scaled by t (what the pools do)
  bump         a mixture (1 - a) N(0, T0^2 I) + a N(mu, s^2 I), mu and s chosen on a grid to
               maximize the band's hits
and, as the price, the share of the proposal's muons outside the band (the low-probability
region that the 1e-6 rates must be learned from).

Hits per muon reward muons where p(hit) is high, but the objective H = int p pi is made of
the core muons with pi ~ 1e-6. So the figure of merit is an error proxy: the surrogate's relative
error on H if each latent cell's pi is learned from the training hits in it,
    E(q) = sum_cells (p pi / H) / sqrt(N q pi),
and the gain of q over T = 2 in effective training muons, (E(T=2) / E(q))^2. It is minimized by
q* ~ p^(2/3) pi^(1/3) (a mild temperature, sqrt(3/2) for a Gaussian, times a gentle tilt towards
high pi), the bound any proposal can reach.

    python proposal_check.py --device cpu       # -> outputs/surrogate_study/figs/proposal_check.txt
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

from quadrature import Muons

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # after this folder: both have a plots.py
from shield import MuonShield  # noqa: E402
from lcso.surrogate import load_surrogate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def latent_grid(shield, n_z, L):
    """Latent points (n_z^2, 2), their cell area, and their pz, pt and envelope indicator (CPU)."""
    zs = torch.linspace(-L, L, n_z, dtype=torch.float64)
    Z = torch.stack(torch.meshgrid(zs, zs, indexing='ij'), -1).view(-1, 2)
    return (Z, float(zs[1] - zs[0]) ** 2) + Muons(shield, 'cpu').momentum(Z)


@torch.no_grad()
def hit_rate(model, pz, pt, U, mu_plus, n_phi, dev, chunk=4096):
    """r(z, u): p(hit) averaged over azimuth and charge, at every latent point and design U: (n, m)."""
    b = model.branch_net(torch.as_tensor(U, dtype=torch.float32, device=dev))       # (m, P)
    phis = (torch.arange(n_phi, dtype=torch.float64) + 0.5) * 2 * math.pi / n_phi
    out = torch.zeros(len(pz), len(U), dtype=torch.float64)
    for c in range(0, len(pz), chunk):
        a, t = pz[c:c + chunk], pt[c:c + chunk]
        k = len(a)
        A, Tt, PH = a.repeat_interleave(n_phi), t.repeat_interleave(n_phi), phis.repeat(k)
        acc = 0.0
        for q, pq in ((1.0, mu_plus), (-1.0, 1 - mu_plus)):
            z0 = torch.zeros_like(A)
            x = torch.stack([Tt * torch.cos(PH), Tt * torch.sin(PH), A, z0, z0, z0 - 1, z0 - 13 * q], 1).float().to(dev)
            pr = torch.sigmoid(model.features(x) @ b.T + model.bias).double()               # (k n_phi, m)
            acc = acc + pq * pr.view(k, n_phi, -1).mean(1).cpu()
        out[c:c + chunk] = acc
    return out


def normal(Z, mu, s):
    return torch.exp(-((Z - torch.as_tensor(mu, dtype=Z.dtype)) ** 2).sum(1) / (2 * s * s)) / (2 * math.pi * s * s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=type(
        'F', (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter), {}))
    parser.add_argument('--model', default=os.path.join(ROOT, 'outputs/lcso.pt'))
    parser.add_argument('--n_z', type=int, default=161)
    parser.add_argument('--L', type=float, default=8.0, help='latent half-width of the grid')
    parser.add_argument('--n_phi', type=int, default=16)
    parser.add_argument('--n_designs', type=int, default=64)
    parser.add_argument('--band', default='30,75', help='p_z band in GeV')
    parser.add_argument('--mix', type=float, default=0.5, help='weight a of the bump')
    parser.add_argument('--T0', type=float, default=2.0, help='temperature of the rest of the mixture')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(16)
    shield = MuonShield()
    model = load_surrogate(args.model, shield.phi_0, args.device)
    Z, dA, pz, pt, ok = latent_grid(shield, args.n_z, args.L)
    U = np.concatenate([np.zeros((1, 43)), np.random.default_rng(0).uniform(-1, 1, (args.n_designs, 43))])
    r = hit_rate(model, pz, pt, U, shield._flow.mu_plus, args.n_phi, args.device) * ok[:, None]
    lo, hi = map(float, args.band.split(','))
    band = ((pz >= lo) & (pz < hi)).double()

    p_cell = normal(Z, (0, 0), 1.0) * dA * ok
    p_cell = p_cell / p_cell.sum()
    pr = p_cell[:, None] * r                                   # (cells, designs): the nominal hits

    def yields(q):
        """Per design: (hits per muon, band hits per muon, error proxy E(q)), and the band's share."""
        w = q * dA * ok
        w = w / w.sum()
        E = (pr / pr.sum(0) / torch.sqrt((w[:, None] * r).clamp_min(1e-300))).sum(0)
        return ((w[:, None] * r).sum(0).numpy(), (w[:, None] * r * band[:, None]).sum(0).numpy(),
                E.numpy(), float((w * band).sum()))

    rows = {f'T = {t:g}': yields(normal(Z, (0, 0), t)) for t in (1.0, 1.5, 2.0, 3.0)}
    ref = rows['T = 2']
    for widths in ((0.25, 0.5, 1.0), (0.5, 1.0)):        # the best bump; and the best not narrower than 0.5
        best = None
        for mx in np.linspace(-4, 4, 33):
            for my in np.linspace(-4, 4, 33):
                for s in widths:
                    q = (1 - args.mix) * normal(Z, (0, 0), args.T0) + args.mix * normal(Z, (mx, my), s)
                    y = yields(q)
                    key = np.median((ref[2] / y[2]) ** 2)                 # the median design's effective-muon gain
                    if best is None or key > best[0]:
                        best = (key, y, (mx, my, s))
        mx, my, s = best[2]
        rows[f'bump: {args.mix:g} N(({mx:+.2f},{my:+.2f}), {s:g}^2) + {1 - args.mix:g} T = {args.T0:g}'] = best[1]
    # the bound: q* ~ p^(2/3) pi^(1/3), pi averaged over the designs
    qs = normal(Z, (0, 0), 1.0) ** (2 / 3) * r.mean(1).clamp_min(1e-300) ** (1 / 3)
    rows['bound: q* ~ p^(2/3) pi^(1/3)'] = yields(qs / (qs * dA).sum())
    lines = [f'surrogate {args.model}: per simulated muon, for {args.n_designs} box designs + nominal, relative to '
             f'T = 2 design by design; band p_z {lo:g}-{hi:g} GeV',
             'effective muons: (E(T=2) / E(q))^2, the factor on the training muons for the same error on H', '',
             f'{"proposal":52s} {"hits":>6s} {"band hits":>10s} {"effective muons: median":>24s} {"10%":>6s} {"nominal":>8s} {"in band":>8s}']
    for name, (h, hb, E, fb) in rows.items():
        eff = (ref[2] / E) ** 2
        lines.append(f'{name:52s} {np.median(h / ref[0]):6.2f} {np.median(hb / ref[1]):10.2f} {np.median(eff):24.2f} '
                     f'{np.quantile(eff, .1):6.2f} {eff[0]:8.2f} {fb:8.1%}')
    out = os.path.join(ROOT, 'outputs/surrogate_study/figs/proposal_check.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
