"""Would importance sampling make the simulator's estimates cheaper?

Sample the muons at momentum temperature T and weight each by the density ratio to T = 1. The
momentum flow maps its latent z to (pz, pt) the same way at every T, and only the latent's
spread changes (N(0, T^2 I) instead of N(0, I)), so the weight is exact and closed form:

    w(z) = T^2 exp(-|z|^2 (1 - 1/T^2) / 2)

(times the ratio of the envelope acceptances, a constant). The T = 1 hit rate is
H = E_T[w hit], and per simulated muon the relative variance is E_T[w^2 hit] / H^2 - 1, against
1 / H - 1 for plain sampling at T = 1. Their ratio is the gain: how many times fewer simulated
muons give the same precision.

Every hit at the nominal design is stored in the pools (shield_pool.py; the nominal designs are
the first --n_nominal of each pool), with its muon, so the latent of every hit, and with it the
gain, follows without new simulation.

    python importance.py        # -> outputs/surrogate_study/figs/importance.txt
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # after this folder: both have a plots.py
from shield import MuonShield  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, 'outputs/surrogate_study/shield_pool')
FIGS = os.path.join(ROOT, 'outputs/surrogate_study/figs')
POOLS = {1.0: 'T1_d1', 1.5: 'T1.5_d1', 2.0: 'T2_m1e6', 3.0: 'T3_m1e6'}


@torch.no_grad()
def latent(gen, mu):
    """The momentum flow's latent z (n, 2) of raw muons (n, 7)."""
    pz, pt = mu[:, 2].double(), mu[:, :2].double().norm(dim=1)
    m = gen.m
    p = torch.hypot(pz, pt)
    uv = torch.stack([torch.log(p - m['p_min']) - torch.log(m['p_max'] - p), torch.log(pt / pz)], 1)
    y = ((uv - torch.as_tensor(gen.mean[:2])) / torch.as_tensor(gen.std[:2])).float().to(next(gen.flow.parameters()).device)
    for layer in gen.flow.momentum.layers:
        y, _ = layer(y, None)
    return y.double().cpu()


def accept(gen, T, n=2_000_000, seed=0):
    """Fraction of the flow's muons at temperature T inside shield.sample's envelope."""
    torch.manual_seed(seed)
    s = gen.flow.momentum.sample(n, temperature=T).double().cpu().numpy() * gen.std[:2] + gen.mean[:2]
    m = gen.m
    p = m['p_min'] + (m['p_max'] - m['p_min']) / (1 + np.exp(-s[:, 0]))
    pz = p / np.sqrt(1 + np.exp(2 * s[:, 1]))
    pt = pz * np.exp(s[:, 1])
    return float(np.mean((p <= MuonShield.P_MAX) & (pt <= MuonShield.PT_MAX) & (pz > MuonShield.PZ_MIN)))


if __name__ == '__main__':
    shield = MuonShield()
    shield._load_flow()
    gen = shield._flow
    a1 = accept(gen, 1.0)
    lines = ['importance sampling of the simulator at the nominal design: muons drawn at momentum temperature T, '
             'weighted to T = 1', f'{"T":>4s} {"designs":>8s} {"muons":>10s} {"hits":>9s} {"H(T=1) per 1e9":>15s} '
             f'{"ESS / hits":>10s} {"gain":>8s}']
    for T, name in POOLS.items():
        shards = [torch.load(f, weights_only=False) for f in sorted(glob.glob(os.path.join(POOL, name, 'shard_*.pt')))]
        n_nom = shards[0]['args']['n_nominal']
        n_mu = shards[0]['args']['n_muons']
        d = torch.cat([t for s in shards for t in s['d']])
        x = torch.cat([t for s in shards for t in s['x']])
        y = torch.cat([t for s in shards for t in s['y']])
        sel = (d < n_nom) & (y > 0)                       # every hit of the nominal designs
        n_total = n_nom * n_mu
        z = latent(gen, x[sel])
        w = T ** 2 * torch.exp(-(z ** 2).sum(1) * (1 - 1 / T ** 2) / 2) * accept(gen, T) / a1
        H = float(w.sum()) / n_total                       # hit rate at T = 1
        m2 = float((w ** 2).sum()) / n_total
        rv_is, rv_mc = m2 / H ** 2 - 1, 1 / H - 1
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        lines.append(f'{T:4.1f} {n_nom:8d} {n_total:10.1e} {int(sel.sum()):9d} {H * 1e9:15.1f} '
                     f'{ess / max(int(sel.sum()), 1):10.3f} {rv_mc / rv_is:8.1f}')
    lines.append('gain: plain T = 1 sampling needs this many times more simulated muons for the same precision on H; '
                 'ESS / hits: the weighted hits\' effective share')
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, 'importance.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
