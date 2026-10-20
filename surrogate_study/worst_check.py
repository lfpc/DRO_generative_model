"""Do the worst directions of different surrogates differ in a way that matters? Simulate them.

Candidates (u, u_i = +-1 at +-10%), each at |u| = 1 and 2 (the box itself reaches |u| = 6.6):
  report    the worst direction of the report's robustness run (outputs/robustness_deeponet)
  existing  the trust-region worst of lcso_deeponet.pt's gradient and Hessian (derivative_check.py)
  consensus the same for the median gradient and Hessian of the 8 best new models
  random    two random directions, for scale
and the box worst cases (every |u_i| <= 1): the report's projected-gradient design and the
consensus quadratic's. Simulated on the stored reference muons of shield_ref.py.

    python worst_check.py designs        # -> <figs>/worst_check_designs.npz
    python shield_ref.py --name T1 --stage simulate --designs <that> --tag worst --worker 0 --n_workers 2
    python worst_check.py report         # -> <figs>/worst_check.txt
"""
import glob
import os
import sys

import numpy as np
import torch

from plots import FIGS
import shield_ref

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # after this folder: both have a plots.py
from analysis import trust_region_max  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(FIGS, 'worst_check_designs.npz')


def box_max(g, H, steps=3000, lr=0.01, seed=0):
    """argmax of g.u + u'Hu/2 over the box |u_i| <= 1, projected gradient ascent from 64 starts."""
    rng = np.random.default_rng(seed)
    best, best_f = None, -np.inf
    for u in np.vstack([np.sign(g), rng.uniform(-1, 1, (63, len(g)))]):
        for _ in range(steps):
            u = np.clip(u + lr * (g + H @ u), -1, 1)
        f = g @ u + 0.5 * u @ H @ u
        if f > best_f:
            best, best_f = u, f
    return best


def candidates():
    chk = np.load(os.path.join(FIGS, 'derivatives_check.npz'))
    names = [str(n) for n in chk['names']]
    g, H = chk['g'], chk['H']
    new = [i for i, n in enumerate(names) if n != 'existing']
    g_c, H_c = np.median(g[new], 0), np.median(H[new], 0)
    ex = names.index('existing')
    rep = np.load(os.path.join(ROOT, 'outputs/robustness_deeponet/robustness.npz'), allow_pickle=True)
    dn = [str(s) for s in rep['dir_names']]
    out = {}
    for r in (1.0, 2.0):
        out[f'report@{r:g}'] = r * rep['dir_vecs'][dn.index('worst')] / np.linalg.norm(rep['dir_vecs'][dn.index('worst')])
        out[f'existing@{r:g}'] = trust_region_max(g[ex], H[ex], r)
        out[f'consensus@{r:g}'] = trust_region_max(g_c, H_c, r)
    rng = np.random.default_rng(3)
    for i in range(2):
        v = rng.standard_normal(43)
        out[f'random{i}@2'] = 2 * v / np.linalg.norm(v)
    out['report box'] = rep['designs'][rep['dir_idx'][dn.index('pga')][-1]]
    out['consensus box'] = box_max(g_c, H_c)
    return out, g, H, names, g_c, H_c


if __name__ == '__main__':
    stage = sys.argv[1]
    C, g, H, names, g_c, H_c = candidates()
    if stage == 'designs':
        np.savez(PATH, names=np.array(list(C)), U=np.array(list(C.values())))
        print(f'{len(C)} designs -> {PATH}')
        for n, u in C.items():
            print(f'  {n:16s} |u| {np.linalg.norm(u):.2f}  max|u_i| {np.abs(u).max():.2f}  '
                  f'cos(report) {u @ C["report@1"] / np.linalg.norm(u) / np.linalg.norm(C["report@1"]):+.2f}  '
                  f'cos(consensus) {u @ C["consensus@1"] / np.linalg.norm(u):+.2f}')
        raise SystemExit
    lines = []
    for ref in ('T1', 'T1.5'):
        d = os.path.join(shield_ref.OUT, ref)
        hits = {}
        for f in glob.glob(os.path.join(d, 'shard_worst_*.pt')):
            sh = torch.load(f, weights_only=False)
            for i, h in sh['hits'].items():
                hits[list(C)[i]] = h.sum()
        if not hits:
            continue
        rn, rU, rh, n, _ = shield_ref.load(ref)
        h0 = rh[np.array([x == 'nominal' for x in rn])]
        s0, e0 = h0.mean(), np.sqrt(h0.mean() / len(h0))
        lines.append(f'\nreference {ref} ({n:.0e} muons): nominal {s0:.0f} +- {e0:.0f} hits (mean of {len(h0)} runs)')
        lines.append(f'  {"design":16s} {"sim hits":>9s} {"sim H/H0":>15s}  predicted rise (quadratic, T = 1): consensus / existing')
        for name, u in C.items():
            if name not in hits:
                continue
            h = hits[name]
            rt = h / s0
            q = lambda gg, HH: (gg @ u + 0.5 * u @ HH @ u)                          # noqa: E731
            pred = f'{1 + q(g_c, H_c) / s0:6.3f} / {1 + q(g[names.index("existing")], H[names.index("existing")]) / s0:6.3f}' \
                if ref == 'T1' else ''
            lines.append(f'  {name:16s} {h:9.0f} {rt:7.3f} +- {rt * np.sqrt(1 / h + (e0 / s0) ** 2):5.3f}  {pred}')
    with open(os.path.join(FIGS, 'worst_check.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
