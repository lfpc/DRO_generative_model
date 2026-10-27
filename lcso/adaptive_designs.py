"""Designs for an adaptive training pool: where the surrogate's conclusions rest and where it is
least sure. Three sources:

  worst cases   every design of analysis/tolerance_boxes.py (box worst case per tolerance, allocation worst
                cases), plus --n_ray points on the line from the nominal design to it and
                --n_near perturbations within its box (u* + --sigma * e * N(0, I), clipped to the
                box of half-widths e). Designs the simulator shows mispredicted (|log pred/sim|
                beyond --n_sigma combined sigma) get twice the perturbations.
  disagreement  the --n_disagree designs, among --n_cand random ones in each of the +-10%, +-5%
                and +-3% boxes, where the ensemble's members disagree most (sd of log H/H_0).
                The members share their biases, so this finds extrapolation, not bias.

Writes --out (n, D) and its .txt summary; simulate it as a pool:

    python -m lcso.adaptive_designs --run outputs/robustness_first --model outputs/lcso_first.pt
    python -m lcso.simulate_pools --name T1.5_adapt --designs outputs/adaptive_designs.npy --n_nominal 0 --seed 13
"""
import os

import numpy as np

from analysis.core import cached_hits, feature_cache, one_group, setup
import cli
from paths import RUN, output

from .surrogate import load_members

if __name__ == '__main__':
    parser = cli.add_common(cli.parser(__doc__))
    parser.add_argument('--run', default=RUN, help='with tolerance_boxes*.npz (analysis/tolerance_boxes.py)')
    parser.add_argument('--out', default=output('adaptive_designs.npy'))
    parser.add_argument('--n_ray', type=int, default=4, help='points on the line to each worst case')
    parser.add_argument('--n_near', type=int, default=12, help='perturbations of each worst case (x2 if mispredicted)')
    parser.add_argument('--sigma', type=float, default=0.25, help='perturbation size, in units of the box half-width')
    parser.add_argument('--n_sigma', type=float, default=3.0, help='"mispredicted" threshold')
    parser.add_argument('--n_disagree', type=int, default=64)
    parser.add_argument('--n_cand', type=int, default=4096, help='random candidates per box')
    parser.add_argument('--cache_grid', default='101x32', help='muon points of the members (coarse is enough to rank)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    t = np.load(os.path.join(args.run, 'tolerance_boxes.npz'))
    ev = np.load(os.path.join(args.run, 'tolerance_boxes_eval.npz'))
    names, U = [str(s) for s in t['names']], t['U']
    D = U.shape[1]
    lines, out = [], []

    # --- worst cases and their neighbourhoods
    path_h = np.exp(t['path_h'])
    for i, (name, u) in enumerate(zip(names, U)):
        if name.startswith('box'):
            e = np.full(D, float(name[3:-1]) / 10)
        else:                                       # alloc<target>: the box of that allocation step
            e = t['path_e'][int(np.argmax(path_h <= float(name[5:])))]
        bad = False
        if np.isfinite(ev['sim'][i]):
            sr = ev['sim'][i] / ev['sim0']
            sr_se = sr * np.sqrt(1 / max(ev['sim'][i], 1) + 1 / ev['sim0'])
            z = np.log(ev['pred'][i] / sr) / np.hypot(ev['pred_se'][i] / ev['pred'][i], sr_se / sr)
            bad = abs(z) > args.n_sigma
        n_near = args.n_near * (2 if bad else 1)
        ray = np.linspace(0, 1, args.n_ray + 2)[1:-1, None] * u
        near = np.clip(u + args.sigma * e * rng.standard_normal((n_near, D)), -e, e)
        out += [u[None], ray, near]
        lines.append(f'{name:12s} {1 + len(ray) + len(near):3d} designs'
                     + (f'; simulator z = {z:+.1f}' if np.isfinite(ev['sim'][i]) else '; not simulated')
                     + (' (mispredicted)' if bad else ''))

    # --- where the members disagree most
    shield, model = setup(args)
    cand = np.concatenate([rng.uniform(-1, 1, (args.n_cand, D)) * x for x in (1.0, 0.5, 0.3)])
    logs = []
    for m in load_members(args.model, shield.phi_0, args.device):
        c = feature_cache(m, shield.muons.points(args.cache_grid), one_group)
        h = cached_hits(m, c, 1, np.concatenate([np.zeros((1, D)), cand]))[:, 0]
        logs.append(np.log(h[1:] / h[0]))
        del c
    sd = np.std(logs, axis=0)
    top = np.argsort(-sd)[:args.n_disagree]
    out.append(cand[top])
    box = np.abs(cand[top]).max(1)
    lines.append(f'disagreement {len(top):3d} designs; members\' sd of log H/H_0 {sd[top].min():.2f}-{sd[top].max():.2f} '
                 f'(median of all candidates {np.median(sd):.2f}); from the +-10/5/3% boxes: '
                 + '/'.join(str(int(((box > lo) & (box <= hi)).sum())) for lo, hi in ((0.5, 1), (0.3, 0.5), (0, 0.3))))

    U_out = np.concatenate(out).astype(np.float32)
    lines.insert(0, f'{len(U_out)} designs for the adaptive pool')
    np.save(args.out, U_out)
    with open(os.path.splitext(args.out)[0] + '.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
