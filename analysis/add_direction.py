"""Add one direction to a finished robustness run (robustness.py), without redoing the rest: its
walk on the surrogate (with the members' standard error) and with the simulator, on the same
muons as the run's walks (the same seed and count), appended to <run>/robustness.npz and
reported at the end of robustness_report.txt. plots.py then draws it with the others.

Directions: worst2 (the trust-region worst direction of robustness.py with the parameters of
--fixed held at nominal, from the run's gradient and Hessian).

    python -m analysis.add_direction --name worst2 --fixed M7:
"""
import os

import numpy as np
import torch

from analysis.core import (cached_hits, members, feature_cache, muon_blocks, one_group, restricted_worst, seed_se, setup,
                           simulate, unit)
import cli
from paths import RUN

parser = cli.add_common(cli.parser(__doc__))
parser.add_argument('--run', default=RUN, help='robustness.py --out_dir')
parser.add_argument('--name', default='worst2', help='name of the direction')
parser.add_argument('--fixed', default='M7:', help='comma-separated label prefixes held at nominal')
parser.add_argument('--n_muons_sim', type=int, default=None, help='simulated muons (default: as the run)')
parser.add_argument('--seed', type=int, default=0, help='seed of the sampled muons (0: the run\'s walks)')
parser.add_argument('--block', type=int, default=int(2e7), help='sampled muons held in memory')
parser.add_argument('--no_sim', dest='sim', action='store_false', help='the surrogate only')
parser.add_argument('--redo', action='store_true', help='replace the direction if the run has it already')
args = parser.parse_args()

path = os.path.join(args.run, 'robustness.npz')
r = dict(np.load(path, allow_pickle=True))
names = [str(s) for s in r['dir_names']]
if args.name in names:
    if not args.redo:
        raise SystemExit(f'{path} has {args.name} already (--redo to replace it)')
    raise SystemExit('--redo: not implemented; rerun robustness.py instead')
labels = [str(s) for s in r['labels']]
fixed = [p for p in args.fixed.split(',') if p]
t_max = float(np.abs(r['dir_t'][names.index('worst')]).max())        # robustness.py --t_max
n_pts = r['dir_t'].shape[1]
T = float(r['temperature'])
n_sim = args.n_muons_sim or int(r['n_muons_sim'])

v = unit(restricted_worst(r['grad'], r['hess'], labels, fixed, t_max))
ts = np.linspace(-1, 1, n_pts) * t_max
new = np.stack([np.clip(tt * v, -1, 1) for tt in ts if tt != 0])
at1 = np.clip(t_max * v, -1, 1)[None]
H0 = float(r['H_0'])
print(f'{args.name}: {", ".join(fixed)} held at nominal; cos to worst '
      f'{v @ r["dir_vecs"][names.index("worst")]:+.3f}; largest components: '
      + ', '.join(f'{labels[k]} {v[k]:+.2f}' for k in np.argsort(-np.abs(v))[:6]))

shield, model = setup(args, ','.join(f'{l}' for l in r.get('fixed', [])))     # the run's parameters
assert list(shield.labels) == labels, 'the run has other parameters than the shield'


def ratios(m):
    c = feature_cache(m, shield.muons.points(str(r['cache_grid']), float(r['n_ref']), T), one_group)
    out = cached_hits(m, c, 1, np.concatenate([np.zeros((1, len(v))), new, at1]))[:, 0]
    del c
    torch.cuda.empty_cache()
    return out[1:] / out[0]


main = ratios(model)
mem = [ratios(m) for m in members(args, shield)]
se = seed_se(mem) if mem else np.zeros_like(main)

sim = np.zeros(len(new))
if args.sim:
    print(f'simulating the nominal design and {len(new)} walk points over {n_sim:,} muons (seed {args.seed})')
    h = simulate(model, shield, muon_blocks(shield, n_sim, args.block, args.seed, T),
                 np.concatenate([np.zeros((1, len(v))), new]), one_group, 1)[:, 0]
    sim = h[1:] * float(r['sim'][0]) / h[0] if r['sim'][0] > 0 else h[1:]     # on the run's nominal count
    print(f'nominal: {h[0]:.0f} hits now, {r["sim"][0]:.0f} in the run; the walk is scaled by their ratio')

base = len(r['designs'])
idx = np.zeros(n_pts, dtype=int)
idx[ts != 0] = base + np.arange(len(new))
phis = model.denormalize_phi(torch.as_tensor(new, dtype=torch.float32, device=args.device)).cpu().numpy()
r.update(dir_names=np.append(r['dir_names'], args.name), dir_vecs=np.vstack([r['dir_vecs'], v]),
         dir_t=np.vstack([r['dir_t'], ts]), dir_idx=np.vstack([r['dir_idx'], idx]),
         dir_at_tmax=np.append(r['dir_at_tmax'], main[-1]), dir_at_tmax_se=np.append(r['dir_at_tmax_se'], se[-1]),
         designs=np.vstack([r['designs'], new]), phis=np.vstack([r['phis'], phis]),
         ratio=np.append(r['ratio'], main[:-1]), ratio_se=np.append(r['ratio_se'], se[:-1]),
         pred=np.append(r['pred'], main[:-1] * H0), sim=np.append(r['sim'], sim))
np.savez(path, **r)

g = r['grad']
lines = ['', '=' * 79, f'ADDED DIRECTION {args.name}  (analysis/add_direction.py; held at nominal: {", ".join(fixed)})',
         '=' * 79,
         f'cos(grad) {g @ v / np.linalg.norm(g):+.3f}, slope/H_0 {g @ v / H0:+.3f}, curv/H_0 {v @ r["hess"] @ v / H0:+.3f}, '
         f'H/H_0 at t={t_max:g}: {main[-1]:.4g} +- {se[-1]:.4g}',
         'largest components: ' + ', '.join(f'{labels[k]} {v[k]:+.2f}' for k in np.argsort(-np.abs(v))[:8])]
for j, (tt, i) in enumerate(zip(ts, idx)):
    if not i:
        continue
    k = i - base
    line = f'  t={tt:+.3f}  surrogate {main[k]:.4g} +- {se[k]:.3g}'
    if args.sim:
        sr = sim[k] / r['sim'][0]
        sr_se = sr * np.sqrt(1 / max(sim[k], 1) + 1 / r['sim'][0])
        line += f'  simulator {sr:.4g} +- {sr_se:.3g}  z {(main[k] - sr) / np.hypot(se[k], sr_se):+.1f}'
    lines.append(line)
with open(os.path.join(args.run, 'robustness_report.txt'), 'a') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'saved {path}')
