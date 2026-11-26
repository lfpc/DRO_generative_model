"""The rise along the worst direction is collective: the unit step, taken apart.

One step of unit length along the worst direction (the metric of the walks: u_i = +-1 is
parameter i at +-10% of nominal, |u| = 1) moves every parameter by its share of the step,
10 v_i %. This simulates, all on the same --n_muons sampled muons:

  worst      the unit step itself, all parameters together
  top<k>     the unit step with only its k largest changes kept (the rest at nominal)
  share<i>   the i-th largest change alone, by the same amount as in the step (10 v_i %)
  full<i>    the i-th largest change alone, at 10% in the same direction (u_i = sign v_i)

so the hits of each parameter alone, at its share and at a full 10%, are compared with the hits
of the parameters moved together by less. "If independent" is the product of the shares' ratios.
Writes <run>/collective.npz and collective_report.txt (plots: collective.png).

    python -m analysis.collective --n_muons 1000000000
"""
import os

import numpy as np

from analysis.core import Report, cached_hits, feature_cache, muon_blocks, setup, simulate
import cli
from paths import RUN

parser = cli.add_common(cli.parser(__doc__), fixed=True)
parser.set_defaults(fixed='M0:')
parser.add_argument('--run', default=RUN, help='robustness.py --out_dir')
parser.add_argument('--direction', default='worst', help='direction of robustness.npz')
parser.add_argument('--t', type=float, default=1.0, help='step length (normalized units)')
parser.add_argument('--top_k', default='1,2,3,5,10,20', help='top<k> designs')
parser.add_argument('--n_single', type=int, default=10, help='largest changes simulated alone')
parser.add_argument('--n_muons', type=int, default=int(1e9), help='sampled muons for the simulator; the surrogate is quoted per as many')
parser.add_argument('--seed', type=int, default=0, help='0 = the muons of robustness.py')
parser.add_argument('--grid', default='201x64', help='muon points of the surrogate')
parser.add_argument('--no_sim', dest='sim', action='store_false', help='surrogate only')
parser.add_argument('--block', type=int, default=int(2e7), help='sampled muons held in memory')
args = parser.parse_args()

r = np.load(os.path.join(args.run, 'robustness.npz'))
shield, model = setup(args)
D, labels = model.dim, shield.labels
NAMES = [str(s) for s in r['dir_names']]
v = r['dir_vecs'][NAMES.index(args.direction)]
v = args.t * v / np.linalg.norm(v)
order = np.argsort(-np.abs(v))

DESIGNS = [('nominal', np.zeros(D)), (args.direction, v)]
for k in map(int, args.top_k.split(',')):
    DESIGNS.append((f'top{k}', np.where(np.isin(np.arange(D), order[:k]), v, 0)))
for i, k in enumerate(order[:args.n_single]):
    DESIGNS.append((f'share{i + 1}', np.eye(D)[k] * v[k]))
for i, k in enumerate(order[:args.n_single]):
    DESIGNS.append((f'full{i + 1}', np.eye(D)[k] * np.sign(v[k])))
U = np.stack([u for _, u in DESIGNS])

zero = lambda raw: raw.new_zeros(len(raw), dtype=int)            # noqa: E731 one group
cache = feature_cache(model, shield.muons.points(args.grid, args.n_muons), zero)
pred = cached_hits(model, cache, 1, U)[:, 0]
sim = np.zeros(len(U))
if args.sim:
    print(f'simulating {len(U)} designs x {args.n_muons:,} muons')
    sim = simulate(model, shield, muon_blocks(shield, args.n_muons, args.block, args.seed), U, zero, 1)[:, 0]

say = Report()
say(f'{args.n_muons:,} muons {args.muons} (seed {args.seed}); {args.direction}, t = {args.t:g}, from {args.run}')
say(f'nominal: simulator {sim[0]:.0f} +- {np.sqrt(sim[0]):.0f} hits, surrogate {pred[0]:.0f}\n')
say(f'{"design":10s} {"parameter / group":28s} {"change":>8s} {"sim Y/Y_0":>16s} {"surrogate":>9s}')
for d, (name, u) in enumerate(DESIGNS[1:], 1):
    k = int(np.argmax(np.abs(u)))
    what = labels[k] if name[:4] in ('shar', 'full') else f'{int((u != 0).sum())} parameters'
    se = sim[d] / sim[0] * np.sqrt(1 / max(sim[d], 1) + 1 / sim[0]) if args.sim else 0
    say(f'{name:10s} {what:28s} {10 * u[k]:+7.2f}% {sim[d] / max(sim[0], 1):7.3f} +- {se:5.3f} {pred[d] / pred[0]:9.3f}')
n_s = args.n_single
S = sim / max(sim[0], 1)
P = pred / pred[0]
idx = {nm: d for d, (nm, _) in enumerate(DESIGNS)}
say('\nif independent (product of the shares alone) vs together:')
for k in map(int, args.top_k.split(',')):
    if k <= n_s:
        sh = [idx[f'share{i + 1}'] for i in range(k)]
        say(f'  top{k:<3d} product {np.prod(S[sh]):.3f} (surrogate {np.prod(P[sh]):.3f}), together {S[idx[f"top{k}"]]:.3f} '
            f'({P[idx[f"top{k}"]]:.3f})')

np.savez(os.path.join(args.run, 'collective.npz'), names=np.array([nm for nm, _ in DESIGNS]), U=U, v=v,
         labels=np.array(labels), sim=sim, pred=pred, simulated=args.sim, n_muons=args.n_muons,
         direction=args.direction, t=args.t, grid=args.grid)
say.save(os.path.join(args.run, 'collective_report.txt'))
print(f'saved {args.run}/collective.npz and collective_report.txt')
