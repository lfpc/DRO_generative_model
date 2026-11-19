"""Where the extra hits come from: simulated hits per p_z range of a few designs, on the same
muons, for the report's figure sim_pz_profiles (drawn by plots.py).

Designs (--designs): nominal; <dir> for a direction of robustness.py at --t; any design of
worst_cases.py by its name (single1: the most sensitive parameter alone at its worse limit); any
design of tolerance_boxes.py by its name (box10%: the worst design in the +-10% box). Writes
<run>/pz_profiles.npz.

    python -m analysis.pz_profiles --designs nominal,ascent,worst,act1,box10%,single1
"""
import os

import numpy as np
import torch

from analysis.core import cached_hits, feature_cache, muon_blocks, setup, simulate
import cli
from paths import RUN

parser = cli.add_common(cli.parser(__doc__), fixed=True)
parser.add_argument('--run', default=RUN, help='robustness.py --out_dir, with worst_cases.npz')
parser.add_argument('--designs', default='nominal,ascent,worst,act1,box10%,single1')
parser.add_argument('--t', type=float, default=1.0, help='step along the directions (1 = one tolerance unit)')
parser.add_argument('--n_muons', type=int, default=int(1e9), help='simulated muons')
parser.add_argument('--seed', type=int, default=0, help='0: the muons of robustness.py and worst_cases.py')
parser.add_argument('--block', type=int, default=int(2e7), help='sampled muons held in memory')
parser.add_argument('--cache_grid', default='201x64', help='muon points of the surrogate prediction')
args = parser.parse_args()

r = np.load(os.path.join(args.run, 'robustness.npz'))
w = np.load(os.path.join(args.run, 'worst_cases.npz'), allow_pickle=True)
tb = np.load(os.path.join(args.run, 'tolerance_boxes.npz'))
dirs, wnames, tnames = [str(s) for s in r['dir_names']], [str(s) for s in w['names']], [str(s) for s in tb['names']]
PZ = w['pz_edges']
names, U, kind = args.designs.split(','), [], []
for n in names:
    if n == 'nominal':
        U.append(np.zeros(len(r['labels'])))
        kind.append('nominal')
    elif n in dirs:
        U.append(np.clip(args.t * r['dir_vecs'][dirs.index(n)], -1, 1))
        kind.append(f'direction at t={args.t:g}')
    elif n in wnames:
        U.append(w['U'][wnames.index(n)])
        kind.append('worst_cases.py design')
    elif n in tnames:
        U.append(tb['U'][tnames.index(n)])
        kind.append('tolerance_boxes.py design')
    else:
        raise SystemExit(f'unknown design {n!r}: nominal, a direction of robustness.py, a design of worst_cases.py '
                         'or of tolerance_boxes.py')
U = np.stack(U)

shield, model = setup(args)


def pz_bin(raw):
    return torch.bucketize(raw[:, 2].contiguous(), torch.as_tensor(PZ[1:-1], dtype=raw.dtype, device=raw.device))


NB = len(PZ) - 1
pred = cached_hits(model, feature_cache(model, shield.muons.points(args.cache_grid, args.n_muons), pz_bin), NB, U)
torch.cuda.empty_cache()
sim = simulate(model, shield, muon_blocks(shield, args.n_muons, args.block, args.seed), U, pz_bin, NB)
out = os.path.join(args.run, 'pz_profiles.npz')
np.savez(out, names=np.array(names), kind=np.array(kind), U=U, pz_edges=PZ, sim_bin=sim, pred_bin=pred,
         n_muons=args.n_muons, t=args.t, labels=r['labels'])
for n, s, p in zip(names, sim.sum(1), pred.sum(1)):
    print(f'{n:10s} simulated {s:10.0f} hits (Y/Y_0 {s / sim[0].sum():8.3f}), surrogate Y/Y_0 {p / pred[0].sum():8.3f}')
print(f'saved {out}')
