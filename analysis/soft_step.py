"""The soft muons' variable, simulated: the step between the entrances of M2 and M3.

discover.py finds that the 10-30 GeV hits depend on the entrance widths of M2 and M3 (both
rectangular) through one antisymmetric combination: a wider M2 with a narrower M3 gives more hits,
and equal changes of both nearly cancel. This simulates, on muons with p_z in --band only:

  each width alone (M2 +step, M3 -step), the step widened (M2 +, M3 -) and narrowed (M2 -, M3 +),
  and both widths changed together (+ and -), where the step stays as it is

with the surrogate's prediction (ensemble mean and spread) per p_z bin of pz_bins.npz.
Writes <run>/soft_step_sim<tag>.npz and soft_step_report<tag>.txt; runs with disjoint --offset add up
(discover.py sums every soft_step_sim*.npz).

    python -m analysis.soft_step
"""
import os

import numpy as np
import torch

from analysis.core import Report, cached_hits, feature_cache, members, setup, simulate
import cli
from paths import RUN

parser = cli.add_common(cli.parser(__doc__), fixed=True)
parser.set_defaults(fixed='M0:')
parser.add_argument('--run', default=RUN, help='robustness.py / bins.py --out_dir')
parser.add_argument('--step', type=float, default=0.05, help='relative change of each width')
parser.add_argument('--band', default='10,30', help='simulate muons with p_z in this range (GeV)')
parser.add_argument('--n_band', type=int, default=int(3e8), help='simulated muons in the band, per design')
parser.add_argument('--block', type=int, default=int(2e7), help='sampled muons held in memory')
parser.add_argument('--offset', type=int, default=0, help='first muon (seeds the blocks): disjoint runs add up')
parser.add_argument('--tag', default='', help='suffix of the output files (soft_step_sim<tag>.npz)')
parser.add_argument('--cache_grid', default='201x64', help='muon points of the surrogate predictions')
args = parser.parse_args()

b = np.load(os.path.join(args.run, 'pz_bins.npz'), allow_pickle=True)
lo, hi = map(float, args.band.split(','))
edges = np.asarray(b['edges'], float)
shield, model = setup(args)
L, D, s = shield.labels, shield.dim, args.step / shield.DELTA
A, B = L.index('M2:dXIn'), L.index('M3:dXIn')
p = f'{args.step:.0%}'
DESIGNS = {'nominal': {}, f'M2 +{p}': {A: s}, f'M3 -{p}': {B: -s},
           f'step wider: M2 +{p}, M3 -{p}': {A: s, B: -s}, f'step narrower: M2 -{p}, M3 +{p}': {A: -s, B: s},
           f'both +{p}': {A: s, B: s}, f'both -{p}': {A: -s, B: -s}}
U = np.zeros((len(DESIGNS), D))
for i, ch in enumerate(DESIGNS.values()):
    for k, v in ch.items():
        U[i, k] = v

et = torch.tensor(edges, dtype=torch.float32)
bin_of = lambda raw: torch.bucketize(raw[:, 2].contiguous(), et.to(raw.device))   # noqa: E731
nb = len(edges) + 1
pts = lambda: shield.muons.points(args.cache_grid, 1e9)                           # noqa: E731
pred = np.array([cached_hits(m, feature_cache(m, pts(), bin_of), nb, U) for m in members(args, shield)])
pred = pred / np.maximum(pred[:, :1], 1e-30)


def blocks():
    for o in range(args.offset, args.offset + args.n_band, args.block):
        torch.manual_seed(o)
        yield o, shield.sample(min(args.block, args.offset + args.n_band - o), pz_band=(lo, hi))
        print(f'  {o + args.block - args.offset:,}/{args.n_band:,}', flush=True)


sim = simulate(model, shield, blocks(), U, bin_of, nb)
names = [f'{edges[k - 1]:g}-{edges[k]:g}' for k in range(1, nb - 1)] + [f'{edges[-1]:g}-inf']
keep = [k for k in range(1, nb - 1) if edges[k - 1] >= lo and edges[k] <= hi]
say = Report()
say(f'{args.n_band:,} muons with p_z in {lo:g}-{hi:g} GeV; M2:dXIn {shield.phi_0[A]:.0f} cm, M3:dXIn {shield.phi_0[B]:.0f} cm')
say('Y/Y_0 per bin: surrogate (members\' mean +- sd) | simulator (Poisson)')
say(f'{"design":32s}' + ''.join(f'{names[k - 1]:>30s}' for k in keep) + f'{"band":>30s}')
for i, n in enumerate(DESIGNS):
    row = f'{n:32s}'
    for ks in [[k] for k in keep] + [keep]:
        sv, s0 = sim[i, ks].sum(), sim[0, ks].sum()
        r = sv / s0
        e = r * np.sqrt(1 / max(sv, 1) + (1 / s0 if i else 0))
        pw = (pred[:, i, ks] * sim[0, ks]).sum(1) / s0                 # surrogate ratio, weighted as the simulator
        row += f'  {pw.mean():6.3f}+-{pw.std(ddof=1):.3f} | {r:6.3f}+-{e:.3f}'
    say(row)
say(f'nominal hits: ' + ', '.join(f'{names[k - 1]} {sim[0, k]:.0f}' for k in keep))
np.savez(os.path.join(args.run, f'soft_step_sim{args.tag}.npz'), designs=list(DESIGNS), U=U, sim=sim, pred=pred, edges=edges,
         band=(lo, hi), keep=keep, offset=args.offset, n_band=args.n_band, step=args.step)
say.save(os.path.join(args.run, f'soft_step_report{args.tag}.txt'))
print(f'saved {args.run}/soft_step_sim{args.tag}.npz and soft_step_report{args.tag}.txt')
