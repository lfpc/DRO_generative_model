"""Concrete worst cases of the muon shield, simulated: what is changed, and what it does.

Takes the directions of a robustness.py run and simulates, all on the same --n_muons sampled
muons at temperature 1, a set of concrete designs (the surrogate's predictions are integrated
over the muon flow on a quadrature grid, quoted per as many muons):

  worst@X%     the worst direction, scaled so that no parameter moves by more than X% (X < 0:
               the opposite side)
  worst-top<k> worst@10% with only its k largest changes kept: how many parameters the rise needs
  single<i>    the i-th most sensitive parameter (surrogate) alone at its worse +-10% face
  box, box/2   the worst design in the +-10% box (pga), and halfway to it
  rand<X>%_<i> random designs with every parameter uniform in +-X%, next to the surrogate's
               Monte Carlo of --n_mc such designs

Reports which parameters each design changes, by how much, and the simulated and predicted
hits, also per p_z bin. Writes <run>/worst_cases.npz and worst_cases_report.txt.

    python worst_cases.py --run outputs/robustness --n_muons 1000000000
    python worst_cases.py --no_sim                                         # surrogate only
"""
import argparse
import os

import numpy as np
import torch

from analysis import Report, cached_hits, feature_cache, grid_spec, muon_blocks, setup, simulate


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument('--run', default='outputs/robustness', help='robustness.py --out_dir')
parser.add_argument('--model', default='outputs/lcso.pt')
parser.add_argument('--n_muons', type=int, default=int(1e9), help='sampled muons for the simulator; the surrogate is quoted per as many')
parser.add_argument('--seed', type=int, default=0, help='0 = the muons of robustness.py')
parser.add_argument('--grid', default='201x64', help='quadrature grid of the surrogate: latent points per side x azimuths')
parser.add_argument('--pcts', default='1,2,3,5,10,-5,-10', help='worst@X%% designs')
parser.add_argument('--top_k', default='1,3,5,10', help='worst-top<k> designs')
parser.add_argument('--n_single', type=int, default=8, help='single-parameter designs')
parser.add_argument('--random', default='2:8,5:8,10:8', help='X:n = n simulated random designs in +-X%%')
parser.add_argument('--n_mc', type=int, default=4096, help='surrogate-only random designs per X')
parser.add_argument('--pz_edges', default='1,10,15,20,25,30,40,50,60,75,100,150,400', help='GeV')
parser.add_argument('--no_sim', dest='sim', action='store_false', help='surrogate only')
parser.add_argument('--block', type=int, default=int(2e7), help='sampled muons held in memory')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

r = np.load(os.path.join(args.run, 'robustness.npz'))
shield, model = setup(args.model, args.device)
D, labels = model.dim, shield.labels
NAMES = [str(s) for s in r['dir_names']]

# =============================================================================
# The designs (normalized: u_i = +-1 is parameter i at +-10% of its nominal value)
# =============================================================================
DESIGNS = [('nominal', np.zeros(D))]
v = r['dir_vecs'][NAMES.index('worst')]
v = v / np.abs(v).max()                                  # largest component 1 = one parameter at 10%
DESIGNS += [(f'worst@{x:g}%', v * x / 10) for x in map(float, args.pcts.split(','))]
for k in map(int, args.top_k.split(',')):
    top = np.argsort(-np.abs(v))[:k]
    DESIGNS.append((f'worst-top{k}', np.where(np.isin(np.arange(D), top), v, 0)))
for i, k in enumerate(np.argsort(-np.maximum(r['oat_plus'], r['oat_minus']))[:args.n_single]):
    DESIGNS.append((f'single{i + 1}', np.eye(D)[k] * (1 if r['oat_plus'][k] >= r['oat_minus'][k] else -1)))
if 'pga' in NAMES:
    box = r['designs'][r['dir_idx'][NAMES.index('pga')][-1]]
    DESIGNS += [('box/2', box / 2), ('box', box)]
rng = np.random.default_rng(args.seed)
RAND = {}                                                # X -> indices of its random designs
for spec in args.random.split(','):
    x, m = float(spec.split(':')[0]), int(spec.split(':')[1])
    RAND[x] = list(range(len(DESIGNS), len(DESIGNS) + m))
    DESIGNS += [(f'rand{x:g}%_{i + 1}', rng.uniform(-1, 1, D) * x / 10) for i in range(m)]
U_mc = {x: rng.uniform(-1, 1, (args.n_mc, D)) * x / 10 for x in RAND}
U = np.stack([u for _, u in DESIGNS])
PHI = model.denormalize_phi(torch.as_tensor(U, dtype=torch.float32, device=args.device)).cpu().numpy()

# =============================================================================
# The surrogate integrated over the muon flow (grid), the simulator on sampled muons; binned in p_z
# =============================================================================
pz_e = np.array([float(s) for s in args.pz_edges.split(',')])
NZ = len(pz_e) - 1


def pz_bin(raw):
    """p_z bin; muons outside the edges fall in the end bins."""
    return torch.bucketize(raw[:, 2].contiguous(), torch.as_tensor(pz_e[1:-1], dtype=raw.dtype, device=raw.device))


cache = feature_cache(model, shield.grid(*grid_spec(args.grid), args.n_muons), pz_bin)
pred_bin = cached_hits(model, cache, NZ, U)
sim_bin = np.zeros_like(pred_bin)
if args.sim:
    print(f'simulating {len(U)} designs x {args.n_muons:,} muons')
    sim_bin = simulate(model, shield, muon_blocks(shield, args.n_muons, args.block, args.seed), U, pz_bin, NZ)
pred, sim = pred_bin.sum(1), sim_bin.sum(1)
mc = {x: cached_hits(model, cache, NZ, u).sum(1) / pred[0] for x, u in U_mc.items()}

# =============================================================================
# Report
# =============================================================================
say = Report()
S0, P0 = sim[0], pred[0]


def sim_ratio(d):
    """H/H_0 and its Poisson error (counts treated as independent: conservative, the designs
    share their muons)."""
    rt = sim[d] / S0
    return rt, rt * np.sqrt(1 / max(sim[d], 1) + 1 / S0)


def changes(d):
    """[(label, unit, nominal, new, %)] of the parameters design d moves, largest first."""
    return [(labels[k], shield.units[k], PHI[0][k], PHI[d][k], 10 * U[d][k])
            for k in np.argsort(-np.abs(U[d])) if U[d][k] != 0]


say(f'{args.n_muons:,} muons from the flow at T = 1 (seed {args.seed}); directions from {args.run}')
if args.sim:
    say(f'nominal design: simulator {S0:.0f} +- {np.sqrt(S0):.0f} hits, surrogate {P0:.0f}')
say('\n' + '=' * 90 + '\nSUMMARY\n' + '=' * 90)
say(f'{"design":14s} {"moved":>5s} {"max":>6s}' + (f' {"sim hits":>9s} {"sim H/H_0":>16s} {"z":>6s}' if args.sim else '')
    + f' {"surrogate":>10s}')
is_rand = sum(RAND.values(), [])
for d, (name, u) in enumerate(DESIGNS):
    if d == 0 or d in is_rand:
        continue
    row = f'{name:14s} {int((u != 0).sum()):5d} {10 * np.abs(u).max():5.1f}%'
    if args.sim:
        rt, se = sim_ratio(d)
        row += f' {sim[d]:9.0f} {rt:8.3f} +- {se:5.3f} {(sim[d] - S0) / np.sqrt(sim[d] + S0):+6.1f}'
    say(row + f' {pred[d] / P0:10.3f}')
say('  z = (H - H_0) / sqrt(H + H_0)')

say('\n' + '=' * 90 + '\nWHAT EACH DESIGN CHANGES  (nominal -> new; all other parameters at nominal)\n' + '=' * 90)
for d, (name, u) in enumerate(DESIGNS):
    if d == 0 or d in is_rand:
        continue
    ch = changes(d)
    say(f'{name}: {len(ch)} parameters' + ('  ->  simulator H/H_0 = %.3f +- %.3f' % sim_ratio(d) if args.sim else '')
        + f', surrogate {pred[d] / P0:.3f}')
    for lab, unit, a, b, pct in ch[:15]:
        say(f'    {lab:20s} {a:10.4g} -> {b:10.4g} {unit:2s} ({b - a:+9.4g} {unit:2s}, {pct:+6.2f}%)')
    if len(ch) > 15:
        say(f'    ... and {len(ch) - 15} more, each by at most {abs(ch[15][4]):.2f}%')

say('\n' + '=' * 90 + '\nRANDOM TOLERANCES  (every parameter independently uniform in +-X%)\n' + '=' * 90)
say(f'surrogate: {args.n_mc} designs per X, integrated over the flow ({args.grid} grid); simulator: the designs listed')
for x, ds in RAND.items():
    q = np.quantile(mc[x], [0.05, 0.5, 0.95, 0.99])
    say(f'+-{x:g}%: surrogate H/H_0 median {q[1]:.3f}, 5-95% [{q[0]:.3f}, {q[2]:.3f}], 99% {q[3]:.3f}, '
        f'P(H > H_0) = {(mc[x] > 1).mean():.0%}')
    say('        surrogate on the simulated designs: ' + ', '.join(f'{pred[d] / P0:.2f}' for d in ds))
    if args.sim:
        sr = np.array([sim_ratio(d)[0] for d in ds])
        say('        simulator:                          ' + ', '.join(f'{s:.2f}' for s in sr)
            + f'  (median simulator/surrogate {np.median(sr / (pred[ds] / P0)):.3f})')

say('\nhits per p_z bin [GeV]' + (' (simulator)' if args.sim else ' (surrogate)'))
H_bin = sim_bin if args.sim else pred_bin
say(f'{"design":14s} ' + ' '.join(f'{f"{a:g}-{b:g}":>8s}' for a, b in zip(pz_e[:-1], pz_e[1:])))
for d, (name, _) in enumerate(DESIGNS):
    say(f'{name:14s} ' + ' '.join(f'{h:8.0f}' for h in H_bin[d]))

np.savez(os.path.join(args.run, 'worst_cases.npz'), names=np.array([nm for nm, _ in DESIGNS]), U=U, phis=PHI,
         labels=np.array(labels), units=np.array(shield.units), n_muons=args.n_muons, pz_edges=pz_e,
         sim_bin=sim_bin, pred_bin=pred_bin, simulated=args.sim,
         mc_pcts=np.array(list(mc)), mc_ratio=np.stack(list(mc.values())), grid=args.grid)
say.save(os.path.join(args.run, 'worst_cases_report.txt'))
print(f'saved {args.run}/worst_cases.npz and worst_cases_report.txt')
