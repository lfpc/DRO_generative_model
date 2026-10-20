"""Robustness split by muon kinematics: which muons drive the sensitivity, and do they agree.

The muons are split into bins of --bin_var (p_z, p_t, p or charge) and, per bin: hits, gradient
and Hessian of the bin's own hits H_b, exact one-at-a-time effects, the cosines between the bins'
worst directions, random designs in the box, and walks along a few directions (surrogate, and
the simulator unless --no_sim). Same frozen-trunk pass over the quadrature grids of the muon
flow as robustness.py, with one set of accumulators per bin (a bin is a cut on the grid, so even
the sparse high-momentum bins have no sampling noise). Writes <out_dir>/<bin_var>_bins.npz and
<bin_var>_bins_report.txt.

Directions (--dirs): ascent, descent, worst, corner, random, bin<k> (+grad of bin k alone,
k 1-based, negative from the end).

    python per_momenta.py --no_sim --bin_var pz --edges 10,20,30,45,75,125,175,225,300
    python per_momenta.py --no_sim --bin_var charge
"""
import argparse
import os

import numpy as np
import torch

from analysis import (Report, branch_jacobian, cached_hits, derivatives, feature_cache, grid_spec, muon_blocks,
                      nominal_pass, setup, simulate, trust_region_max, unit)


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument('--model', default='outputs/lcso.pt')
parser.add_argument('--bin_var', default='pz', choices=['pz', 'pt', 'p', 'charge'])
parser.add_argument('--edges', default='10,20,30,45,75,125,175,225,300',
                    help='interior bin edges in GeV (ignored for charge)')
parser.add_argument('--temperature', type=float, default=1.0, help='flow (momentum) temperature of the muons')
parser.add_argument('--grid', default='801x64', help='quadrature grid of the derivatives: latent points per side x azimuths')
parser.add_argument('--cache_grid', default='201x64', help='quadrature grid whose trunk features are cached')
parser.add_argument('--n_ref', type=float, default=1e9, help='hits are quoted per this many muons')
parser.add_argument('--n_muons_sim', type=int, default=int(1e8), help='sampled muons for the simulated walks')
parser.add_argument('--seed', type=int, default=0, help='seed of the sampled muons')
parser.add_argument('--dirs', default='ascent,worst,bin1,bin-1,random')
parser.add_argument('--t_max', type=float, default=1.0, help='walk half-length (L2, normalized)')
parser.add_argument('--n_points', type=int, default=5, help='points per walk (made odd)')
parser.add_argument('--no_sim', dest='sim', action='store_false', help='walks on the cached surrogate only')
parser.add_argument('--n_mc', type=int, default=2048, help='random designs in the box')
parser.add_argument('--top', type=int, default=12, help='rows in the printed tables')
parser.add_argument('--block', type=int, default=int(2e7), help='sampled muons held in memory (simulator)')
parser.add_argument('--out_dir', default='outputs/robustness')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
say = Report()
shield, model = setup(args.model, args.device)
D, labels = model.dim, shield.labels

# --- the bins
if args.bin_var == 'charge':
    EDGES, SHORT = torch.tensor([0.0]), ['mu+', 'mu-']        # pdg -13 is mu+, +13 is mu-
else:
    EDGES = torch.tensor([float(s) for s in args.edges.split(',')])
    ends = ['0'] + [f'{v:g}' for v in EDGES.tolist()] + ['inf']
    SHORT = [f'{a}-{b}' for a, b in zip(ends[:-1], ends[1:])]
B = len(SHORT)


def bin_of(raw):
    v = {'pz': raw[:, 2], 'pt': raw[:, :2].norm(dim=1), 'p': raw[:, :3].norm(dim=1),
         'charge': raw[:, 6]}[args.bin_var]
    return torch.bucketize(v.contiguous(), EDGES.to(v))


def grid(spec):
    return shield.grid(*grid_spec(spec), args.n_ref, args.temperature)


say(f'model {args.model}; the muon flow at T = {args.temperature:g} on grids of {args.grid} and {args.cache_grid} '
    f'latent x azimuth points; hits per {args.n_ref:.0e} muons; {B} bins in {args.bin_var}: ' + ' | '.join(SHORT))

# =============================================================================
# One pass: the derivatives of every bin's hits, and the feature cache
# =============================================================================
acc = nominal_pass(model, grid(args.grid), bin_of, B)
cache = feature_cache(model, grid(args.cache_grid), bin_of)
J = branch_jacobian(model)
grad_b, hess_b = (np.array(x) for x in zip(*[derivatives(model, J, acc.T[k], acc.A[k]) for k in range(B)]))
grad, hess = grad_b.sum(0), hess_b.sum(0)
H_b, H = acc.H, acc.H.sum()
Hn = np.maximum(H_b, 1e-30)                                # an empty bin must not divide by zero
H0c = cached_hits(model, cache, B, np.zeros((1, D)))[0]


def cached_ratio(U):
    """H_b(u)/H_b(0) per bin, on the cache: (m, B)."""
    return cached_hits(model, cache, B, U) / np.maximum(H0c, 1e-30)


gn = np.linalg.norm(grad_b, axis=1)
ghat_b = grad_b / np.maximum(gn, 1e-30)[:, None]
curv_b = np.einsum('bi,bij,bj->b', ghat_b, hess_b, ghat_b)
COS = ghat_b @ ghat_b.T

say(f'\n{"bin":>10s} {"muons":>14s} {"hits":>10s} {"share":>7s} {"rate":>9s} {"|g|/H":>7s} {"curv/H":>7s} {"cos(g)":>7s}')
for k in range(B):
    say(f'{SHORT[k]:>10s} {acc.N[k]:14,.0f} {H_b[k]:10.2f} {H_b[k] / H:7.1%} {H_b[k] / max(acc.N[k], 1):9.2e} '
        f'{gn[k] / Hn[k]:7.2f} {curv_b[k] / Hn[k]:7.2f} {ghat_b[k] @ unit(grad):+7.3f}')
say(f'{"total":>10s} {acc.N.sum():14,.0f} {H:10.2f} {1:7.1%} {H / acc.N.sum():9.2e} {np.linalg.norm(grad) / H:7.2f}')
say('  |g|/H: the fractional change of the bin\'s hits per normalized unit along its own worst direction;\n'
    '  cos(g): alignment of that direction with the total gradient')

# --- exact one-at-a-time, per bin
oat = cached_ratio(np.concatenate([np.eye(D), -np.eye(D)]))
oat_p, oat_m = oat[:D] - 1, oat[D:] - 1
rel = grad_b / Hn[:, None]                                 # (B, D) first order, per bin
order = np.argsort(-np.abs(rel).max(0))
say('\nexact one-at-a-time, the worse of u_i = +-1, as a fraction of each bin\'s hits:')
say(f'{"parameter":20s} ' + ' '.join(f'{s:>9s}' for s in SHORT))
for k in order[:args.top]:
    say(f'{labels[k]:20s} ' + ' '.join(f'{v:+9.3f}' for v in np.maximum(oat_p[k], oat_m[k])))

say('\ncosine between the bins\' own worst directions:')
for k in range(B):
    say(f'{SHORT[k]:>10s} ' + ' '.join(f'{c:+6.2f}' for c in COS[k]))
off = COS[~np.eye(B, dtype=bool)]
say(f'  off-diagonal: min {off.min():+.2f}, median {np.median(off):+.2f}, max {off.max():+.2f}; '
    'negative = the bins want opposite changes')

U_mc = (torch.rand(args.n_mc, D, generator=torch.Generator().manual_seed(args.seed + 2)) * 2 - 1).numpy()
mc = cached_ratio(U_mc)
say(f'\n{args.n_mc:,} random designs in the box, H_b/H_b(0): median ' + ', '.join(
    f'{s} {m:.2f}' for s, m in zip(SHORT, np.median(mc, 0))))


# =============================================================================
# Walks
# =============================================================================
def direction(name):
    if name in ('ascent', 'descent'):
        return unit(grad) * (1 if name == 'ascent' else -1)
    if name == 'worst':
        return unit(trust_region_max(grad, hess, args.t_max))
    if name == 'corner':
        return unit(np.sign(grad))
    if name == 'random':
        return unit(np.random.default_rng(args.seed + 3).standard_normal(D))
    if name.startswith('bin'):
        k = int(name[3:])
        return unit(grad_b[k - 1 if k > 0 else B + k])
    raise ValueError(f'unknown direction {name!r}')


NAMES = args.dirs.split(',')
V = np.stack([direction(nm) for nm in NAMES])
ts = np.linspace(-args.t_max, args.t_max, args.n_points | 1)
designs = np.concatenate([np.zeros((1, D))] + [np.clip(np.outer(ts[ts != 0], v), -1, 1) for v in V])
IDX = np.zeros((len(V), len(ts)), dtype=int)
IDX[:, ts != 0] = 1 + np.arange(len(V) * (len(ts) - 1)).reshape(len(V), -1)
say(f'\n{"direction":10s} {"cos(g)":>7s}  per-bin slope (dH_b/dt)/H_b')
for nm, v in zip(NAMES, V):
    say(f'{nm:10s} {unit(grad) @ v:+7.3f}  ' + ' '.join(f'{s:+6.2f}' for s in grad_b @ v / Hn))

pred_cache = cached_ratio(designs)
pred = cached_hits(model, cache, B, designs)
sim = np.zeros((len(designs), B))
if args.sim:
    blocks = muon_blocks(shield, args.n_muons_sim, args.block, args.seed, args.temperature)
    sim = simulate(model, shield, blocks, designs, bin_of, B)
    say(f'\nH_b at the far end of each walk relative to nominal, surrogate|simulator ({args.n_muons_sim:,} muons):')
    say(f'{"direction":10s} ' + ' '.join(f'{s:>13s}' for s in SHORT + ['total']))
    for nm, idx in zip(NAMES, IDX):
        i = idx[-1]
        cells = [f'{p / max(p0, 1e-30):.2f}|{s / max(s0, 1):.2f}'
                 for p, p0, s, s0 in zip(pred[i], pred[0], sim[i], sim[0])]
        cells.append(f'{pred[i].sum() / pred[0].sum():.2f}|{sim[i].sum() / max(sim[0].sum(), 1):.2f}')
        say(f'{nm:10s} ' + ' '.join(f'{c:>13s}' for c in cells))

stem = os.path.join(args.out_dir, f'{args.bin_var}_bins')
np.savez(stem + '.npz', labels=np.array(labels), bin_var=args.bin_var, bin_labels=np.array(SHORT),
         edges=EDGES.numpy(), temperature=args.temperature, counts=acc.N, H_b=H_b, H_0_cache=H0c,
         grad_b=grad_b, hess_b=hess_b, grad=grad, hess=hess, cos=COS, oat_plus=oat_p, oat_minus=oat_m, mc=mc,
         dir_names=np.array(NAMES), dir_vecs=V, t=ts, dir_idx=IDX, designs=designs,
         pred_cache=pred_cache, pred=pred, sim=sim, simulated=args.sim)
say.save(stem + '_report.txt')
print(f'saved {stem}.npz and {stem}_report.txt')
