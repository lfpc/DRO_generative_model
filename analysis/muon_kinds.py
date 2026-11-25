"""Which muons the worst designs work on: the worst direction of each p_z bin, and of each kind
of muon (p_z, p_t, direction of p_t, charge).

The muons are split into cells: p_z bin x p_t bin x psi bin x charge, with psi the azimuth of
the transverse momentum measured from +x and folded to [0, 180] deg (the shield is symmetric
under y -> -y). The first magnets push mu+ towards -x and mu- towards +x, so psi_q, the
azimuth of (q p_x, |p_y|), is 0 for a muon that starts against that push and 180 along it. The
magnets are symmetric under x -> -x with the charge flipped but the cavern is not, so a mu+ at
psi and a mu- at 180 - psi follow the same pattern with about twice the hits for the mu+: the
two charges are kept apart. One pass of the surrogate over the --grid muon points gives every cell's
hits, gradient and Hessian at nominal (analysis.core: the accumulators of a cell are sums over
its points, so a p_z bin's are the sums over its cells). Then:

  * per p_z bin: its own worst design (trust_region_max of its quadratic model at --t_max, as
    robustness.py's 'worst' for all muons), and the exact surrogate hits of every bin at every
    bin's worst design and at robustness.py's 'worst' (on the cached --cache_grid points);
  * per cell: its own worst design, the exact rise of its hits there and under 'worst', and the
    cosine of its worst direction with 'worst'; ranked by the hits it can gain (absolute, per
    --n_ref muons) and by the factor.

Optionally (--n_muons_sim) the simulator checks the per-cell hits at nominal, at 'worst' and at
the worst designs of the --sim_bins p_z bins. Writes <out_dir>/muon_kinds.npz and
muon_kinds_report.txt; plots.py draws muon_kinds_*.png.

    python -m analysis.muon_kinds --device cpu
    python -m analysis.muon_kinds --n_muons_sim 1000000000
"""
import os
import time

import numpy as np
import torch

from analysis.core import (Report, branch_jacobian, cached_hits, derivatives, feature_cache, muon_blocks, setup, simulate,
                           trust_region_max, unit)
import cli
from paths import RUN

parser = cli.add_common(cli.parser(__doc__), fixed=True)
parser.set_defaults(fixed='M0:')                          # as the main run (robustness.py --fixed M0:)
parser.add_argument('--pz_edges', default='10,20,30,45,75,125,200,300', help='interior p_z bin edges (GeV)')
parser.add_argument('--pt_edges', default='0.5,1,2,3', help='interior p_t bin edges (GeV)')
parser.add_argument('--n_psi', type=int, default=8, help='bins of psi in [0, 180] deg (a divisor of half the grids\' azimuths, or the bins get unequal shares of them)')
parser.add_argument('--grid', default='401x64', help='muon points of the derivatives (muons.py)')
parser.add_argument('--cache_grid', default='401x64', help='muon points of the exact surrogate evaluations')
parser.add_argument('--n_ref', type=float, default=1e9, help='hits are quoted per this many muons')
parser.add_argument('--t_max', type=float, default=1.0, help='the tolerance ball (L2 norm of u), as robustness.py')
parser.add_argument('--run', default=RUN, help="robustness.py's --out_dir (its 'worst' direction)")
parser.add_argument('--min_hits', type=float, default=1.0, help='cells with fewer nominal hits (per n_ref) are left out of the factor ranking')
parser.add_argument('--top', type=int, default=15, help='rows of the rankings')
parser.add_argument('--n_muons_sim', type=int, default=0, help='simulated muons for the check (0: none)')
parser.add_argument('--sim_bins', default='30-45,300-inf', help='p_z bins whose worst designs are simulated')
parser.add_argument('--block', type=int, default=int(2e7), help='sampled muons held in memory (simulator)')
parser.add_argument('--seed', type=int, default=0, help='seed of the sampled muons')
parser.add_argument('--chunk', type=int, default=2 ** 20, help='muon points per accumulation step')
parser.add_argument('--out_dir', default=RUN)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
say = Report()
t_start = time.time()
shield, model = setup(args)
D, labels = model.dim, shield.labels
dev = model.bias.device

# --- the cells
PZ = torch.tensor([float(v) for v in args.pz_edges.split(',')])
PT = torch.tensor([float(v) for v in args.pt_edges.split(',')])
NZ, NT, NP = len(PZ) + 1, len(PT) + 1, args.n_psi
G = NZ * NT * NP * 2


def edges_names(e):
    ends = ['0'] + [f'{v:g}' for v in e.tolist()] + ['inf']
    return [f'{a}-{b}' for a, b in zip(ends[:-1], ends[1:])]


PZ_NAMES, PT_NAMES = edges_names(PZ), edges_names(PT)
PSI_EDGES = np.linspace(0, 180, NP + 1)
PSI_NAMES = [f'{a:.0f}-{b:.0f}' for a, b in zip(PSI_EDGES[:-1], PSI_EDGES[1:])]
CHARGES = ['mu+', 'mu-']


def cell_of(x):
    """Cell id ((iz * NT + it) * NP + ip) * 2 + iq of muons x (n, 7; pdg -13 is mu+)."""
    pz, pt = x[:, 2], x[:, :2].norm(dim=1)
    iz = torch.bucketize(pz.contiguous(), PZ.to(x))
    it = torch.bucketize(pt.contiguous(), PT.to(x))
    psi = torch.rad2deg(torch.atan2(x[:, 1].abs(), x[:, 0]))           # [0, 180]
    ip = torch.clamp((psi / (180 / NP)).long(), 0, NP - 1)
    iq = (x[:, 6] > 0).long()                                           # 0: mu+, 1: mu-
    return ((iz * NT + it) * NP + ip) * 2 + iq


IDX = np.array(np.unravel_index(np.arange(G), (NZ, NT, NP, 2))).T      # (G, 4): iz, it, ip, iq


def cell_name(c):
    iz, it, ip, iq = IDX[c]
    return f'p_z {PZ_NAMES[iz]:>7s}  p_t {PT_NAMES[it]:>5s}  psi {PSI_NAMES[ip]:>7s}  {CHARGES[iq]}'


# =============================================================================
# One pass: hits, gradient and Hessian of every cell
# =============================================================================
@torch.no_grad()
def accumulate(points):
    """Per cell: T = sum w s t, A = sum w s (1 - 2p) t t^T, H = sum w p, N = sum w (analysis.core),
    with the points of a chunk sorted by cell so each cell is one slice."""
    b0 = model.branch_net(torch.zeros(1, D, device=dev))[0]
    P = b0.numel()
    T = torch.zeros(G, P, dtype=torch.float64, device=dev)
    A = torch.zeros(G, P, P, dtype=torch.float64, device=dev)
    H = torch.zeros(G, dtype=torch.float64, device=dev)
    N = torch.zeros(G, dtype=torch.float64, device=dev)
    for x, w in points:
        for c0 in range(0, len(x), args.chunk):
            xc, wc = x[c0:c0 + args.chunk].to(dev), w[c0:c0 + args.chunk].to(dev)
            g = cell_of(xc)
            o = torch.argsort(g)
            g, xc, wc = g[o], xc[o], wc[o]
            t = model.features(xc)
            p = torch.sigmoid(t @ b0 + model.bias)
            s = p * (1 - p)
            H.index_add_(0, g, p.double() * wc)
            N.index_add_(0, g, wc)
            T.index_add_(0, g, (s.double() * wc)[:, None] * t.double())
            cells, counts = torch.unique_consecutive(g, return_counts=True)
            lo = 0
            for c, n in zip(cells.tolist(), counts.tolist()):
                tk = t[lo:lo + n]
                a = (wc[lo:lo + n].float() * s[lo:lo + n] * (1 - 2 * p[lo:lo + n]))
                A[c] += ((tk * a[:, None]).T @ tk).double()
                lo += n
    return T, A, H.cpu().numpy(), N.cpu().numpy()


say(f'model {args.model}; muons {args.muons} on the points {args.grid} (derivatives) and {args.cache_grid} (exact); '
    f'hits per {args.n_ref:.0e} muons; tolerance ball |u| <= {args.t_max:g}')
say(f'{G} cells: p_z {" | ".join(PZ_NAMES)}; p_t {" | ".join(PT_NAMES)}; psi (deg from +x, |p_y|) {" | ".join(PSI_NAMES)}; mu+, mu-')
T, A, Hc, Nc = accumulate(shield.muons.points(args.grid, args.n_ref))
print(f'  accumulated ({time.time() - t_start:.0f}s)', flush=True)
J = branch_jacobian(model)


def grad_hess(Tk, Ak):
    return derivatives(model, J, Tk, Ak)


live = np.where(Nc > 0)[0]
grad_c, hess_c = np.zeros((G, D)), np.zeros((G, D, D))
for c in live:
    grad_c[c], hess_c[c] = grad_hess(T[c], A[c])
print(f'  cell derivatives ({time.time() - t_start:.0f}s)', flush=True)
grad, hess = grad_c.sum(0), hess_c.sum(0)
H0 = Hc.sum()

# the p_z bins: sums of their cells
ZB = IDX[:, 0]
grad_z = np.array([grad_c[ZB == k].sum(0) for k in range(NZ)])
hess_z = np.array([hess_c[ZB == k].sum(0) for k in range(NZ)])
Hz = np.array([Hc[ZB == k].sum() for k in range(NZ)])
Nz = np.array([Nc[ZB == k].sum() for k in range(NZ)])

r = np.load(os.path.join(args.run, 'robustness.npz'))
assert list(r['labels']) == list(labels), 'robustness.npz is of another parameter set'
v_worst = args.t_max * r['dir_vecs'][[str(n) for n in r['dir_names']].index('worst')]
say(f'H_0 = {H0:.1f} (robustness.py: {float(r["H_0"]):.1f}); "worst" from {args.run}/robustness.npz')

# =============================================================================
# The exact surrogate on the cache: every cell at every design
# =============================================================================
cache = feature_cache(model, shield.muons.points(args.cache_grid, args.n_ref), cell_of)
print(f'  feature cache: {len(cache.w):,} points ({time.time() - t_start:.0f}s)', flush=True)


def exact(U):
    """Hits of every cell at designs U (m, D): (m, G)."""
    return cached_hits(model, cache, G, np.atleast_2d(U))


E0 = exact(np.zeros(D))[0]                       # nominal, on the cache points
E_worst = exact(v_worst)[0]

# --- per p_z bin
V_z = np.array([trust_region_max(grad_z[k], hess_z[k], args.t_max) if Hz[k] > 0 else np.zeros(D) for k in range(NZ)])
E_z = exact(V_z)                                 # (NZ designs, G cells)


def by_bin(E):
    return np.array([E[..., ZB == k].sum(-1) for k in range(NZ)]).T


Z0, Zw, ZZ = by_bin(E0), by_bin(E_worst), by_bin(E_z)            # (NZ,), (NZ,), (NZ designs, NZ bins)
cos_z = np.array([[unit(a) @ unit(b) if np.any(a) and np.any(b) else np.nan for b in V_z] for a in V_z])
cos_zw = np.array([unit(v) @ unit(v_worst) if np.any(v) else np.nan for v in V_z])

say(f'\n=== per p_z bin: its own worst design (|u| = {args.t_max:g}) vs robustness.py\'s "worst"')
say(f'{"p_z bin":>9s} {"muons":>13s} {"Y_0":>8s} {"share":>6s}   {"own worst":>9s} {"worst":>7s}   {"cos":>5s}  {"all muons at own":>16s}')
for k in range(NZ):
    if Z0[k] <= 0:
        continue
    say(f'{PZ_NAMES[k]:>9s} {Nz[k]:13,.0f} {Z0[k]:8.2f} {Z0[k] / Z0.sum():6.1%}   x{ZZ[k, k] / Z0[k]:8.2f} x{Zw[k] / Z0[k]:6.2f}   '
        f'{cos_zw[k]:+.2f}  x{ZZ[k].sum() / Z0.sum():6.2f}')
say(f'{"all":>9s} {Nc.sum():13,.0f} {Z0.sum():8.2f} {"":6s}   {"":9s} x{Zw.sum() / Z0.sum():6.2f}')
say('  own worst: the bin\'s hits at its own worst design / nominal; worst: at robustness.py\'s "worst"; cos: of the two\n'
    '  directions; all muons at own: the total hits at the bin\'s worst design / nominal')
say('\nevery bin\'s hits (rows) at every bin\'s worst design (columns), / nominal:')
say(f'{"":>9s} ' + ' '.join(f'{n:>8s}' for n in PZ_NAMES) + f' {"worst":>8s}')
for b in range(NZ):
    if Z0[b] > 0:
        say(f'{PZ_NAMES[b]:>9s} ' + ' '.join(f'{ZZ[k, b] / Z0[b]:8.2f}' for k in range(NZ)) + f' {Zw[b] / Z0[b]:8.2f}')
say('\nlargest components of each bin\'s worst direction (unit vector):')
for k in range(NZ):
    if Z0[k] <= 0:
        continue
    v = unit(V_z[k])
    o = np.argsort(-np.abs(v))[:6]
    say(f'{PZ_NAMES[k]:>9s}: ' + ', '.join(f'{labels[i]} {v[i]:+.2f}' for i in o))

# --- per cell
V_c = np.zeros((G, D))
for c in live:
    if Hc[c] > 0 and np.any(grad_c[c]):
        with np.errstate(divide='ignore', invalid='ignore'):
            v = trust_region_max(grad_c[c], hess_c[c], args.t_max)
        V_c[c] = v if np.all(np.isfinite(v)) else args.t_max * unit(grad_c[c])   # degenerate: the gradient
own = np.zeros(G)
for i in range(0, len(live), 64):                # the exact hits of each cell at its own worst design
    cs = live[i:i + 64]
    own[cs] = exact(V_c[cs])[np.arange(len(cs)), cs]
cos_cw = np.array([unit(V_c[c]) @ unit(v_worst) if np.any(V_c[c]) else np.nan for c in range(G)])
gain_own, gain_worst = own - E0, E_worst - E0

say(f'\n=== kinds of muon: {len(live)} cells with muons')
say('psi: azimuth of p_t from +x (folded |p_y|). The magnets are symmetric under x -> -x with the charge flipped,\n'
    'the cavern is not: mu+ at psi and mu- at 180 - psi should follow one pattern, the mu+ with more hits.')
sym = np.array([[E0[np.ravel_multi_index((iz, it, ip, 0), (NZ, NT, NP, 2))],
                 E0[np.ravel_multi_index((iz, it, NP - 1 - ip, 1), (NZ, NT, NP, 2))]]
                for iz in range(NZ) for it in range(NT) for ip in range(NP)])
ok = sym.sum(1) > args.min_hits
say(f'mirror check, nominal hits of mu+ at psi vs mu- at 180 - psi over {ok.sum()} cell pairs with > {args.min_hits:g} hits: '
    f'median ratio {np.median(sym[ok, 0] / np.maximum(sym[ok, 1], 1e-30)):.2f}, '
    f'correlation of the logs {np.corrcoef(np.log(sym[ok, 0] + 1e-3), np.log(sym[ok, 1] + 1e-3))[0, 1]:.2f}')


def table(order, title):
    say(f'\n{title}')
    say(f'{"cell":52s} {"muons":>12s} {"P(hit)":>8s} {"Y_0":>8s} {"own worst":>10s} {"(+hits)":>9s} {"worst":>7s} {"cos":>5s}')
    for c in order[:args.top]:
        say(f'{cell_name(c):52s} {Nc[c]:12,.0f} {E0[c] / max(Nc[c], 1):8.1e} {E0[c]:8.2f} x{own[c] / max(E0[c], 1e-30):9.2f} '
            f'{gain_own[c]:+9.1f} x{E_worst[c] / max(E0[c], 1e-30):6.2f} {cos_cw[c]:+.2f}')


table(np.argsort(-E0), 'the cells with the most hits at nominal:')
table(np.argsort(-gain_own), 'the cells that can gain the most hits (at their own worst design, absolute):')
fac = np.where(E0 > args.min_hits, own / np.maximum(E0, 1e-30), 0)
table(np.argsort(-fac), f'the cells whose hits can grow the most (factor; Y_0 > {args.min_hits:g}):')
table(np.argsort(-gain_worst), 'the cells that gain the most under robustness.py\'s "worst":')

# marginals over the variables: share of the nominal hits and of the gain under "worst"
say('\nshares of the nominal hits and of the extra hits at "worst", per variable (psi_q: the azimuth of (q p_x, |p_y|),\n'
    '0 = towards +x for mu+ and -x for mu-, i.e. against the first magnets, which push mu+ to -x;\n'
    'mu+ at psi and mu- at 180 - psi together):')
for name, sel in (('p_z', None),):
    h0 = np.array([E0[ZB == k].sum() for k in range(NZ)])
    gw = np.array([gain_worst[ZB == k].sum() for k in range(NZ)])
    say(f'  {name:>5s}: ' + '  '.join(f'{n}: {a / E0.sum():5.1%} / {b / gain_worst.sum():5.1%}' for n, a, b in zip(PZ_NAMES, h0, gw)))
PSIQ = np.where(IDX[:, 3] == 0, IDX[:, 2], NP - 1 - IDX[:, 2])
h0 = np.array([E0[PSIQ == j].sum() for j in range(NP)])
gw = np.array([gain_worst[PSIQ == j].sum() for j in range(NP)])
say(f'  psi_q: ' + '  '.join(f'{n}: {a / E0.sum():5.1%} / {b / gain_worst.sum():5.1%}' for n, a, b in zip(PSI_NAMES, h0, gw)))
for name, axis, names in (('p_t', 1, PT_NAMES), ('psi', 2, PSI_NAMES)):
    for q in range(2):
        m = IDX[:, 3] == q
        h0 = np.array([E0[m & (IDX[:, axis] == j)].sum() for j in range(len(names))])
        gw = np.array([gain_worst[m & (IDX[:, axis] == j)].sum() for j in range(len(names))])
        say(f'  {name:>4s} {CHARGES[q]}: ' + '  '.join(f'{n}: {a / E0.sum():5.1%} / {b / gain_worst.sum():5.1%}'
                                                    for n, a, b in zip(names, h0, gw)))

# =============================================================================
# The simulator's check
# =============================================================================
sim, sim_names = np.zeros((0, G)), []
if args.n_muons_sim:
    picks = [n for n in args.sim_bins.split(',') if n]
    U = [np.zeros(D), v_worst] + [V_z[PZ_NAMES.index(n)] for n in picks]
    sim_names = ['nominal', 'worst'] + [f'own worst of p_z {n}' for n in picks]
    say(f'\n=== the simulator, {args.n_muons_sim:.0e} muons: hits per p_z bin at ' + ', '.join(sim_names))
    sim = simulate(model, shield, muon_blocks(shield, args.n_muons_sim, args.block, args.seed), np.array(U), cell_of, G)
    sim *= args.n_ref / args.n_muons_sim
    pred = exact(np.array(U))
    for j, n in enumerate(sim_names):
        say(f'{n:28s} sim ' + ' '.join(f'{v:7.1f}' for v in by_bin(sim[j])) + f'   total {sim[j].sum():8.1f}')
        say(f'{"":28s} sur ' + ' '.join(f'{v:7.1f}' for v in by_bin(pred[j])) + f'   total {pred[j].sum():8.1f}')

stem = os.path.join(args.out_dir, 'muon_kinds')
np.savez(stem + '.npz', labels=np.array(labels), pz_names=np.array(PZ_NAMES), pt_names=np.array(PT_NAMES),
         psi_names=np.array(PSI_NAMES), psi_edges=PSI_EDGES, charges=np.array(CHARGES), shape=np.array([NZ, NT, NP, 2]),
         n_ref=args.n_ref, t_max=args.t_max, N=Nc, H_grid=Hc, E0=E0, E_worst=E_worst, own=own, V_c=V_c, cos_cw=cos_cw,
         grad_c=grad_c, V_z=V_z, E_z=E_z, cos_z=cos_z, cos_zw=cos_zw, v_worst=v_worst,
         sim=sim, sim_names=np.array(sim_names), n_muons_sim=args.n_muons_sim)
say.save(stem + '_report.txt')
print(f'saved {stem}.npz and {stem}_report.txt ({time.time() - t_start:.0f}s)')
