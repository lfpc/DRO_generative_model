"""Physical relations behind the surrogate's sensitivities: which few physical quantities of the
design (B*L of the magnets, yoke fields, apertures, the length) explain how each class of muons
responds to the 58 parameters.

Per muon bin (bins.py's npz; 'all' = robustness.npz), in log Y, where a product of parameters is
a sum: with g = dlogY/du and K = d2logY/du2 = hess/Y - g g^T, and a library of quantities phi_j(u),

    logY = f(log phi_1, ..., log phi_n)   =>   g = sum_j a_j grad(log phi_j)
                                               K = sum_j a_j hess(log phi_j) + sum_jl C_jl grad_j grad_l^T

a_j = dlogY/dlogphi_j is the bin's elasticity to phi_j (Y ~ phi_j^a_j). The quantities are picked
greedily on the gradient, among the physical ones and the bare parameters (atoms), so a physical
quantity is chosen only if it explains more than the parameters it is made of. The Hessian then
tests the relation: the term sum_j a_j hess(log phi_j) has no free parameter (the a_j come from the
gradient), e.g. the cross term of B and L in log(sum_m B_m L_m). Reported: the gradient's explained
fraction, and the Hessian's with that term and without it (C refitted either way).

--sim simulates designs that test one relation: changes of B and L of M1-M4 that keep each
magnet's B*L fixed (iso) against the same changes of B or of L alone, on muons with
p_z > --band (shield.sample's pz_band), per bin, against the surrogate members' predictions.

    python -m analysis.relations                       # the fits, from <run>/pz_bins.npz
    python -m analysis.relations --bins charge
    python -m analysis.relations --sim                 # + the iso-B*L designs, simulated
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import cli
from analysis.core import Report
from paths import RUN, figures, save_figure
from shield import COLUMNS, FREE, NOMINAL, MuonShield

parser = cli.add_common(cli.parser(__doc__), fixed=True)
parser.set_defaults(fixed='M0:')
parser.add_argument('--run', default=RUN, help='robustness.py / bins.py --out_dir')
parser.add_argument('--bins', default='pz', help='<run>/<bins>_bins.npz of bins.py')
parser.add_argument('--max_terms', type=int, default=6, help='quantities per bin at most')
parser.add_argument('--stop', type=float, default=0.97, help='stop adding quantities once this fraction of |g|^2 is explained')
parser.add_argument('--sim', action='store_true', help='simulate the iso-B*L designs')
parser.add_argument('--step', type=float, default=0.05, help='relative change of B (or L) in the simulated designs')
parser.add_argument('--band', type=float, default=125, help='simulate muons with p_z above this (GeV)')
parser.add_argument('--n_band', type=int, default=int(2e8), help='simulated muons in the band, per design')
parser.add_argument('--block', type=int, default=int(1e7), help='sampled muons held in memory')
parser.add_argument('--cache_grid', default='201x64', help='muon points of the surrogate predictions')
args = parser.parse_args()
say = Report()
DELTA = MuonShield.DELTA

# =============================================================================
# The library: log phi_j(u) of the full design, with its gradient and Hessian at u = 0
# =============================================================================
b = dict(np.load(os.path.join(args.run, f'{args.bins}_bins.npz')))
LABELS = [str(l) for l in b['labels']]
ALL = [f'M{m}:{COLUMNS[c]}' for m, c in FREE]
keep = torch.tensor([ALL.index(l) for l in LABELS])
IDX = torch.tensor(FREE)
phi0 = NOMINAL[IDX[:, 0], IDX[:, 1]].double()


def full(u):
    """The (8, 15) design of normalized u (the analysed parameters; the rest at nominal)."""
    phi = phi0.clone().index_put((keep,), phi0[keep] * (1 + DELTA * u))
    f = NOMINAL.double().index_put((IDX[:, 0], IDX[:, 1]), phi)
    tie = f.clone()                                     # shield.full_design's tied columns
    tie[:, 13], tie[:, 5], tie[:, 11] = f[:, 12], f[:, 4], f[:, 10]
    tie[:5, 3], tie[:5, 7], tie[:5, 9] = f[:5, 2], f[:5, 6], f[:5, 8]
    return tie


def col(F, m, name):
    return F[m, COLUMNS.index(name)]


FIELD = [1, 2, 3, 4, 6, 7]                               # magnets with a field (M5 has none)


def quantities(F):
    """name -> positive scalar of the design F."""
    BL = {m: col(F, m, 'NI').abs() * 2 * col(F, m, 'Z_len') for m in FIELD}
    q = {f'B*L M{m}': BL[m] for m in FIELD}
    q['B*L M1-4'] = sum(BL[m] for m in (1, 2, 3, 4))
    q['B*L M6-7'] = BL[6] + BL[7]
    q['B*L all'] = sum(BL.values())
    q['length'] = (F[:, 0] + 2 * F[:, 1]).sum()
    q['L M1-4'] = sum(2 * col(F, m, 'Z_len') for m in (1, 2, 3, 4))
    for m in FIELD:                                      # flux B dX through the yoke dXYoke
        for s in ('In', 'Out') if m in (6, 7) else ('In',):
            q[f'yoke B M{m}{s}'] = col(F, m, 'NI').abs() * col(F, m, f'dX{s}') / col(F, m, f'dXYoke{s}')
    for m in range(1, 8):
        q[f'aperture M{m}'] = col(F, m, 'dXIn') * col(F, m, 'dYIn')
        q[f'iron M{m}'] = (col(F, m, 'dXIn') + col(F, m, 'dXYokeIn')) * col(F, m, 'dY_yokeIn')
    for l in LABELS:                                     # the atoms: each parameter alone
        m, c = l.split(':')
        q[l] = col(F, int(m[1:]), c).abs()
    return q


names = list(quantities(full(torch.zeros(len(LABELS), dtype=torch.float64))))
u0 = torch.zeros(len(LABELS), dtype=torch.float64)
D = len(LABELS)
GQ, HQ = [], []
for n in names:
    fn = lambda u: torch.log(quantities(full(u))[n])
    GQ.append(torch.autograd.functional.jacobian(fn, u0).numpy())
    HQ.append(torch.autograd.functional.hessian(fn, u0).numpy())
GQ, HQ = np.array(GQ), np.array(HQ)
live = np.linalg.norm(GQ, axis=1) > 1e-12                # quantities the analysed parameters move
names, GQ, HQ = [n for n, k in zip(names, live) if k], GQ[live], HQ[live]
ATOM = np.array([n in LABELS for n in names])
say(f'{len(names)} quantities: {(~ATOM).sum()} physical + {ATOM.sum()} parameters; bins of {args.bins} from {args.run}')


# =============================================================================
# Fits per bin
# =============================================================================
def select(g):
    """Greedy forward selection on the gradient: indices, coefficients, explained fraction per step."""
    S, frac, gg = [], [], g @ g
    for _ in range(args.max_terms):
        best = None
        for j in range(len(names)):
            if j in S:
                continue
            A = GQ[S + [j]].T
            a = np.linalg.lstsq(A, g, rcond=None)[0]
            e = 1 - np.sum((g - A @ a) ** 2) / gg
            if best is None or e > best[1]:
                best = (j, e)
        S.append(best[0])
        frac.append(best[1])
        if best[1] >= args.stop:
            break
    a = np.linalg.lstsq(GQ[S].T, g, rcond=None)[0]
    return S, a, frac


def hessian_fit(K, S, a):
    """Explained fraction of K with the parameter-free term sum a_j hess(log phi_j) and without it,
    C refitted on sym(grad_j grad_l^T) either way; on the parameters S's quantities touch."""
    on = np.abs(GQ[S]).sum(0) > 1e-12
    Gs = GQ[S][:, on]
    basis = [np.outer(Gs[i], Gs[j]) + np.outer(Gs[j], Gs[i]) for i in range(len(S)) for j in range(i, len(S))]
    B = np.array([x.ravel() for x in basis]).T
    Ks = K[np.ix_(on, on)]
    curv = np.einsum('j,jkl->kl', a, HQ[S][:, on][:, :, on])
    out = []
    for target in (Ks - curv, Ks):
        c = np.linalg.lstsq(B, target.ravel(), rcond=None)[0]
        out.append(1 - np.sum((target.ravel() - B @ c) ** 2) / np.sum(Ks ** 2))
    return out, on.sum()


r = np.load(os.path.join(args.run, 'robustness.npz'))
bins = [str(s) for s in b['bin_labels']] + ['all']
Yb = list(b['H_b']) + [float(r['H_0'])]
Gb = list(b['grad_b']) + [r['grad']]
Hb = list(b['hess_b']) + [r['hess']]
say('\nper bin: the quantities picked (elasticity a = dlogY/dlog phi, so Y ~ phi^a), the explained fraction\n'
    'of |dlogY/du|^2 after each, and of the log-Hessian on the parameters they touch, with the\n'
    'parameter-free curvature term of the quantities (the a_j from the gradient) and without it')
fits = {}
for k, name in enumerate(bins):
    if Yb[k] < 1:
        say(f'\n{name}: {Yb[k]:.2f} hits, skipped')
        continue
    g = Gb[k] / Yb[k]
    K = Hb[k] / Yb[k] - np.outer(g, g)
    S, a, frac = select(g)
    (with_c, without_c), n_on = hessian_fit(K, S, a)
    fits[name] = dict(S=S, a=a, frac=frac, hess=(with_c, without_c))
    say(f'\n{name} GeV: Y = {Yb[k]:.1f}, |dlogY/du| = {np.linalg.norm(g):.2f}')
    for j, aj, f in zip(S, a, frac):
        say(f'  {names[j]:18s} {"(parameter)" if ATOM[j] else "":12s} a = {aj:+8.2f}   |g|^2 explained {f:6.1%}')
    say(f'  log-Hessian on {n_on} parameters explained: {with_c:6.1%} with the curvature term, {without_c:6.1%} without')

# =============================================================================
# Figure: the elasticities of the physical quantities picked, per bin
# =============================================================================
phys = sorted({names[j] for f in fits.values() for j in f['S'] if not ATOM[j]},
              key=lambda n: names.index(n))
cols = list(fits)
if phys:
    E = np.full((len(phys), len(cols)), np.nan)
    for c, n in enumerate(cols):
        f = fits[n]
        for j, aj in zip(f['S'], f['a']):
            if names[j] in phys:
                E[phys.index(names[j]), c] = aj
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.5, 2.2 + 0.32 * len(phys)), height_ratios=[len(phys), 3], sharex=True)
    v = np.nanmax(np.abs(E))
    im = ax.imshow(E, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    for (i, c), e in np.ndenumerate(E):
        if np.isfinite(e):
            ax.text(c, i, f'{e:+.1f}', ha='center', va='center', fontsize=7)
    ax.set_yticks(range(len(phys)), phys, fontsize=8)
    fig.colorbar(im, ax=ax, label=r'$\partial\log Y/\partial\log\phi$', fraction=0.03)
    x = np.arange(len(cols))
    bx.bar(x - 0.2, [fits[n]['frac'][-1] for n in cols], 0.4, color='0.4', label=r'$|\nabla\log Y|^2$')
    bx.bar(x + 0.2, [max(fits[n]['hess'][0], 0) for n in cols], 0.4, color='#0072B2', label='log-Hessian')
    bx.set_ylim(0, 1)
    bx.set_ylabel('explained', fontsize=8)
    bx.set_xticks(x, cols, rotation=45, fontsize=8)
    bx.set_xlabel(f'${args.bins[0]}_{args.bins[1]}$ bin [GeV]' if args.bins != 'charge' else 'charge')
    bx.legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save_figure(fig, figures(args.run), f'relations_{args.bins}', f'relations_{args.bins}')

# =============================================================================
# The iso-B*L designs, simulated
# =============================================================================
if args.sim:
    from analysis.core import cached_hits, feature_cache, members, setup, simulate
    shield, model = setup(args)
    assert shield.labels == LABELS
    edges = torch.tensor(b['edges'], dtype=torch.float32)
    bin_of = lambda raw: torch.bucketize(raw[:, 2].contiguous(), edges.to(raw.device))
    s = args.step
    dl = (1 / (1 + s) - 1) / DELTA                       # L / (1 + s): B*L unchanged with B * (1 + s)
    designs = {'nominal': {}, f'iso B*L (B +{s:.0%}, L /{1 + s:g})': {'NI': s / DELTA, 'Z_len': dl},
               f'iso B*L (B -{s:.0%}, L /{1 - s:g})': {'NI': -s / DELTA, 'Z_len': (1 / (1 - s) - 1) / DELTA},
               f'B +{s:.0%}': {'NI': s / DELTA}, f'L /{1 + s:g}': {'Z_len': dl}}
    U = np.zeros((len(designs), D))
    for i, ch in enumerate(designs.values()):
        for c, v in ch.items():
            for m in (1, 2, 3, 4):
                U[i, LABELS.index(f'M{m}:{c}')] = v
    nb = len(edges) + 1
    pts = lambda: shield.muons.points(args.cache_grid, 1e9)
    pred = np.array([cached_hits(m, feature_cache(m, pts(), bin_of), nb, U) for m in members(args, shield)])
    pred = pred / pred[:, :1]                            # (K, designs, bins) relative to nominal

    def blocks():
        for lo in range(0, args.n_band, args.block):
            torch.manual_seed(lo)
            yield lo, shield.sample(min(args.block, args.n_band - lo), pz_band=(args.band, np.inf))
    sim = simulate(model, shield, blocks(), U, bin_of, nb)
    hi = [k for k in range(1, nb) if edges[k - 1] >= args.band]          # bins above the band's edge
    say(f'\nsimulated on {args.n_band:,} muons with p_z > {args.band:g} GeV; Y/Y_0 per bin, surrogate (members\' sd) vs simulator (Poisson)')
    say(f'{"design":32s}' + ''.join(f'{bins[k]:>24s}' for k in hi))
    for i, n in enumerate(designs):
        rat = sim[i, hi] / sim[0, hi]
        err = rat * np.sqrt(1 / np.maximum(sim[i, hi], 1) + (1 / np.maximum(sim[0, hi], 1) if i else 0))
        say(f'{n:32s}' + ''.join(f'{pred[:, i, k].mean():8.3f}+-{pred[:, i, k].std(ddof=1):.3f} {rv:6.3f}+-{e:.3f}'
                                 for k, rv, e in zip(hi, rat, err)))
    np.savez(os.path.join(args.run, 'relations_sim.npz'), designs=list(designs), U=U, sim=sim, pred=pred,
             edges=b['edges'], band=args.band, n_band=args.n_band)

# =============================================================================
# Figure: the B*L relation of M1-M4 for the high-momentum muons
# =============================================================================
HIGH = [n for n in bins[:-1] if n in fits and float(n.split('-')[0]) >= args.band]
if args.bins == 'pz' and HIGH:
    sim_file = os.path.join(args.run, 'relations_sim.npz')
    has_sim = os.path.exists(sim_file)
    fig, axs = plt.subplots(1, 3 if has_sim else 2, figsize=(13 if has_sim else 9, 3.8),
                            width_ratios=[1.3, 1, 1.4][:3 if has_sim else 2])
    BL = np.array([float(NOMINAL[m, 14].abs() * 2 * NOMINAL[m, 1]) for m in (1, 2, 3, 4)])
    w = BL / BL.sum()                                    # each magnet's share of sum B*L
    # (a) gradients of the top bin: B and L of each magnet, against a * share * DELTA
    ax = axs[0]
    k = bins.index(HIGH[-1])
    g = Gb[k] / Yb[k]
    a = fits[HIGH[-1]]['a'][fits[HIGH[-1]]['S'].index(names.index('B*L M1-4'))] if names.index('B*L M1-4') in fits[HIGH[-1]]['S'] else np.nan
    x = np.arange(4)
    for j, (c, lab, col_) in enumerate((('NI', 'field $B$', '#D55E00'), ('Z_len', 'length $L$', '#0072B2'))):
        ax.bar(x + (j - 0.5) * 0.35, [g[LABELS.index(f'M{m}:{c}')] for m in (1, 2, 3, 4)], 0.35, color=col_, label=lab)
    ax.plot(x, a * w * DELTA, 'k_', ms=28, mew=2, label=fr'$a\,B_mL_m/\sum B L\cdot\delta$, $a={a:.0f}$')
    ax.set_xticks(x, [f'M{m}' for m in (1, 2, 3, 4)])
    ax.set_ylabel(r'$\partial\log Y/\partial u$ (per $+10\%$)')
    ax.set_title(f'gradient, $p_z$ {HIGH[-1]} GeV', fontsize=10)
    ax.set_ylim(-1.35, 0)
    ax.legend(fontsize=7.5, frameon=False, loc='lower center', ncol=3, markerscale=0.4)
    ax.axhline(0, color='k', lw=0.6)
    # (b) the cross term of B_m and L_m in the log-Hessian against its prediction DELTA * g_m
    ax = axs[1]
    mk = ['o', 's', '^', 'D', 'v', 'P']
    lim = 0
    for n, m_ in zip(HIGH, mk):
        k = bins.index(n)
        g = Gb[k] / Yb[k]
        K = Hb[k] / Yb[k] - np.outer(g, g)
        # c of f'' from the pairs of different magnets (no B*L cross term there), then the prediction
        # for the pair of one magnet: DELTA g_m (the B*L term) + c g_B g_L
        ix = [(LABELS.index(f'M{m}:NI'), LABELS.index(f'M{m}:Z_len')) for m in (1, 2, 3, 4)]
        flat = [i for p_ in ix for i in p_]
        other = [(i, j) for i in flat for j in flat if i < j and (i, j) not in ix and (j, i) not in ix]
        gg = np.array([g[i] * g[j] for i, j in other])
        c = gg @ np.array([K[i, j] for i, j in other]) / (gg @ gg)
        pr = [DELTA * (g[i] + g[j]) / 2 + c * g[i] * g[j] for i, j in ix]
        ob = [K[i, j] for i, j in ix]
        ax.scatter(pr, ob, marker=m_, s=30, label=f'{n} GeV', color='0.25', facecolor='none' if n != HIGH[-1] else '0.25')
        lim = max(lim, np.abs(pr).max(), np.abs(ob).max())
    ax.plot([-1.15 * lim, 1.15 * lim], [-1.15 * lim, 1.15 * lim], color='red', ls='--', lw=1, label='$y=x$')
    ax.set_xlabel(r'predicted $\delta\,g_m + c\,g_{B_m} g_{L_m}$')
    ax.set_ylabel(r'surrogate $\partial^2\log Y/\partial B_m\partial L_m$')
    ax.set_title('cross term of $B_m$ and $L_m$, M1-M4', fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    # (c) the simulated iso-B*L designs
    if has_sim:
        ax = axs[2]
        q = np.load(sim_file)
        e = list(q['edges'])
        hb = [k for k in range(1, len(e) + 1) if e[k - 1] >= float(q['band'])]
        lab_b = [bins[k] for k in hb]
        dn = [str(d) for d in q['designs']]
        cols = ['#009E73', '#56B4E9', '#D55E00', '#0072B2']
        xx = np.arange(len(hb))
        for i in range(1, len(dn)):
            off = (i - 2.5) * 0.18
            sim = q['sim'][i, hb] / q['sim'][0, hb]
            err = sim * np.sqrt(1 / q['sim'][i, hb] + 1 / q['sim'][0, hb])
            pm, ps = q['pred'][:, i, hb].mean(0), q['pred'][:, i, hb].std(0, ddof=1)
            ax.errorbar(xx + off, sim, yerr=err, fmt='o', color=cols[i - 1], ms=5, label=dn[i])
            ax.errorbar(xx + off, pm, yerr=ps, fmt='_', color=cols[i - 1], ms=12, mew=2)
        ax.axhline(1, color='k', lw=0.6)
        ax.set_yscale('log')
        ax.set_yticks([0.2, 0.5, 1, 2, 5], ['0.2', '0.5', '1', '2', '5'])
        ax.minorticks_off()
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(xx, lab_b)
        ax.set_xlabel('$p_z$ bin [GeV]')
        ax.set_ylabel('$Y/Y_0$')
        ax.set_title('simulated (dots) vs surrogate (bars)', fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc='upper left')
    fig.tight_layout()
    save_figure(fig, figures(args.run), 'relations_bl', 'relations_bl')

say.save(os.path.join(args.run, f'relations_{args.bins}_report.txt'))
