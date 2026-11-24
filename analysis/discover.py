"""Rediscovering the field integral from the surrogate, without being told it.

The surrogate's log hits per p_z bin, log Y_b(u), and their gradients are computed on --n_designs
random designs of the whole tolerance box (all 58 parameters varied), in log-parameter coordinates
x = log phi (elasticities dlogY/dlogphi). Nothing about magnets or fields goes in; each step below
reads structure off the surrogate and the next one builds on it:

  1. dimension: the global active subspace of log Y_b, C_b = E_u[grad grad^T]. If its first
     eigenvalue carries almost everything, Y_b depends on one combination of the parameters,
     locally a monomial prod phi_i^v_i (v the first eigenvector, in elasticity units).
  2. support: the parameters with a sizeable exponent, and how they pair (equal exponents of two
     parameters: they enter as their product).
  3. combination: whether the pairs P_m multiply or add. Their share of the exponent, design by
     design, is fitted with a CES aggregate (sum_m c_m P_m^rho)^(1/rho): rho = 0 is a product
     (fixed shares), rho = 1 a sum (shares that follow P_m). c_m are the weights of the terms.
  4. weights: c_m against the position of the magnets along z.
  5. collapse: log Y_b against the discovered variable, on the same random designs.

The simulated check (analysis.relations --sim, if <run>/relations_sim.npz exists) is drawn too.

    python -m analysis.discover                      # surrogate gradients on the designs, then plots
    python -m analysis.discover --plot_only
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import least_squares

import cli
from analysis.core import Report
from paths import RUN, figures, save_figure
from shield import COLUMNS, NOMINAL, MuonShield

parser = cli.add_common(cli.parser(__doc__), fixed=True)
parser.set_defaults(fixed='M0:')
parser.add_argument('--run', default=RUN, help='robustness.py / bins.py --out_dir')
parser.add_argument('--bins', default='pz', help='<run>/<bins>_bins.npz of bins.py (its edges)')
parser.add_argument('--n_designs', type=int, default=1024, help='random designs in the box')
parser.add_argument('--box', type=float, default=1.0, help='designs uniform in |u_i| <= box (1 = +-10%%)')
parser.add_argument('--batch', type=int, default=128, help='designs per surrogate pass')
parser.add_argument('--cache_grid', default='201x64', help='muon points of the surrogate')
parser.add_argument('--focus', default='300-inf', help='the bin of steps 2 and 3')
parser.add_argument('--support', type=float, default=0.25, help='step 2: exponents above this fraction of the largest')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--plot_only', action='store_true', help='from <run>/discover.npz')
args = parser.parse_args()
say = Report()
DELTA = MuonShield.DELTA
out_npz = os.path.join(args.run, 'discover.npz')
edges = np.load(os.path.join(args.run, f'{args.bins}_bins.npz'))['edges']
bin_labels = [f'{a:g}-{b:g}' for a, b in zip(np.r_[0, edges], np.r_[edges, np.inf])]
bin_labels[-1] = f'{edges[-1]:g}-inf'
nb = len(bin_labels)

# =============================================================================
# The surrogate's log hits and their gradients on random designs
# =============================================================================
if not args.plot_only:
    from analysis.core import feature_cache, members, setup
    shield, model = setup(args)
    ed = torch.tensor(edges, dtype=torch.float32)
    bin_of = lambda raw: torch.bucketize(raw[:, 2].contiguous(), ed.to(raw.device))
    g = torch.Generator().manual_seed(args.seed)
    U = (torch.rand(args.n_designs, shield.dim, generator=g) * 2 - 1) * args.box
    U[0] = 0                                              # the nominal design first
    G, LY = [], []
    for k, m in enumerate(members(args, shield)):
        c = feature_cache(m, shield.muons.points(args.cache_grid, 1e9), bin_of)
        gk, yk = [], []
        for i in range(0, len(U), args.batch):
            u = U[i:i + args.batch].to(c.t.device).requires_grad_(True)
            p = torch.sigmoid(m.branch_net(u) @ c.t.float().T + m.bias)
            ly = torch.zeros(len(u), nb, dtype=torch.float64, device=u.device).index_add(1, c.g, p.double() * c.w).log()
            gk.append(torch.stack([torch.autograd.grad(ly[:, j].sum(), u, retain_graph=j < nb - 1)[0]
                                   for j in range(nb)], 1).cpu().numpy())
            yk.append(ly.detach().cpu().numpy())
        G.append(np.concatenate(gk))
        LY.append(np.concatenate(yk))
        print(f'member {k}: {len(U)} designs', flush=True)
        del c
    np.savez(out_npz, U=U.numpy(), G=np.array(G, np.float32), LY=np.array(LY), labels=np.array(shield.labels),
             phi0=shield.phi_0.numpy(), edges=edges)
    say(f'saved {out_npz}')

r = np.load(out_npz)
U, G, LY, labels, phi0 = r['U'], r['G'].astype(np.float64), r['LY'], [str(l) for l in r['labels']], r['phi0']
K, N, _, D = G.shape
X = G * ((1 + DELTA * U) / DELTA)[None, :, None, :]         # dlogY/dlogphi on each design: (K, N, bins, D)
phi = phi0 * (1 + DELTA * U)
say(f'{K} members, {N} designs uniform in |u| <= {args.box:g}, {D} parameters; gradients in log phi\n')

# =============================================================================
# 1. Dimension: the global active subspace per bin
# =============================================================================
def eig(g):
    """Eigenvalues (fractions) and first eigenvector, in elasticity units, of E[g g^T]; oriented
    so that increasing it lowers Y."""
    C = g.T @ g / len(g)
    w, v = np.linalg.eigh(C)
    w, v = w[::-1], v[:, ::-1]
    v1 = v[:, 0] * -np.sign(v[:, 0] @ g.mean(0))
    return w / w.sum(), np.sqrt(w[0]) * v1


frac = np.array([[eig(X[k, :, b])[0][:3] for b in range(nb)] for k in range(K)])     # (K, bins, 3)
load = np.array([[eig(X[k, :, b])[1] for b in range(nb)] for k in range(K)])         # (K, bins, D)
say('1. dimension: share of E|grad log Y_b|^2 in the first three directions (members\' mean)')
for b in range(nb):
    top = np.argsort(-np.abs(load[:, b].mean(0)))[:5]
    say(f'  {bin_labels[b]:>8} GeV: {frac[:, b, 0].mean():.3f} {frac[:, b, 1].mean():.3f} {frac[:, b, 2].mean():.3f}   '
        + ' '.join(f'{labels[i]} {load[:, b, i].mean():+.1f}' for i in top))

# =============================================================================
# 2. Support and pairing, in the focus bin
# =============================================================================
fb = bin_labels.index(args.focus)
e = load[:, fb].mean(0)
sup = [i for i in np.argsort(-np.abs(e)) if abs(e[i]) > args.support * np.abs(e).max()]
say(f'\n2. support in {args.focus} GeV: exponents above {args.support:.0%} of the largest (members\' sd)')
for i in sup:
    say(f'  {labels[i]:<14} {e[i]:+6.2f} +- {load[:, fb, i].std(ddof=1):.2f}')
rest = [i for i in range(D) if i not in sup]
say(f'  the other {len(rest)}: largest |exponent| {np.abs(e[rest]).max():.2f} ({labels[rest[np.argmax(np.abs(e[rest]))]]}), '
    f'share of |v|^2 {np.sum(e[rest] ** 2) / np.sum(e ** 2):.1%}')
say('  by magnet: the parameters of each magnet in the support, and the ratio of the two largest exponents')
mags = []
for m in sorted({int(labels[i][1]) for i in sup}):
    mi = [i for i in sup if labels[i].startswith(f'M{m}:')]
    line = '    M' + str(m) + ': ' + ', '.join(f'{labels[i][3:]} {e[i]:+.2f}' for i in mi)
    if len(mi) >= 2:
        rat = load[:, fb, mi[1]] / load[:, fb, mi[0]]
        line += f'   ratio {rat.mean():.2f} +- {rat.std(ddof=1):.2f}'
        if 2 / 3 < rat.mean() < 3 / 2 and {labels[i][3:] for i in mi[:2]} == {'NI', 'Z_len'}:
            mags.append(m)
            line += '  -> pair: field x length'
    say(line)
pairs = [(labels.index(f'M{m}:NI'), labels.index(f'M{m}:Z_len')) for m in mags]
say(f'  pairs B_m L_m: {", ".join(f"M{m}" for m in mags)}; the rest of the support is a second, smaller term')
iB, iL = np.array(pairs).T
P = phi[:, iB] * phi[:, iL]                                   # B_m L_m on each design (T cm)

# =============================================================================
# 3. How the pairs combine: CES shares, design by design
# =============================================================================
def ces_fit(s):
    """rho, c (mean 1) of shares s (N, M) = c_m P_m^rho / sum c P^rho."""
    lP = np.log(P / P.mean())                          # one scale for all: c_m weigh B_m L_m in T cm

    def res(t):
        lc = np.r_[0, t[1:]]
        z = lc + t[0] * lP
        return (np.exp(z) / np.exp(z).sum(1, keepdims=True) - s).ravel()
    t = least_squares(res, np.r_[0.5, np.zeros(len(mags) - 1)]).x
    c = np.exp(np.r_[0, t[1:]])
    return t[0], c / c.mean()


def shares(k, b):
    ex = -(X[k, :, b][:, iB] + X[k, :, b][:, iL]) / 2       # the exponent of each pair on each design
    return ex / ex.sum(1, keepdims=True)


hi = [b for b in range(nb) if np.r_[0, edges][b] >= 125]
fits = [[ces_fit(shares(k, b)) for k in range(K)] for b in hi]
rho = np.array([[f[0] for f in fb_] for fb_ in fits])                       # (bins, K)
cw = np.array([[f[1] for f in fb_] for fb_ in fits])                        # (bins, K, M)
say(f'\n3. combination: the shares of the pairs, fitted by c_m P_m^rho / sum c P^rho (rho 0: product, 1: sum)')
for n, b in enumerate(hi):
    say(f'  {bin_labels[b]:>8} GeV: rho = {rho[n].mean():.2f} +- {rho[n].std(ddof=1):.2f}   c = '
        + ' '.join(f'{x:.3f}' for x in cw[n].mean(0)) + '  (+- ' + ' '.join(f'{x:.3f}' for x in cw[n].std(0, ddof=1)) + ')')

# =============================================================================
# 4. The weights against z: the lever arm to the sensitive planes
# =============================================================================
F0 = NOMINAL.numpy()
zc, z = [], 0.0
for row in F0:
    z += row[0] / 100
    zc.append(z + row[1] / 100)
    z += 2 * row[1] / 100
zm = np.array(zc)[mags]                                     # centres of the magnets (m)
zs = [p['position'] for p in MuonShield.PLANES]
lever = {s: (s - zm) / np.mean(s - zm) for s in zs}
cn = cw[hi.index(fb)]                                     # (K, M)
fit = np.array([np.polyfit(zm, c, 1) for c in cn])
z0 = -fit[:, 1] / fit[:, 0]
say(f'\n4. weights vs z (magnet centres {", ".join(f"{x:.1f}" for x in zm)} m), {args.focus} GeV:')
say('  fitted c_m  ' + ' '.join(f'{x:.3f}' for x in cn.mean(0)))
for s in zs:
    say(f'  (z_s - z_m), z_s = {s:g} m: ' + ' '.join(f'{x:.3f}' for x in lever[s]))
say(f'  c_m on a line through zero at z0 = {z0.mean():.0f} +- {z0.std(ddof=1):.0f} m; the planes at {zs[0]:g}, {zs[1]:g} m')

# =============================================================================
# 5. Collapse: log Y_b against the discovered variable
# =============================================================================
c_hat = cn.mean(0)
xi = (P * c_hat).sum(1)                                     # sum_m c_m B_m L_m
ly = LY.mean(0) - LY.mean(0)[0]                            # members' mean, relative to nominal
lphi = np.log(1 + DELTA * U)
others = [i for i in range(D) if i not in iB and i not in iL]


def r2(y, A):
    A = np.c_[A, np.ones(len(y))]
    return 1 - np.var(y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) / np.var(y)


def poly(v, n=3):
    return np.stack([v ** k for k in range(1, n + 1)], 1)


say(f'\n5. collapse: R^2 of log Y_b over the {N} designs')
rows = {'cubic in log Xi': lambda y, lx: r2(y, poly(lx)),
        'cubic in log sum B L': lambda y, lx: r2(y, poly(np.log(P.sum(1) / P[0].sum()))),
        'best single parameter (cubic)': lambda y, lx: max(r2(y, poly(lphi[:, i])) for i in range(D)),
        'cubic in log Xi + other params linear': lambda y, lx: r2(y, np.c_[poly(lx), lphi[:, others]]),
        'all 58 linear in log phi (monomial)': lambda y, lx: r2(y, lphi)}
lx = np.log(xi / xi[0])
R2 = {k: [f_(ly[:, b], lx) for b in hi] for k, f_ in rows.items()}
for k, v in R2.items():
    say(f'  {k:<40} ' + ' '.join(f'{bin_labels[b]}: {x:.3f}' for b, x in zip(hi, v)))
partial = {}                                                # log Y_b minus the others' fitted linear part
for b in hi:
    A = np.c_[poly(lx), lphi[:, others], np.ones(N)]
    c = np.linalg.lstsq(A, ly[:, b], rcond=None)[0]
    partial[b] = ly[:, b] - lphi[:, others] @ c[3:3 + len(others)] - c[-1]
say.save(os.path.join(args.run, 'discover_report.txt'))

# =============================================================================
# Figure
# =============================================================================
COL_B, COL_L, GREY = '#D55E00', '#0072B2', '0.65'
BINC = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7']
fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))

a = ax[0, 0]
xx = np.arange(nb)
f = frac.mean(0)
a.bar(xx, f[:, 0], color='k', label='first direction')
a.bar(xx, f[:, 1], bottom=f[:, 0], color=GREY, label='second')
a.bar(xx, f[:, 2], bottom=f[:, 0] + f[:, 1], color='0.85', label='third')
a.set_xticks(xx, bin_labels, rotation=45, fontsize=8)
a.set_xlabel('$p_z$ bin [GeV]')
a.set_ylabel(r'share of $E\,|\nabla\log Y_b|^2$')
a.set_ylim(0, 1.12)
a.legend(fontsize=7, frameon=False, loc='upper left', ncol=3)
a.set_title('(a) how many directions matter', fontsize=10, loc='left')

a = ax[0, 1]
order = sorted(range(D), key=lambda i: (labels[i][1], COLUMNS.index(labels[i][3:])))   # by magnet
col = [COL_B if labels[i].endswith(':NI') else COL_L if labels[i].endswith(':Z_len') else GREY for i in order]
a.bar(np.arange(D), e[order], color=col, width=0.8, yerr=load[:, fb].std(0, ddof=1)[order], error_kw=dict(lw=0.6, ecolor='k'))
mag_of = np.array([int(labels[i][1]) for i in order])
for m in np.unique(mag_of):
    idx = np.where(mag_of == m)[0]
    a.text(idx.mean(), e.max() * 1.08, f'M{m}', ha='center', fontsize=8)
    if idx[0] > 0:
        a.axvline(idx[0] - 0.5, color='0.85', lw=0.6)
a.set_xticks([])
a.set_xlim(-1, D)
a.set_ylim(min(0, e.min()) - 0.5, e.max() * 1.18)
a.set_xlabel('the 58 parameters, by magnet')
a.set_ylabel(r'$-\partial\log Y/\partial\log\phi$ along it')
from matplotlib.patches import Patch
a.legend(handles=[Patch(color=COL_B, label='field $B$'), Patch(color=COL_L, label='length $L$'),
                  Patch(color=GREY, label='others')], fontsize=7, frameon=False, loc='lower left')
a.set_title(f'(b) the first direction, $p_z$ {args.focus} GeV', fontsize=10, loc='left')

a = ax[0, 2]
w = P / P.sum(1, keepdims=True)
s = np.mean([shares(k, fb) for k in range(K)], 0)
dw, ds = w - w.mean(0), s - s.mean(0)
for m in range(len(mags)):
    a.scatter(dw[:, m], ds[:, m], s=4, alpha=0.4, color=BINC[m], label=f'M{mags[m]}', rasterized=True)
lim = np.abs(dw).max() * 1.1
t = np.linspace(-lim, lim, 2)
a.plot(t, t, 'k-', lw=1, label='sum ($\\rho=1$)')
a.plot(t, 0 * t, 'k:', lw=1, label='product ($\\rho=0$)')
rf = rho[hi.index(fb)]
a.text(0.97, 0.05, f'fitted $\\rho = {rf.mean():.2f}\\pm{rf.std(ddof=1):.2f}$', transform=a.transAxes, ha='right', fontsize=9)
a.set_xlabel(r'share of $B_mL_m$ in $\sum B L$ (minus its mean)')
a.set_ylabel("magnet's share of the exponent (minus its mean)")
a.legend(fontsize=7, frameon=False, loc='upper left', markerscale=3, ncol=2)
a.set_title('(c) the pairs add up', fontsize=10, loc='left')

a = ax[1, 0]
for n, b in enumerate(hi):
    a.errorbar(zm + (n - 1.5) * 0.25, cw[n].mean(0), yerr=cw[n].std(0, ddof=1), fmt='o', ms=4, color=BINC[n],
               label=f'{bin_labels[b]} GeV')
zz = np.linspace(zm.min() - 1, zm.max() + 1, 2)
for s_, ls in zip(zs, ('--', ':')):
    a.plot(zz, (s_ - zz) / np.mean(s_ - zm), 'k', ls=ls, lw=1, label=f'$(z_s - z)$, $z_s={s_:g}$ m')
a.axhline(1, color=GREY, lw=0.6)
a.set_xticks(zm, [f'M{m}\n{z_:.1f}' for m, z_ in zip(mags, zm)])
a.set_xlabel('magnet centre $z$ [m]')
a.set_ylabel('weight $c_m$ (mean 1)')
a.legend(fontsize=7, frameon=False, loc='upper right')
a.set_title('(d) the weights: lever arm to the planes', fontsize=10, loc='left')

a = ax[1, 1]
lx = np.log(xi / xi[0])
for n, b in enumerate(hi):
    a.scatter(lx, partial[b], s=3, alpha=0.35, color=BINC[n], rasterized=True,
              label=f'{bin_labels[b]} GeV')
a.axhline(0, color=GREY, lw=0.6)
a.axvline(0, color=GREY, lw=0.6)
a.set_xlabel(r'$\log(\Xi/\Xi_0)$, $\Xi=\sum_m c_m B_m L_m$')
a.set_ylabel(r'$\log(Y_b/Y_{b,0})$ minus the other parameters')
a.legend(fontsize=7, frameon=False, loc='upper right', markerscale=3)
a.set_title(f'(e) {N} random designs, all 58 parameters varied', fontsize=10, loc='left')

a = ax[1, 2]
sim = os.path.join(args.run, 'relations_sim.npz')
if os.path.exists(sim):
    q = np.load(sim, allow_pickle=True)
    se = q['edges']
    hb = [k for k in range(1, len(se) + 1) if se[k - 1] >= q['band']]
    lab_b = [f'{se[k - 1]:g}-{se[k]:g}' if k < len(se) else f'{se[k - 1]:g}-inf' for k in hb]
    dn = [str(d) for d in q['designs']]
    cols = ['#009E73', '#56B4E9', COL_B, COL_L]
    x2 = np.arange(len(hb))
    for i in range(1, len(dn)):
        off = (i - 2.5) * 0.15
        sv = q['sim'][i, hb] / q['sim'][0, hb]
        err = sv * np.sqrt(1 / q['sim'][i, hb] + 1 / q['sim'][0, hb])
        pm, ps = q['pred'][:, i, hb].mean(0), q['pred'][:, i, hb].std(0, ddof=1)
        a.errorbar(x2 + off, sv, yerr=err, fmt='o', color=cols[i - 1], ms=5, label=dn[i])
        a.errorbar(x2 + off, pm, yerr=ps, fmt='_', color=cols[i - 1], ms=12, mew=2)
    a.axhline(1, color='k', lw=0.6)
    a.set_yscale('log')
    a.set_yticks([0.2, 0.5, 1, 2, 5], ['0.2', '0.5', '1', '2', '5'])
    a.minorticks_off()
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x2, lab_b)
    a.set_xlabel('$p_z$ bin [GeV]')
    a.set_ylabel('$Y/Y_0$')
    a.legend(fontsize=7, frameon=False, loc='upper left')
    a.set_title('(f) M1-M4 changed: simulated (dots), surrogate (bars)', fontsize=10, loc='left')
else:
    a.axis('off')
fig.tight_layout()
save_figure(fig, figures(args.run), 'discover', 'discover')
