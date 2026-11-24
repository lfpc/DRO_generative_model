"""Rediscovering the field integral from the surrogate, without being told it.

Nothing about magnets or fields goes in; each step reads structure off the surrogate and the
next builds on it. Two objects, both from the surrogate alone:

  * per muon, at the nominal design: the active subspace C_b = E_x[grad_u p grad_u p^T] over the
    muons x of each p_z bin (closed form for the DeepONet: grad_u p = p(1-p) J^T t(x));
  * per design: the gradient of log Y_b on --n_designs random designs of the whole box (all
    parameters varied), in log-parameter coordinates (elasticities dlogY/dlogphi).

  1. dimension: if C_b's first eigenvalue carries almost everything, every muon of the bin sees
     the design through one and the same combination of the parameters;
  2. the combination: its parameters (the first eigenvector), how they pair (equal loadings: a
     product), and how its weights split between the pairs (against B_m L_m and against
     B_m L_m (z_s - z_m), the lever arm to the sensitive plane);
  3. the regimes: the first eigenvector per bin;
  4. sum or product: at the nominal design a sum and a product of the pairs have the same
     gradient; across the designs a pair's share of the exponent follows its own B_m L_m for a
     sum and is fixed for a product. Fitted with the CES aggregate (sum c_m P_m^rho)^(1/rho);
  5. collapse: log Y_b against the discovered variable on the random designs;
  6. the simulated check of analysis.relations --sim (<run>/relations_sim.npz), if present.

Figures: discover_variable (1-3) and discover_action (4-6) for the report, and one PNG per step,
with larger type and a one-line conclusion as title, in <figures>/discover_slides/.

    python -m analysis.discover                      # surrogate on the designs and the muons, then plots
    python -m analysis.discover --plot_only
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
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
parser.add_argument('--focus', default='300-inf', help='the bin of steps 2 and 4')
parser.add_argument('--support', type=float, default=0.25, help='step 2: loadings above this fraction of the largest')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--plot_only', action='store_true', help='from <run>/discover.npz and discover_muon.npz')
args = parser.parse_args()
say = Report()
DELTA = MuonShield.DELTA
design_npz, muon_npz = (os.path.join(args.run, f) for f in ('discover.npz', 'discover_muon.npz'))
edges = np.load(os.path.join(args.run, f'{args.bins}_bins.npz'))['edges']
bin_labels = [f'{a:g}-{b:g}' for a, b in zip(np.r_[0, edges], np.r_[edges, np.inf])]
bin_labels[-1] = f'{edges[-1]:g}-inf'
nb = len(bin_labels)

# =============================================================================
# The surrogate: per muon at nominal, and per design
# =============================================================================
if not args.plot_only:
    from analysis.core import branch_jacobian, feature_cache, members, setup
    shield, model = setup(args)
    ed = torch.tensor(edges, dtype=torch.float32)
    bin_of = lambda raw: torch.bucketize(raw[:, 2].contiguous(), ed.to(raw.device))
    gen = torch.Generator().manual_seed(args.seed)
    U = (torch.rand(args.n_designs, shield.dim, generator=gen) * 2 - 1) * args.box
    U[0] = 0                                              # the nominal design first
    G, LY, CM = [], [], []
    for k, m in enumerate(members(args, shield)):
        c = feature_cache(m, shield.muons.points(args.cache_grid, 1e9), bin_of)
        # per muon: C_b = J^T E_x[(p(1-p))^2 t t^T] J at u = 0
        J = branch_jacobian(m)
        with torch.no_grad():
            t = c.t.float()
            p = torch.sigmoid(t @ m.branch_net(torch.zeros(1, shield.dim, device=t.device))[0] + m.bias)
            s2 = (p * (1 - p)).double() ** 2 * c.w
            cm = []
            for b in range(nb):
                sel = c.g == b
                tb = t[sel].double()
                cm.append((J.T @ ((tb * s2[sel, None]).T @ tb / c.w[sel].sum()) @ J).cpu().numpy())
        CM.append(cm)
        # per design: grad log Y_b
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
    np.savez(design_npz, U=U.numpy(), G=np.array(G, np.float32), LY=np.array(LY), labels=np.array(shield.labels),
             phi0=shield.phi_0.numpy(), edges=edges)
    np.savez(muon_npz, C=np.array(CM), labels=np.array(shield.labels), edges=edges)
    say(f'saved {design_npz}, {muon_npz}')

r = np.load(design_npz)
U, G, LY, labels, phi0 = r['U'], r['G'].astype(np.float64), r['LY'], [str(l) for l in r['labels']], r['phi0']
CM = np.load(muon_npz)['C']                                  # (K, bins, D, D), in u
K, N, _, D = G.shape
X = G * ((1 + DELTA * U) / DELTA)[None, :, None, :]         # dlogY/dlogphi on each design: (K, N, bins, D)
phi = phi0 * (1 + DELTA * U)
fb = bin_labels.index(args.focus)
hi = [b for b in range(nb) if np.r_[0, edges][b] >= 125]
say(f'{K} members; per muon at nominal, and on {N} designs uniform in |u| <= {args.box:g}; {D} parameters\n')

# =============================================================================
# 1. Dimension, per muon
# =============================================================================
g_nom = X[:, 0].mean(0)                                     # (bins, D): the bins' gradients at nominal


def first(C, b):
    """Eigenvalue shares and first eigenvector (unit) of C; oriented so that a positive loading
    lowers the bin's hits."""
    w, v = np.linalg.eigh(C)
    w, v = w[::-1], v[:, ::-1]
    return w / w.sum(), v[:, 0] * -np.sign(v[:, 0] @ g_nom[b])


E = [[first(CM[k, b], b) for b in range(nb)] for k in range(K)]
frac = np.array([[e[0][:3] for e in row] for row in E])     # (K, bins, 3)
vec = np.array([[e[1] for e in row] for row in E])          # (K, bins, D)
say('1. dimension: share of E_x[|grad p|^2] in the first three directions, per muon (members\' mean)')
for b in range(nb):
    top = np.argsort(-np.abs(vec[:, b].mean(0)))[:5]
    say(f'  {bin_labels[b]:>8} GeV: {frac[:, b, 0].mean():.3f} {frac[:, b, 1].mean():.3f} {frac[:, b, 2].mean():.3f}   '
        + ' '.join(f'{labels[i]} {vec[:, b, i].mean():+.2f}' for i in top))
one = [b for b in range(nb) if frac[:, b, 0].mean() > 0.9]
say(f'  first direction above 90%: {", ".join(bin_labels[b] for b in one)}')

# =============================================================================
# 2. The combination: support, pairs, weights
# =============================================================================
e = vec[:, fb].mean(0)
sup = [i for i in np.argsort(-np.abs(e)) if abs(e[i]) > args.support * np.abs(e).max()]
say(f'\n2. the first direction in {args.focus} GeV: loadings above {args.support:.0%} of the largest (members\' sd)')
for i in sup:
    say(f'  {labels[i]:<14} {e[i]:+.3f} +- {vec[:, fb, i].std(ddof=1):.3f}')
rest = [i for i in range(D) if i not in sup]
say(f'  the other {len(rest)}: share of |v|^2 {np.sum(e[rest] ** 2):.1%}')
say('  by magnet: its parameters in the support, and the ratio of the two largest loadings')
mags = []
for m in sorted({int(labels[i][1]) for i in sup}):
    mi = [i for i in sup if labels[i].startswith(f'M{m}:')]
    line = f'    M{m}: ' + ', '.join(f'{labels[i][3:]} {e[i]:+.3f}' for i in mi)
    if len(mi) >= 2:
        rat = vec[:, fb, mi[1]] / vec[:, fb, mi[0]]
        line += f'   ratio {rat.mean():.2f} +- {rat.std(ddof=1):.2f}'
        if 2 / 3 < rat.mean() < 3 / 2 and {labels[i][3:] for i in mi[:2]} == {'NI', 'Z_len'}:
            mags.append(m)
            line += '  -> pair: field x length'
    say(line)
pairs = [(labels.index(f'M{m}:NI'), labels.index(f'M{m}:Z_len')) for m in mags]
iB, iL = np.array(pairs).T
P = phi[:, iB] * phi[:, iL]                                   # B_m L_m on each design (T cm)
zc, z = [], 0.0
for row in NOMINAL.numpy():
    z += row[0] / 100
    zc.append(z + row[1] / 100)
    z += 2 * row[1] / 100
zm = np.array(zc)[mags]                                     # centres of the paired magnets (m)
zs = [p['position'] for p in MuonShield.PLANES]
share = (vec[:, :, iB] + vec[:, :, iL]) / (vec[:, :, iB] + vec[:, :, iL]).sum(-1, keepdims=True)   # (K, bins, M)
w_bl = P[0] / P[0].sum()
w_lev = {s: P[0] * (s - zm) / (P[0] * (s - zm)).sum() for s in zs}
say(f'  the pairs\' shares of the direction ({args.focus} GeV) against B_m L_m and B_m L_m (z_s - z_m):')
say('    eigenvector   ' + ' '.join(f'{x:.3f}' for x in share[:, fb].mean(0)) + '  (+- ' + ' '.join(f'{x:.3f}' for x in share[:, fb].std(0, ddof=1)) + ')')
say('    B L           ' + ' '.join(f'{x:.3f}' for x in w_bl))
for s in zs:
    say(f'    B L (z_s - z), z_s = {s:g} m ' + ' '.join(f'{x:.3f}' for x in w_lev[s]))
lam = np.array([np.polyfit(zm, share[k, fb] / w_bl, 1) for k in range(K)])
z0 = -lam[:, 1] / lam[:, 0]
say(f'  eigenvector / B L on a line in z through zero at z0 = {z0.mean():.0f} +- {z0.std(ddof=1):.0f} m; the planes at {zs[0]:g}, {zs[1]:g} m')

# =============================================================================
# 3. The regimes: the first direction per bin
# =============================================================================
rows = []
for b in range(nb):
    for i in np.argsort(-np.abs(vec[:, b].mean(0)))[:4]:
        if i not in rows:
            rows.append(i)
for i_ in np.ravel(pairs):                                   # every paired field and length, M1 included
    if i_ not in rows:
        rows.append(i_)
PAIR_FIRST = {'Z_len': 0, 'NI': 1}                          # per magnet: length, field, then the others
rows = sorted(rows, key=lambda i: (labels[i][1], PAIR_FIRST.get(labels[i][3:], 2), COLUMNS.index(labels[i][3:])))
say(f'\n3. the first direction per bin, on the {len(rows)} parameters that lead in some bin')
for b in range(nb):
    say(f'  {bin_labels[b]:>8}: ' + ' '.join(f'{labels[i]} {vec[:, b, i].mean():+.2f}' for i in rows if abs(vec[:, b, i].mean()) > 0.15))

# =============================================================================
# 4. Sum or product: the shares across the designs
# =============================================================================
def ces_fit(s):
    """rho, c (mean 1) of shares s (N, M) = c_m P_m^rho / sum c P^rho."""
    lP = np.log(P / P.mean())                           # one scale for all: c_m weigh B_m L_m in T cm

    def res(t):
        z_ = np.r_[0, t[1:]] + t[0] * lP
        return (np.exp(z_) / np.exp(z_).sum(1, keepdims=True) - s).ravel()
    t = least_squares(res, np.r_[0.5, np.zeros(len(mags) - 1)]).x
    c = np.exp(np.r_[0, t[1:]])
    return t[0], c / c.mean()


def shares(k, b):
    ex = -(X[k, :, b][:, iB] + X[k, :, b][:, iL]) / 2       # each pair's exponent on each design
    return ex / ex.sum(1, keepdims=True)


fits = [[ces_fit(shares(k, b)) for k in range(K)] for b in hi]
rho = np.array([[f[0] for f in row] for row in fits])      # (bins, K)
cw = np.array([[f[1] for f in row] for row in fits])       # (bins, K, M)
say('\n4. sum or product: the pairs\' shares across the designs, fitted by c_m P_m^rho / sum c P^rho (rho 0: product, 1: sum)')
for n, b in enumerate(hi):
    say(f'  {bin_labels[b]:>8} GeV: rho = {rho[n].mean():.2f} +- {rho[n].std(ddof=1):.2f}   c = '
        + ' '.join(f'{x:.3f}' for x in cw[n].mean(0)) + '  (+- ' + ' '.join(f'{x:.3f}' for x in cw[n].std(0, ddof=1)) + ')')

# =============================================================================
# 5. Collapse
# =============================================================================
xi = (P * w_lev[zs[0]] / w_bl).sum(1)                     # sum_m B_m L_m (z_s - z_m), up to a constant
lx = np.log(xi / xi[0])
ly = LY.mean(0) - LY.mean(0)[0]
lphi = np.log(1 + DELTA * U)
others = [i for i in range(D) if i not in iB and i not in iL]


def r2(y, A):
    A = np.c_[A, np.ones(len(y))]
    return 1 - np.var(y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) / np.var(y)


def poly(v, n=3):
    return np.stack([v ** k for k in range(1, n + 1)], 1)


say(f'\n5. collapse: R^2 of log Y_b over the {N} designs; Xi = sum B_m L_m (z_s - z_m), z_s = {zs[0]:g} m')
R2 = {'cubic in log Xi': [r2(ly[:, b], poly(lx)) for b in hi],
      'best single parameter (cubic)': [max(r2(ly[:, b], poly(lphi[:, i])) for i in range(D)) for b in hi],
      'cubic in log Xi + the others linear': [r2(ly[:, b], np.c_[poly(lx), lphi[:, others]]) for b in hi],
      'all 58 linear in log phi (monomial)': [r2(ly[:, b], lphi) for b in hi]}
for k_, v in R2.items():
    say(f'  {k_:<38} ' + ' '.join(f'{bin_labels[b]}: {x:.3f}' for b, x in zip(hi, v)))
partial, elast = {}, {}
for b in hi:
    A = np.c_[poly(lx), lphi[:, others], np.ones(N)]
    c = np.linalg.lstsq(A, ly[:, b], rcond=None)[0]
    partial[b] = ly[:, b] - lphi[:, others] @ c[3:3 + len(others)] - c[-1]
    elast[b] = c[0]
say('  dlogY/dlogXi at nominal: ' + ' '.join(f'{bin_labels[b]}: {elast[b]:.1f}' for b in hi))
say.save(os.path.join(args.run, 'discover_report.txt'))

# =============================================================================
# Panels: each draws one conclusion on an axis
# =============================================================================
COL_B, COL_L, GREY = '#D55E00', '#0072B2', '0.65'
BINC = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7']
MAGC = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7']


def p_dimension(a, big):
    xx = np.arange(nb)
    f = frac.mean(0)
    a.bar(xx, f[:, 0], color='k', label='first direction')
    a.bar(xx, f[:, 1], bottom=f[:, 0], color=GREY, label='second')
    a.bar(xx, f[:, 2], bottom=f[:, 0] + f[:, 1], color='0.85', label='third')
    a.set_xticks(xx, bin_labels, rotation=45)
    a.set_xlabel('$p_z$ bin [GeV]')
    a.set_ylabel(r'share of $E_x\,|\nabla_u p|^2$')
    a.set_ylim(0, 1.12)
    a.legend(frameon=False, loc='upper left', ncol=3, fontsize='small')


def p_direction(a, big):
    order = sorted(range(D), key=lambda i: (labels[i][1], COLUMNS.index(labels[i][3:])))
    col = [COL_B if labels[i].endswith(':NI') else COL_L if labels[i].endswith(':Z_len') else GREY for i in order]
    a.bar(np.arange(D), e[order], color=col, width=0.8, yerr=vec[:, fb].std(0, ddof=1)[order],
          error_kw=dict(lw=0.6, ecolor='k'))
    mag_of = np.array([int(labels[i][1]) for i in order])
    top = e.max() * 1.1
    for m in np.unique(mag_of):
        idx = np.where(mag_of == m)[0]
        a.text(idx.mean(), top, f'M{m}', ha='center', fontsize='small')
        if idx[0] > 0:
            a.axvline(idx[0] - 0.5, color='0.85', lw=0.6)
    a.axhline(0, color='k', lw=0.5)
    a.set_xticks([])
    a.set_xlim(-1, D)
    a.set_ylim(min(0, e.min()) - 0.05, e.max() * 1.22)
    a.set_xlabel(f'the {D} parameters, by magnet')
    a.set_ylabel(f'loading, $p_z$ {args.focus} GeV')
    a.legend(handles=[Patch(color=COL_B, label='field $B$'), Patch(color=COL_L, label='length $L$'),
                      Patch(color=GREY, label='others')], frameon=False, loc='lower left', fontsize='small')


def p_regimes(a, big):
    M = vec.mean(0)[:, rows].T                              # (rows, bins)
    v = np.abs(M).max()
    im = a.imshow(M, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    a.set_yticks(range(len(rows)), [labels[i] for i in rows], fontsize='x-small' if not big else 'small')
    a.set_xticks(range(nb), bin_labels, rotation=45)
    a.set_xlabel('$p_z$ bin [GeV]')
    for j in range(nb):
        for i in range(len(rows)):
            if abs(M[i, j]) > 0.2:
                a.text(j, i, f'{M[i, j]:+.2f}'.replace('0.', '.'), ha='center', va='center', fontsize=6 if not big else 8,
                       color='w' if abs(M[i, j]) > 0.5 * v else 'k')
    mag = [labels[i][:2] for i in rows]
    for k in range(1, len(rows)):
        if mag[k] != mag[k - 1]:
            a.axhline(k - 0.5, color='k', lw=1.2)
    for k, i in enumerate(rows):
        if labels[i].endswith(':NI') and k and rows[k - 1] == labels.index(labels[i][:3] + 'Z_len'):
            a.add_patch(plt.Rectangle((-0.5, k - 1.5), nb, 2, fill=False, ec='k', lw=0.6, ls='--'))
    plt.colorbar(im, ax=a, fraction=0.04, pad=0.02, label='loading (+: fewer hits)')


def p_weights(a, big):
    x = np.arange(len(mags))
    a.bar(x, share[:, fb].mean(0), width=0.6, color='0.3', yerr=share[:, fb].std(0, ddof=1), capsize=3,
          label=f'first direction, {args.focus} GeV')
    a.plot(x, w_bl, 's', color=COL_L, ms=8 if big else 6, label=r'$B_mL_m/\sum BL$')
    a.plot(x, w_lev[zs[0]], 'D', color=COL_B, ms=8 if big else 6, label=fr'$B_mL_m(z_s-z_m)/\sum$, $z_s={zs[0]:g}$ m')
    a.set_xticks(x, [f'M{m}\n$z={z_:.1f}$ m' for m, z_ in zip(mags, zm)])
    a.set_ylabel("magnet's share of the direction")
    lo = min(share[:, fb].mean(0).min(), w_bl.min()) * 0.85
    a.set_ylim(lo, max(share[:, fb].mean(0).max(), w_bl.max()) * 1.13)
    a.legend(frameon=False, loc='upper left', fontsize='small')


def p_sum(a, big):
    w = P / P.sum(1, keepdims=True)
    s = np.mean([shares(k, fb) for k in range(K)], 0)
    dw, ds = w - w.mean(0), s - s.mean(0)
    for m in range(len(mags)):
        a.scatter(dw[:, m], ds[:, m], s=4 if not big else 6, alpha=0.4, color=MAGC[m], label=f'M{mags[m]}', rasterized=True)
    lim = np.abs(dw).max() * 1.1
    t = np.linspace(-lim, lim, 2)
    a.plot(t, t, 'k-', lw=1.2, label=r'sum, $\rho=1$')
    a.plot(t, 0 * t, 'k:', lw=1.2, label=r'product, $\rho=0$')
    rf = rho[hi.index(fb)]
    a.text(0.97, 0.05, f'fitted $\\rho = {rf.mean():.2f}\\pm{rf.std(ddof=1):.2f}$', transform=a.transAxes, ha='right')
    a.set_xlabel(r'share of $B_mL_m$ in $\sum BL$ (minus its mean)')
    a.set_ylabel("share of the exponent (minus its mean)")
    a.legend(frameon=False, loc='upper left', markerscale=3, ncol=2, fontsize='small')


def p_collapse(a, big):
    for n, b in enumerate(hi):
        a.scatter(lx, partial[b], s=3 if not big else 5, alpha=0.35, color=BINC[n], rasterized=True,
                  label=fr'{bin_labels[b]} GeV, $Y\propto\Xi^{{{elast[b]:.0f}}}$')
    a.axhline(0, color=GREY, lw=0.6)
    a.axvline(0, color=GREY, lw=0.6)
    a.set_xlabel(r'$\log(\Xi/\Xi_0)$, $\Xi=\sum_m B_mL_m(z_s-z_m)$')
    a.set_ylabel(r'$\log(Y_b/Y_{b,0})$, other parameters removed')
    a.legend(frameon=False, loc='upper right', markerscale=3, fontsize='small')


def p_sim(a, big):
    q = np.load(os.path.join(args.run, 'relations_sim.npz'), allow_pickle=True)
    se = q['edges']
    hb = [k for k in range(1, len(se) + 1) if se[k - 1] >= q['band']]
    lab_b = [f'{se[k - 1]:g}-{se[k]:g}' if k < len(se) else f'{se[k - 1]:g}-inf' for k in hb]
    names = ['$BL$ fixed: $B$ +5%, $L$ /1.05', '$BL$ fixed: $B$ -5%, $L$ /0.95', '$B$ +5%', '$L$ /1.05']
    cols = ['#009E73', '#56B4E9', COL_B, COL_L]
    x2 = np.arange(len(hb))
    for i in range(1, len(q['designs'])):
        off = (i - 2.5) * 0.15
        sv = q['sim'][i, hb] / q['sim'][0, hb]
        err = sv * np.sqrt(1 / q['sim'][i, hb] + 1 / q['sim'][0, hb])
        pm, ps = q['pred'][:, i, hb].mean(0), q['pred'][:, i, hb].std(0, ddof=1)
        a.errorbar(x2 + off, sv, yerr=err, fmt='o', color=cols[i - 1], ms=6 if big else 5, label=names[i - 1])
        a.errorbar(x2 + off, pm, yerr=ps, fmt='_', color=cols[i - 1], ms=14 if big else 12, mew=2)
    a.axhline(1, color='k', lw=0.6)
    a.set_yscale('log')
    a.set_yticks([0.2, 0.5, 1, 2, 5], ['0.2', '0.5', '1', '2', '5'])
    a.minorticks_off()
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x2, lab_b)
    a.set_xlabel('$p_z$ bin [GeV]')
    a.set_ylabel('$Y/Y_0$: simulated (dots), surrogate (bars)')
    a.legend(frameon=False, loc='upper left', fontsize='small')


f1, f3 = frac[:, fb, 0].mean(), rho[hi.index(fb)].mean()
PANELS = {   # name: (panel, report title, slide title)
    'dimension': (p_dimension, 'how many directions, per muon',
                  f'Above {edges[4]:g} GeV every muon sees the design through one variable'),
    'direction': (p_direction, f'the direction, $p_z$ {args.focus} GeV',
                  'That variable is field and length of M1-M4, in equal measure'),
    'regimes': (p_regimes, 'the direction per $p_z$ bin',
                'Two regimes: apertures for soft muons, field integral for hard ones'),
    'weights': (p_weights, "the magnets' shares: the lever arm",
                'Each magnet counts by B L times its distance to the detector'),
    'sum': (p_sum, 'sum or product, across 1024 designs',
            f'The magnets add up: a sum, not a product ($\\rho={f3:.2f}$)'),
    'collapse': (p_collapse, 'collapse on 1024 random designs',
                 'All designs fall on one curve per momentum bin'),
    'sim': (p_sim, 'M1-M4 changed: simulated and surrogate',
            'The simulator agrees: only the product B L matters'),
}
sim_ok = os.path.exists(os.path.join(args.run, 'relations_sim.npz'))


def compose(names, name, shape, size):
    fig, ax = plt.subplots(*shape, figsize=size)
    for n, (a, key) in enumerate(zip(np.ravel(ax), names)):
        if key == 'sim' and not sim_ok:
            a.axis('off')
            continue
        PANELS[key][0](a, False)
        a.set_title(f'({chr(97 + n)}) {PANELS[key][1]}', fontsize=10, loc='left')
    fig.tight_layout()
    save_figure(fig, figures(args.run), name, name)
    plt.close(fig)


compose(['dimension', 'direction', 'regimes', 'weights'], 'discover_variable', (2, 2), (12, 8.6))
compose(['sum', 'collapse', 'sim'], 'discover_action', (1, 3), (15, 4.6))
slides = os.path.join(figures(args.run), 'discover_slides')
os.makedirs(slides, exist_ok=True)
with plt.rc_context({'font.size': 13}):
    for n, (key, (fn, _, title)) in enumerate(PANELS.items()):
        if key == 'sim' and not sim_ok:
            continue
        fig, a = plt.subplots(figsize=(8, 5.4))
        fn(a, True)
        a.set_title(title, fontsize=14, loc='left')
        fig.tight_layout()
        fig.savefig(os.path.join(slides, f'{n + 1}_{key}.png'), dpi=200)
        plt.close(fig)
say(f'slides: {slides}/')
