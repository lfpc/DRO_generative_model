"""Rediscovering the field integral from the surrogate, without being told it.

Nothing about magnets or fields goes in; each step reads structure off the surrogate and the
next builds on it. Two objects, both from the surrogate alone:

  * per muon, at the nominal design: the active subspace C_b = E_x[grad_u p grad_u p^T] over the
    muons x of each p_z bin (closed form for the DeepONet: grad_u p = p(1-p) J^T t(x));
  * per design: the gradient of log Y_b on --n_designs random designs of the whole box (all
    parameters varied), in log-parameter coordinates (elasticities dlogY/dlogphi).

  1. dimension: if C_b's first eigenvalue carries almost everything, every muon of the bin sees
     the design through one and the same combination of the parameters;
  2. its ingredients: the first eigenvector's support, and per magnet the two largest loadings;
  3. product and pairing: Y depends on B and L only through B L exactly when dlogY/dlogB =
     dlogY/dlogL on every design; its residual for every field with every length, and the
     assignment of lengths to fields with the smallest total;
  4. sum or product of the pairs P_m: at the nominal design they have the same gradient; across the
     designs a pair's share of the exponent follows its own P_m for a sum and is fixed for a
     product (CES aggregate (sum c_m P_m^rho)^(1/rho) fitted);
  4b. the weights: for a sum, dY/dP_m : dY/dP_n = c_m : c_n on every design, so c_m is measured;
     then a line in the magnets' positions, and where it reaches zero (the lever arm);
  5. collapse: log Y_b against Xi = sum c_m P_m on the random designs;
  6. second order: the nominal log-Hessian on the fields and lengths against f(log Xi);
  7. the simulated check of analysis.relations --sim (<run>/relations_sim.npz), if present;
  and the regimes (the first direction per bin) and the soft muons' entrance step.

Figures: discover_variable (steps 1-5) and discover_action (6, 7, the regimes, the soft muons) for the report, and one PNG per step,
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


# 3b. Product and pairing, from one identity. Y depends on B and L only through B L exactly when
# dlogY/dlogB = dlogY/dlogL on every design (the characteristics of x df/dx = y df/dy are xy =
# const). Its residual over the designs, for every field with every length: the pairing is the
# assignment of lengths to fields with the smallest total residual.
import itertools
from scipy.optimize import linear_sum_assignment

# Which length goes with which field, across the designs: for a sum of terms B_m L_sigma(m), field
# m's share of the exponent follows its own term's share (slope 1, nothing fitted, no weights): the
# variance of the field shares that each of the 24 pairings explains.
perms = list(itertools.permutations(range(len(mags))))
ident = perms.index(tuple(range(len(mags))))


def pairing_r2(k, b, perm):
    gB = X[k, :, b][:, iB]
    sB = gB / gB.sum(1, keepdims=True)
    T = phi[:, iB] * phi[:, iL[list(perm)]]
    w = T / T.sum(1, keepdims=True)
    ds, dw = sB - sB.mean(0), w - w.mean(0)
    return 1 - ((ds - dw) ** 2).sum() / (ds ** 2).sum()


pr2 = np.array([[[pairing_r2(k, b, p) for p in perms] for k in range(K)] for b in hi])    # (bins, K, perms)


def identity_residual(k, b):
    a_, c_ = X[k, :, b][:, iB], X[k, :, b][:, iL]
    return np.array([[((a_[:, m] - c_[:, n]) ** 2).mean() / ((a_[:, m] ** 2 + c_[:, n] ** 2) / 2).mean()
                      for n in range(len(mags))] for m in range(len(mags))])


IDR = np.array([[identity_residual(k, b) for k in range(K)] for b in hi])        # (bins, K, M, M)
assign = [linear_sum_assignment(IDR[n].mean(0))[1] for n in range(len(hi))]
say('\n3b. pairing: variance of the field shares across the designs explained by each pairing of fields and lengths (no fit, no weights)')
for n, b in enumerate(hi):
    m_ = pr2[n].mean(0)
    o = np.argsort(-m_)
    say(f'  {bin_labels[b]:>8} GeV: own magnet {m_[ident]:.2f} +- {pr2[n, :, ident].std(ddof=1):.2f}; best other '
        f'{m_[o[o != ident][0]]:.2f}; median {np.median(m_):.2f}; rank of own magnet {list(o).index(ident) + 1} of {len(perms)}')
say('    product: residual of dlogY/dlogB_m = dlogY/dlogL_n over the designs (rows: fields, columns: lengths)')
for n, b in enumerate(hi):
    Rm = IDR[n].mean(0)
    say(f'  {bin_labels[b]:>8} GeV: assignment ' + ' '.join(f'B{mags[m]}-L{mags[a_]}' for m, a_ in enumerate(assign[n]))
        + f'; own magnet {np.trace(Rm) / len(mags):.3f} per pair, best other assignment '
        + f'{min(sum(Rm[m, p[m]] for m in range(len(mags))) for p in itertools.permutations(range(len(mags))) if list(p) != list(range(len(mags)))) / len(mags):.3f}')
    for m in range(len(mags)):
        say(f'      B{mags[m]}: ' + ' '.join(f'{x:.3f}' for x in Rm[m]))

# 3c. The soft muons: the two entrance widths that lead at 10-30 GeV. Equal and opposite loadings
# in log phi: one antisymmetric combination. A ratio d3/d2 has dlogY/dlog d2 = -dlogY/dlog d3 on
# every design; a difference d2 - d3 has dY/dd2 = -dY/dd3 (in cm). Family:
# dlogY/dlog d3 = -(d3/d2)^s dlogY/dlog d2, s = 0 ratio, 1 difference.
from scipy.optimize import minimize_scalar
soft = [b for b in range(nb) if 10 <= np.r_[0, edges][b] < 30]
lead = {b: list(np.argsort(-np.abs(vec[:, b].mean(0)))[:2]) for b in soft}
step_pair = sorted(lead[soft[0]], key=lambda i: labels[i])
i2, i3 = step_pair
qd = phi[:, i3] / phi[:, i2]
say(f'\n3c. soft muons: {labels[i2]} and {labels[i3]} (nominal {phi0[i2]:g} and {phi0[i3]:g} cm)')
step = {}
for b in soft:
    v_ = vec[:, b].mean(0)
    say(f'  {bin_labels[b]:>6} GeV: loadings {v_[i2]:+.2f} {v_[i3]:+.2f} ({v_[i2] ** 2 + v_[i3] ** 2:.0%} of the direction)')
    rr, rd, ss = [], [], []
    for k in range(K):
        gl = X[k, :, b][:, [i2, i3]]
        ga = gl / phi[:, [i2, i3]]
        rr.append((gl.sum(1) ** 2).mean() / (gl ** 2).sum(1).mean())
        rd.append((ga.sum(1) ** 2).mean() / (ga ** 2).sum(1).mean())
        ss.append(minimize_scalar(lambda s: ((gl[:, 1] + qd ** s * gl[:, 0]) ** 2).mean(), bounds=(-5, 5), method='bounded').x)
    step[b] = (np.array(rr), np.array(rd), np.array(ss))
    say(f'          antisymmetry residual on the designs: as a ratio {np.mean(rr):.3f}, as a difference {np.mean(rd):.3f}; '
        f's = {np.mean(ss):+.1f} +- {np.std(ss, ddof=1):.1f} (0 ratio, 1 difference)')

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

# 4b. The weights. For a sum s = sum_m c_m P_m the ratio of the plain derivatives dY/dP_m : dY/dP_n
# is c_m : c_n on every design: the weights are measured, not fitted. Then what they depend on:
# a line in the magnet's position z_m, and where it reaches zero.
def weights(k, b):
    eP = (X[k, :, b][:, iB] + X[k, :, b][:, iL]) / 2           # dlogY/dlogP_m
    dP = eP / P                                                 # proportional to dY/dP_m
    w_ = (dP / dP.mean(1, keepdims=True)).mean(0)
    return w_ / w_.mean(), (dP / dP.mean(1, keepdims=True)).std(0) / w_


CW = np.array([[weights(k, b)[0] for k in range(K)] for b in hi])               # (bins, K, M)
CWv = np.array([[weights(k, b)[1] for k in range(K)] for b in hi])
LINE = np.array([[np.polyfit(zm, CW[n, k], 1) for k in range(K)] for n in range(len(hi))])   # (bins, K, 2)
Z0 = -LINE[..., 1] / LINE[..., 0]
LRES = np.array([[np.abs(np.polyval(LINE[n, k], zm) - CW[n, k]).max() for k in range(K)] for n in range(len(hi))])
say(f'\n4b. the weights per unit B L, c_m from dY/dP_m : dY/dP_n (mean 1), at the magnets\' centres z = '
    + ' '.join(f'{z_:.2f}' for z_ in zm) + ' m')
for n, b in enumerate(hi):
    say(f'  {bin_labels[b]:>8} GeV: c = ' + ' '.join(f'{x:.3f}' for x in CW[n].mean(0)) + '  (+- '
        + ' '.join(f'{x:.3f}' for x in CW[n].std(0, ddof=1)) + f'); spread over the designs {CWv[n].mean():.0%}; '
        f'a line in z: max residual {LRES[n].mean():.3f}, zero at z0 = {Z0[n].mean():.0f} +- {Z0[n].std(ddof=1):.0f} m')
say(f'  the sensitive planes at {zs[0]:g} and {zs[1]:g} m: the weight is the distance from the magnet to near the detector')
c_xi = CW[hi.index(fb)].mean(0)

# =============================================================================
# 5. Collapse
# =============================================================================
xi = (P * c_xi).sum(1)                                      # sum_m c_m B_m L_m, the measured weights
lx = np.log(xi / xi[0])
ly = LY.mean(0) - LY.mean(0)[0]
lphi = np.log(1 + DELTA * U)
others = [i for i in range(D) if i not in iB and i not in iL]


def r2(y, A):
    A = np.c_[A, np.ones(len(y))]
    return 1 - np.var(y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) / np.var(y)


def poly(v, n=3):
    return np.stack([v ** k for k in range(1, n + 1)], 1)


say(f'\n5. collapse: R^2 of log Y_b over the {N} designs; Xi = sum c_m B_m L_m, c_m measured in 4b ({args.focus} GeV)')
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

# 6. Second order. If log Y = f(log Xi), the Hessian of log Y at nominal on the fields and lengths
# of M1-M4 is f' Hess(log Xi) + f'' q q^T, q = grad log Xi: Hess(log Xi) holds each magnet's
# field-length cross term (the product) and -q q^T (the sum); f' is the gradient along Xi, f''
# the one number fitted. On bins.py's Hessian (the ensemble mean, --grid points), in u; the NI
# diagonals are left out, B(NI) being nonlinear (saturation).
pzb = np.load(os.path.join(args.run, f'{args.bins}_bins.npz'))
idx8 = [i for pr in pairs for i in pr]
HES = {}
say(f'\n6. second order: the log-Hessian of Y_b on the {len(idx8)} fields and lengths at nominal, against log Y = f(log Xi)')
for b in hi:
    Yb, gb, Hb = pzb['H_b'][b], pzb['grad_b'][b], pzb['hess_b'][b]
    gl = gb / Yb
    Lm = (Hb / Yb - np.outer(gl, gl))[np.ix_(idx8, idx8)]
    f1 = np.mean(gl[iL] / (DELTA * c_xi * P[0] / (c_xi * P[0]).sum()))    # f' from the lengths' gradient
    q = gl[idx8] / f1
    sh = c_xi * P[0] / (c_xi * P[0]).sum()
    Cx = np.zeros((len(idx8), len(idx8)))
    for m in range(len(mags)):
        Cx[2 * m, 2 * m + 1] = Cx[2 * m + 1, 2 * m] = q[2 * m] * q[2 * m + 1] / sh[m]
    Q = np.outer(q, q)
    A0 = f1 * (Cx - Q)
    keep = ~(np.eye(len(idx8), dtype=bool) & (np.arange(len(idx8)) % 2 == 0)[:, None])
    f2 = ((Lm - A0)[keep] @ Q[keep]) / (Q[keep] @ Q[keep])
    tot = ((Lm[keep] - Lm[keep].mean()) ** 2).sum()
    R2h = lambda M_: 1 - ((Lm - M_)[keep] ** 2).sum() / tot
    f2q = (Lm[keep] @ Q[keep]) / (Q[keep] @ Q[keep])
    HES[b] = dict(meas=Lm, pred=A0 + f2 * Q, keep=keep, r2=R2h(A0 + f2 * Q), r2_noprod=R2h(f2q * Q))
    say(f'  {bin_labels[b]:>8} GeV: f\' = {f1:.1f}, f\'\' = {f2:+.1f} (fitted); R^2 {HES[b]["r2"]:.2f}; '
        f'without the products\' cross terms (f\'\' q q^T alone) {HES[b]["r2_noprod"]:.2f}')
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
    zz = np.linspace(0, max(zs[1], Z0.mean() + 5), 2)
    for n, b in enumerate(hi):
        if n == 0:
            continue
        cm_, ce_ = CW[n].mean(0), CW[n].std(0, ddof=1)
        a.errorbar(zm + (n - 2) * 0.25, cm_, yerr=ce_, fmt='o', color=BINC[n], ms=7 if big else 5, capsize=2,
                   label=f'{bin_labels[b]} GeV: zero at {Z0[n].mean():.0f} $\\pm$ {Z0[n].std(ddof=1):.0f} m')
        a.plot(zz, np.polyval(LINE[n].mean(0), zz), '-', color=BINC[n], lw=0.9)
    for z_ in zs:
        a.axvline(z_, color='k', lw=2)
    a.text(zs[0], 0.06, ' sensitive\n planes', fontsize='small', va='bottom', ha='right')
    for m, z_ in zip(mags, zm):
        a.text(z_, 0.06, f'M{m}', ha='center', fontsize='small')
    a.axhline(0, color=GREY, lw=0.6)
    a.set_xlim(0, zs[1] + 3)
    a.set_ylim(0, 1.25)
    a.set_xlabel('z of the magnet\'s centre [m]')
    a.set_ylabel('weight per unit $B_mL_m$, $c_m$ (mean 1)')
    a.legend(frameon=False, loc='upper center', fontsize='small')


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
    a.set_xlabel(r'$\log(\Xi/\Xi_0)$, $\Xi=\sum_m c_mB_mL_m$')
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



def p_pairing(a, big):
    m_ = pr2.mean(1)                                       # (bins, perms)
    o = np.argsort(-m_[hi.index(fb)])
    x = np.arange(len(perms))
    for n, b in enumerate(hi):
        if n == 0:
            continue
        a.plot(x, m_[n][o], 'o-', ms=3 if not big else 4, lw=0.8, color=BINC[n], label=f'{bin_labels[b]} GeV')
    a.plot([list(o).index(ident)], [m_[hi.index(fb)][ident]], 'k*', ms=14 if big else 11, zorder=5, label='own magnet')
    a.set_xticks(x, [''.join(f'{p + 1}' for p in perms[i]) for i in o], rotation=90, fontsize='x-small')
    a.set_xlabel(r'pairing: the lengths of M$_{\sigma(1)}$..M$_{\sigma(4)}$ with the fields of M1..M4')
    a.set_ylabel('variance of the field shares explained')
    a.axhline(0, color=GREY, lw=0.6)
    a.legend(frameon=False, loc='upper right', fontsize='small')


def p_hessian(a, big):
    for n, b in enumerate(hi):
        if n == 0:
            continue
        h = HES[b]
        a.scatter(h['pred'][h['keep']], h['meas'][h['keep']], s=14 if not big else 22, color=BINC[n], alpha=0.8,
                  edgecolors='k', linewidths=0.3, label=f'{bin_labels[b]} GeV: $R^2$ {h["r2"]:.2f} '
                  f'(no product term {h["r2_noprod"]:.2f})')
    v = max(max(np.abs(HES[b]['meas'][HES[b]['keep']]).max() for b in hi[1:]), 1e-3) * 1.1
    a.plot([-v, v], [-v, v], 'k-', lw=0.8)
    a.axhline(0, color=GREY, lw=0.5)
    a.axvline(0, color=GREY, lw=0.5)
    a.set_xlabel(r"$f'\,\mathrm{Hess}(\log\Xi) + f''qq^\top$ (one number fitted)")
    a.set_ylabel(r'$\partial^2\log Y/\partial u_i\partial u_j$, surrogate')
    a.legend(frameon=False, loc='upper left', fontsize='small')


def p_step(a, big):
    for n, b in enumerate(soft):
        g = X[:, :, b].mean(0)
        a.scatter(g[:, i2], g[:, i3], s=4 if not big else 6, alpha=0.4, color=BINC[n], rasterized=True,
                  label=f'{bin_labels[b]} GeV')
    gg = np.concatenate([X[:, :, b].mean(0)[:, [i2, i3]] for b in soft])
    lim = np.abs(gg).max() * 1.05
    t_ = np.linspace(-lim, lim, 2)
    a.plot(t_, -t_, 'k-', lw=1.2, label=f'ratio {labels[i3][3:]}$_3$/{labels[i2][3:]}$_2$')
    a.plot(t_, -phi0[i3] / phi0[i2] * t_, 'k--', lw=1.2, label='difference (nominal widths)')
    a.set_xlim(min(gg[:, 0].min(), 0) - 2, gg[:, 0].max() + 2)       # the quadrant the designs are in
    a.set_ylim(gg[:, 1].min() - 2, max(gg[:, 1].max(), 0) + 2)
    a.set_aspect('equal', adjustable='datalim')
    a.axhline(0, color=GREY, lw=0.5)
    a.axvline(0, color=GREY, lw=0.5)
    a.set_xlabel(f'$\\partial\\log Y/\\partial\\log$ {labels[i2]}')
    a.set_ylabel(f'$\\partial\\log Y/\\partial\\log$ {labels[i3]}')
    a.legend(frameon=False, loc='lower left', markerscale=3, fontsize='small')


f1, f3 = frac[:, fb, 0].mean(), rho[hi.index(fb)].mean()
PANELS = {   # name: (panel, report title, slide title), in the order of the argument
    'dimension': (p_dimension, 'how many directions, per muon',
                  f'Above {edges[4]:g} GeV every muon sees the design through one variable'),
    'direction': (p_direction, f'the direction, $p_z$ {args.focus} GeV',
                  'That variable is field and length of M1-M4, in equal measure'),
    'pairing': (p_pairing, f'which length goes with which field, across {N} designs',
                "The hits see each field times its own magnet's length"),
    'weights': (p_weights, 'the weight of each magnet: the lever arm',
                'Each B L counts by its distance to near the detector'),
    'regimes': (p_regimes, 'the direction per $p_z$ bin',
                'Two regimes: apertures for soft muons, field integral for hard ones'),
    'step': (p_step, 'soft muons: the two entrances, across 1024 designs',
             'Soft muons see the step between the entrances of M2 and M3'),
    'sum': (p_sum, 'sum or product, across 1024 designs',
            f'The magnets add up: a sum, not a product ($\\rho={f3:.2f}$)'),
    'collapse': (p_collapse, 'collapse on 1024 random designs',
                 'All designs fall on one curve per momentum bin'),
    'hessian': (p_hessian, 'second order: fields and lengths of M1-M4 at nominal',
                'The curvature has the structure of a sum of products'),
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


compose(['dimension', 'direction', 'pairing', 'sum', 'weights', 'collapse'], 'discover_variable', (3, 2), (12, 12.6))
compose(['hessian', 'sim', 'regimes', 'step'], 'discover_action', (2, 2), (12, 9.4))
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
