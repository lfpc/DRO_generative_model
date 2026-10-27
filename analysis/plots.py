"""Every figure of a robustness run, drawn from its saved results (nothing is recomputed).

Reads whichever of these exist in <run> and writes figures/<run name>/ (paths.figures):
  robustness.npz     robustness_{sensitivity,magnets,spectra,directions,walks,summary,calibration,mc,scans,plane}.png
  worst_cases.npz    worst_direction_scaled, worst_parameters, random_tolerances, sim_pz_profiles .png
  <var>_bins.npz     <var>_bins_{overview,sensitivity,alignment,walks,mc}.png
  scan_sim_*.npz     (scan_sim.py) the simulated points of robustness_scans.png
  tolerance_boxes.npz (tolerance_boxes.py) tolerance_boxes, tolerance_allocation .png

    python -m analysis.plots outputs/results/robustness
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.core import BAD, GREY, OK
import cli
from paths import RUN, figures

parser = cli.parser(__doc__)
parser.add_argument('run', nargs='?', default=RUN)
parser.add_argument('--n_sigma', type=float, default=3.0, help='significance for "resolved"')
parser.add_argument('--profiles', default='nominal,worst@5%,worst@10%,worst@-10%,box,single4,rand10%_3',
                    help='worst-case designs shown in sim_pz_profiles.png')
args = parser.parse_args()
FIGS = figures(args.run)
os.makedirs(FIGS, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=150)
    plt.close(fig)
    print(f'saved {FIGS}/{name}')


def poisson_ratio(h, h0):
    """h/h0 and its Poisson error, the two counts treated as independent."""
    rt = h / h0
    return rt, rt * np.sqrt(1 / np.maximum(h, 1) + 1 / h0)


def log_ylim(ax, *curves):
    if all(np.all(c > 0) for c in curves):
        ax.set_yscale('log')
        ax.set_ylim(0.5 * min(c.min() for c in curves), 2 * max(c.max() for c in curves))


# =============================================================================
# robustness.npz
# =============================================================================
def robustness_figures(r):
    labels, D, H0 = list(r['labels']), len(r['labels']), float(r['H_0'])
    NAMES = [str(s) for s in r['dir_names']]
    sim = r['sim']
    SIM = sim[0] > 0
    if SIM:
        sr, sr_se = poisson_ratio(sim, sim[0])
        sr_se[0] = 0

    # --- per-parameter sensitivity
    lin = r['grad'] / H0
    op, om, op_se, om_se = r['oat_plus'], r['oat_minus'], r['oat_plus_se'], r['oat_minus_se']
    worse = np.where(np.abs(op) > np.abs(om), op, om)
    resolved = np.abs(worse) > args.n_sigma * np.where(np.abs(op) > np.abs(om), op_se, om_se)
    o = np.argsort(-np.abs(worse))[::-1]
    y = np.arange(D)
    fig, ax = plt.subplots(1, 2, figsize=(13, 9), constrained_layout=True, sharey=True)
    ax[0].barh(y, lin[o], color=[(BAD if v > 0 else OK) if res else GREY for v, res in zip(lin[o], resolved[o])])
    ax[0].set_yticks(y, [labels[k] for k in o], fontsize=7)
    ax[0].axvline(0, color='k', lw=0.8)
    ax[0].set(xlabel='$(\\partial H/\\partial u_i)\\,/\\,H_0$   (per full tolerance)', title='First-order sensitivity')
    ax[1].hlines(y, om[o], op[o], color=GREY, lw=1)
    ax[1].errorbar(op[o], y, xerr=op_se[o], fmt='o', ms=4, color=BAD, lw=0.8, label='$u_i = +1$')
    ax[1].errorbar(om[o], y, xerr=om_se[o], fmt='o', ms=4, color=OK, lw=0.8, label='$u_i = -1$')
    ax[1].axvline(0, color='k', lw=0.8)
    ax[1].set(xlabel='$H/H_0 - 1$', title='Exact one-at-a-time, $\\pm$10%')
    ax[1].legend()
    fig.suptitle(f'Parameter sensitivity of the hit count; grey = not resolved at {args.n_sigma:g}$\\sigma$  '
                 f'({resolved.sum()}/{D} resolved)')
    save(fig, 'robustness_sensitivity.png')

    # --- per magnet
    magnet = np.array([int(s[1]) for s in labels])
    mags = np.unique(magnet)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax[0].bar([f'M{m}' for m in mags], [np.linalg.norm(lin[magnet == m]) / np.linalg.norm(lin) for m in mags], color=OK)
    ax[0].set(ylabel='$\\|\\nabla_{\\mathrm{magnet}} H\\| / \\|\\nabla H\\|$', title='Share of the gradient per magnet')
    ax[1].bar([f'M{m}' for m in mags], [np.abs(worse[magnet == m]).max() for m in mags], color=BAD)
    ax[1].set(ylabel='max $|H/H_0 - 1|$', title='Largest single-parameter effect per magnet')
    save(fig, 'robustness_magnets.png')

    # --- spectra
    mu, mu_se, nu = r['hess_eigvals'], r['hess_eigval_se'], r['act_eigvals']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].errorbar(np.arange(1, D + 1), mu / H0, yerr=mu_se / H0, fmt='o-', ms=4, color=OK, lw=1, capsize=2)
    ax[0].set(xlabel='index (most positive first)', ylabel='eigenvalue $/H_0$', title='Hessian at the nominal design')
    ax[0].set_yscale('symlog', linthresh=max(1e-3, np.abs(mu / H0).max() * 1e-3))
    pos = np.arange(1, (nu > nu.max() * 1e-12).sum() + 1)
    ax[1].semilogy(pos, nu[:len(pos)], 'o-', ms=4, color=OK)
    ax[1].set(xlabel='index', ylabel='eigenvalue', title=f'Active subspace, participation {float(r["act_participation"]):.1f}')
    tw = ax[1].twinx()
    tw.plot(pos, (np.cumsum(nu) / nu.sum())[:len(pos)], 's--', ms=3, color=BAD)
    tw.plot(pos, np.cumsum((r['act_eigvecs'].T @ r['grad'] / np.linalg.norm(r['grad'])) ** 2)[:len(pos)],
            '^:', ms=3, color='0.3')
    tw.set_ylabel('cumulative share (red) / of $\\nabla H$ (grey)', color=BAD)
    tw.set_ylim(0, 1.02)
    fig.suptitle('Curvature and active directions')
    save(fig, 'robustness_spectra.png')

    # --- directions
    M = r['dir_vecs']
    fig, ax = plt.subplots(figsize=(max(9, D * 0.28), 1.4 + 0.55 * len(NAMES)), constrained_layout=True)
    lim = np.abs(M).max()
    im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(np.arange(D), labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(NAMES)), NAMES)
    fig.colorbar(im, ax=ax, label='component (unit $\\ell_2$)')
    ax.set_title('Directions in parameter space')
    save(fig, 'robustness_directions.png')

    # --- walks
    nr = int(np.ceil(len(NAMES) / 3))
    fig, axes = plt.subplots(nr, 3, figsize=(13.8, 3.9 * nr), constrained_layout=True, squeeze=False)
    for a in axes.ravel()[len(NAMES):]:
        a.axis('off')
    for a, nm, v, ts, idx in zip(axes.ravel(), NAMES, M, r['dir_t'], r['dir_idx']):
        tt = np.linspace(ts[0], ts[-1], 101)
        a.plot(tt, 1 + (tt * (r['grad'] @ v) + 0.5 * tt ** 2 * (v @ r['hess'] @ v)) / H0, ':', color='0.4', lw=1.5,
               label='quadratic model')
        a.errorbar(ts, r['ratio'][idx], yerr=r['ratio_se'][idx], fmt='o-', color=OK, ms=4, lw=2, capsize=2,
                   label='surrogate')
        curves = [r['ratio'][idx]]
        if SIM:
            a.errorbar(ts, sr[idx], yerr=sr_se[idx], fmt='s', ms=5, color=BAD, capsize=3, lw=1.5, label='simulator')
            curves.append(sr[idx])
        a.axvline(0, color=GREY, lw=1, ls='--')
        a.axhline(1, color=GREY, lw=0.8, ls=':')
        a.set(title=nm, xlabel='step $t$ along $v$ (normalized units)', ylabel='$H(t)\\,/\\,H_0$')
        log_ylim(a, *curves)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Surrogate vs. simulator along the robustness directions')
    save(fig, 'robustness_walks.png')

    # --- one tolerance unit along each direction
    unit = [(nm, idx[-1]) for nm, idx, ts in zip(NAMES, r['dir_idx'], r['dir_t']) if abs(ts[-1] - 1) < 1e-3]
    x = np.arange(len(unit))
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    ax.bar(x - 0.2, [r['ratio'][i] for _, i in unit], 0.4, yerr=[r['ratio_se'][i] for _, i in unit], color=OK,
           label='surrogate', capsize=2)
    if SIM:
        ax.bar(x + 0.2, [sr[i] for _, i in unit], 0.4, yerr=[sr_se[i] for _, i in unit], color=BAD,
               label='simulator', capsize=2)
    ax.axhline(1, color='k', lw=0.8, ls=':')
    ax.set_xticks(x, [nm for nm, _ in unit])
    ax.set(ylabel='$H/H_0$ at $\\|u\\| = 1$', title='One tolerance unit along each direction (= one parameter at its $\\pm$10% face)')
    ax.legend()
    save(fig, 'robustness_summary.png')

    # --- calibration over every simulated design
    if SIM:
        fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
        for j, (nm, idx, ts) in enumerate(zip(NAMES, r['dir_idx'], r['dir_t'])):
            k = [i for i, t in zip(idx, ts) if t != 0 and sim[i] > 0]
            ax.errorbar(sr[k], r['ratio'][k], xerr=sr_se[k], yerr=r['ratio_se'][k], fmt='o', ms=5,
                        color=plt.cm.tab10(j % 10), label=nm, capsize=2)
        lo, hi = 0.5 * min(sr[sr > 0].min(), r['ratio'].min()), 2 * max(sr.max(), r['ratio'].max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        ax.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo, hi), xlabel='simulator $H/H_0$',
               ylabel='surrogate $H/H_0$',
               title=f'Every walked design (nominal: surrogate {r["pred"][0]:.0f}, simulator {sim[0]:.0f} hits)')
        ax.legend(fontsize=8)
        save(fig, 'robustness_calibration.png')

    # --- random designs in the box
    mc, star = r['mc'], float(r['h_star_out'])
    q = np.quantile(mc, [0.5, 0.95])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    edges = np.logspace(np.log10(max(mc.min(), 1e-8) * 0.9), np.log10(max(mc.max(), star) * 1.2), 60)
    ax.hist(np.clip(mc, edges[0], edges[-1]), bins=edges, color=OK, alpha=0.8)
    ax.set_xscale('log')
    ax.axvline(1.0, color='k', lw=1.2, ls='--', label='nominal')
    ax.axvline(q[0], color=OK, lw=1, ls='-.', label=f'median {q[0]:.2f}')
    ax.axvline(q[1], color=GREY, lw=1, ls=':', label=f'95th percentile {q[1]:.1f}')
    ax.axvline(star, color=BAD, lw=1.5, label=f'worst found by the surrogate ({star:.2g})')
    if SIM and 'pga' in NAMES:
        i = r['dir_idx'][NAMES.index('pga')][-1]
        ax.axvline(sr[i], color=BAD, lw=1.5, ls='--', label=f'same design, simulated ({sr[i]:.1f})')
    ax.set(xlabel='$H/H_0$ (surrogate)', ylabel='designs', title=f'{len(mc):,} designs uniform in the $\\pm$10% box')
    ax.legend(fontsize=8)
    save(fig, 'robustness_mc.png')

    if 'scan' not in r:
        return
    # --- 1-D scans: the surrogate, and the simulated points of scan_sim.py if there are any
    sims = [np.load(f) for f in sorted(glob.glob(os.path.join(args.run, 'scan_sim_*.npz')))]
    n = len(r['scan'])
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.1 * ((n + 1) // 2), 5.6), constrained_layout=True,
                             sharex=True, squeeze=False)
    for j, (k, s) in enumerate(zip(r['scan_params'], r['scan'])):
        ax = axes.flat[j]
        ax.plot(r['scan_t'] * 10, s, '-', lw=2, color=OK, label='surrogate')
        for i, ss in enumerate(sims):
            m = ss['params'] == k
            if m.any():
                rt, se = poisson_ratio(ss['hits'][m], float(ss['nominal']))
                ax.errorbar(ss['t'][m] * 10, rt, yerr=se, fmt='s', ms=5, color=BAD, capsize=2,
                            label='simulator' if i == 0 else None)
        ax.axhline(1, color='k', lw=0.8, ls=':')
        ax.axvline(0, color=GREY, lw=0.8, ls='--')
        ax.set_title(labels[k], fontsize=10)
        if j % axes.shape[1] == 0:
            ax.set_ylabel('$H/H_0$')
        if j >= axes.shape[1]:
            ax.set_xlabel('change (% of nominal)')
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    handles = {l: h for ax in axes.flat for h, l in zip(*ax.get_legend_handles_labels())}
    axes.flat[0].legend(handles.values(), handles.keys(), fontsize=8)
    fig.suptitle(f'The {n} most sensitive parameters, one at a time' +
                 (f' (simulator: {int(sims[0]["n_muons"]):.0e} muons, Poisson errors)' if sims else ''))
    save(fig, 'robustness_scans.png')

    # --- the plane of two directions, with the simulated points of their walks
    s, plane = r['plane_s'], r['plane']
    (a_name, b_name), (va, vb) = [str(x) for x in r['plane_dirs']], r['plane_axes']
    fig, ax = plt.subplots(figsize=(7, 5.8), constrained_layout=True)
    lv = np.log10(plane)
    lim = np.abs(lv).max()
    im = ax.pcolormesh(s, s, lv, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
    ax.contour(s, s, plane, levels=[0.5, 1, 2, 4, 8], colors='k', linewidths=0.7)
    fig.colorbar(im, ax=ax, label='$\\log_{10} H/H_0$ (surrogate)')
    if SIM:
        for nm, marker in ((a_name, 's'), (b_name, 'o')):
            k = NAMES.index(nm)
            for t, i in zip(r['dir_t'][k], r['dir_idx'][k]):
                if t != 0:
                    p = t * M[k]
                    ax.scatter(p @ va, p @ vb, c=[np.log10(max(sr[i], 1e-3))], cmap='RdBu_r', vmin=-lim, vmax=lim,
                               marker=marker, s=90, edgecolors='k', zorder=3)
                    ax.annotate(f'{sr[i]:.2f}', (p @ va, p @ vb), textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax.plot(0, 0, 'k+', ms=12)
    ax.set(xlabel=f'along {a_name} (normalized units)', ylabel=f'along {b_name}, orthogonal to {a_name}',
           title=f'Surrogate in the {a_name}-{b_name} plane; markers: simulator')
    save(fig, 'robustness_plane.png')


# =============================================================================
# worst_cases.npz
# =============================================================================
def worst_case_figures(w):
    names = [str(s) for s in w['names']]
    idx = {nm: d for d, nm in enumerate(names)}
    U, labels, units, phis = w['U'], list(w['labels']), list(w['units']), w['phis']
    SIM = bool(w['simulated'])
    pred, sim = w['pred_bin'].sum(1), w['sim_bin'].sum(1)
    P = pred / pred[0]
    if SIM:
        S, S_se = poisson_ratio(sim, sim[0])

    def describe(nm):
        d = idx[nm]
        if nm.startswith('single'):
            k = np.flatnonzero(U[d])[0]
            return f'{labels[k]} {10 * U[d][k]:+.0f}% ({phis[d][k] - phis[0][k]:+.3g} {units[k]})'
        if nm.startswith('worst-top'):
            return f'worst, top {nm[9:]} only'
        if nm.startswith('worst@-'):
            return f'opposite of worst, max {nm[7:]}'
        if nm.startswith('worst@'):
            return f'worst, max {nm[6:]}'
        if nm.startswith('rand'):
            return f'a random $\\pm${nm[4:nm.index("%")]}% design'
        return {'box': 'box worst case'}.get(nm, nm)

    # --- the worst direction, scaled by the largest change of any parameter
    v = U[idx['worst@10%']]
    wd = sorted([d for nm, d in idx.items() if nm.startswith('worst@')], key=lambda d: 10 * np.abs(U[d]).max() * np.sign(U[d] @ v))
    x = np.array([10 * np.abs(U[d]).max() * np.sign(U[d] @ v) for d in wd])
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.plot(np.r_[x, 0], np.r_[P[wd], 1], 'o', color=OK, ms=6, label='surrogate')
    if SIM:
        ax.errorbar(np.r_[x, 0], np.r_[S[wd], 1], yerr=np.r_[S_se[wd], 0], fmt='s', ms=6, color=BAD, capsize=3, label='simulator')
        for xi, s in zip(x, S[wd]):
            ax.annotate(f'x{s:.2f}', (xi, s), textcoords='offset points', xytext=(6, -12), fontsize=8)
    ax.axhline(1, color='k', lw=0.8, ls=':')
    ax.set_yscale('log')
    ax.set(xlabel='largest change of any parameter along the worst direction (% of nominal)', ylabel='$H/H_0$',
           title=f'Worst direction, {int((v != 0).sum())} parameters moved together')
    ax.legend()
    save(fig, 'worst_direction_scaled.png')

    # --- the worst direction truncated to its largest changes, and single parameters
    groups = [nm for nm in names if nm.startswith('worst-top')] + ['worst@10%'] + [nm for nm in names if nm.startswith('single')]
    ds = [idx[nm] for nm in groups]
    y = np.arange(len(ds))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(ds) + 1.5), constrained_layout=True)
    ax.barh(y + 0.2, P[ds], 0.4, color=OK, label='surrogate')
    if SIM:
        ax.barh(y - 0.2, S[ds], 0.4, xerr=S_se[ds], color=BAD, label='simulator', capsize=2)
        for yi, s, e in zip(y, S[ds], S_se[ds]):
            ax.annotate(f'{s:.2f} ± {e:.2f}', (s + e, yi - 0.2), xytext=(4, -3), textcoords='offset points', fontsize=7)
    ax.axvline(1, color='k', lw=0.8, ls=':')
    ax.set_yticks(y, [describe(nm).replace('worst, max 10%', f'worst, all {int((v != 0).sum())} parameters (max 10%)')
                      for nm in groups], fontsize=8)
    ax.set(xlabel='$H/H_0$', title='How much of the worst case each parameter (or group) accounts for')
    ax.legend(loc='lower right')
    save(fig, 'worst_parameters.png')

    # --- random tolerances
    pcts = list(w['mc_pcts'])
    fig, axes = plt.subplots(1, len(pcts), figsize=(5 * len(pcts), 4), constrained_layout=True, squeeze=False)
    for a, x, mr in zip(axes[0], pcts, w['mc_ratio']):
        a.hist(mr, bins=np.logspace(np.log10(mr.min() * 0.9), np.log10(max(mr.max(), 1.1) * 1.1), 40), color=OK,
               alpha=0.7, label=f'surrogate ({len(mr)} designs)')
        ds = [d for nm, d in idx.items() if nm.startswith(f'rand{x:g}%_')]
        for j, d in enumerate(ds):
            a.axvline(P[d], color=OK, lw=1, ls=':', label='surrogate, simulated designs' if j == 0 else '_')
            if SIM:
                a.axvline(S[d], color=BAD, lw=1.5, label='simulator' if j == 0 else '_')
        a.axvline(1, color='k', lw=1, ls='--')
        a.set(xscale='log', xlabel='$H/H_0$', ylabel='designs', title=f'every parameter uniform in $\\pm${x:g}%')
        a.legend(fontsize=7)
    fig.suptitle('Random independent tolerances')
    save(fig, 'random_tolerances.png')

    # --- where the extra hits come from: hits per p_z range
    ze = w['pz_edges']
    ranges = [(1, 30), (30, 40), (40, 50), (50, 100), (100, 150), (150, 400)]
    H = w['sim_bin'] if SIM else w['pred_bin']
    Hz = np.stack([H[:, (ze[:-1] >= a) & (ze[1:] <= b)].sum(1) for a, b in ranges], 1)
    show = [nm for nm in args.profiles.split(',') if nm in idx]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x, wd = np.arange(len(ranges)), 0.8 / len(show)
    for j, nm in enumerate(show):
        h = Hz[idx[nm]]
        ax.bar(x - 0.4 + wd * (j + 0.5), np.maximum(h, 0.5), wd, yerr=np.sqrt(np.maximum(h, 1)) if SIM else None,
               color='k' if nm == 'nominal' else plt.cm.tab10(j), label=describe(nm), capsize=1.5, error_kw={'lw': 0.7})
    ax.set_yscale('log')
    ax.set_xticks(x, [f'{a}-{b}' for a, b in ranges])
    ax.set(xlabel='$p_z$ [GeV]', ylabel=f'{"simulated" if SIM else "predicted"} hits ({float(w["n_muons"]):.0e} muons)',
           title='Where the extra hits come from: hits per $p_z$ range')
    ax.legend(fontsize=8, ncol=2)
    save(fig, 'sim_pz_profiles.png')


# =============================================================================
# <var>_bins.npz
# =============================================================================
def bin_figures(b):
    var, SHORT, labels = str(b['bin_var']), list(b['bin_labels']), list(b['labels'])
    B, D = len(SHORT), len(labels)
    H_b, Hn = b['H_b'], np.maximum(b['H_b'], 1e-30)
    gn = np.linalg.norm(b['grad_b'], axis=1)
    ghat = b['grad_b'] / np.maximum(gn, 1e-30)[:, None]
    curv = np.einsum('bi,bij,bj->b', ghat, b['hess_b'], ghat)
    COL = [plt.cm.viridis(k / max(B - 1, 1) * 0.9) for k in range(B)]
    xb = np.arange(B)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ax[0, 0].bar(xb, b['counts'] / b['counts'].sum(), color=COL)
    ax[0, 0].set(ylabel='share of the muons', title='Population')
    ax[0, 1].bar(xb, H_b / H_b.sum(), color=COL)
    ax[0, 1].set(ylabel='share of the hits', title='Where the hits come from')
    ax[1, 0].bar(xb, gn / Hn, color=COL)
    ax[1, 0].set(ylabel='$|\\nabla H_b| / H_b$', title='Relative sensitivity')
    ax[1, 1].bar(xb, curv / Hn, color=[BAD if c > 0 else OK for c in curv])
    ax[1, 1].axhline(0, color='k', lw=0.8)
    ax[1, 1].set(ylabel='$v^T \\nabla^2 H_b v / H_b$ along its own $v$', title='Curvature')
    for a in ax.ravel():
        a.set_xticks(xb, SHORT, rotation=30, ha='right', fontsize=8)
    fig.suptitle(f'Muon regions in {var}')
    save(fig, f'{var}_bins_overview.png')

    rel = b['grad_b'] / Hn[:, None]
    o = np.argsort(-np.abs(rel).max(0))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(4 + 0.9 * B, 10), constrained_layout=True, sharey=True)
    for a, M, title in ((ax[0], rel[:, o].T, 'First order: $(\\partial H_b/\\partial u_i)/H_b$'),
                        (ax[1], np.maximum(b['oat_plus'], b['oat_minus'])[o], 'Exact one-at-a-time (worse side)')):
        lim = np.abs(M).max()
        fig.colorbar(a.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto'), ax=a)
        a.set_xticks(xb, SHORT, rotation=45, ha='right', fontsize=8)
        a.set_title(title, fontsize=10)
    ax[0].set_yticks(np.arange(D), [labels[k] for k in o], fontsize=7)
    fig.suptitle(f'Sensitivity per parameter and muon region ({var})')
    save(fig, f'{var}_bins_sensitivity.png')

    COS = b['cos']
    fig, ax = plt.subplots(figsize=(1.6 + 0.8 * B, 1.2 + 0.8 * B), constrained_layout=True)
    fig.colorbar(ax.imshow(COS, cmap='RdBu_r', vmin=-1, vmax=1), ax=ax, label='cosine')
    for k in range(B):
        for j in range(B):
            ax.text(j, k, f'{COS[k, j]:+.2f}', ha='center', va='center', fontsize=7, color='k' if abs(COS[k, j]) < 0.6 else 'w')
    ax.set_xticks(xb, SHORT, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(xb, SHORT, fontsize=8)
    ax.set_title('Alignment of the regions\' own worst directions')
    save(fig, f'{var}_bins_alignment.png')

    SIM = bool(b['simulated'])
    NAMES = [str(s) for s in b['dir_names']]
    nr = int(np.ceil(len(NAMES) / 3))
    fig, axes = plt.subplots(nr, 3, figsize=(14.4, 4.0 * nr), constrained_layout=True, squeeze=False)
    for a in axes.ravel()[len(NAMES):]:
        a.axis('off')
    pred = b['pred'] if SIM else b['pred_cache'] * b['H_0_cache']
    for a, nm, idx in zip(axes.ravel(), NAMES, b['dir_idx']):
        for k in range(B):
            a.plot(b['t'], pred[idx, k] / max(pred[0, k], 1e-30), '-o', ms=3, color=COL[k], lw=1.5, label=SHORT[k])
            if SIM:
                rt, se = poisson_ratio(b['sim'][idx, k], max(b['sim'][0, k], 1))
                a.errorbar(b['t'], rt, yerr=se, fmt='s', ms=4, color=COL[k], capsize=2, lw=1, alpha=0.85, label='_')
        a.plot(b['t'], pred[idx].sum(1) / pred[0].sum(), '--', color='k', lw=2, label='total')
        a.axvline(0, color=GREY, lw=1, ls='--')
        a.axhline(1, color=GREY, lw=0.8, ls=':')
        a.set_yscale('log')
        a.set(title=nm, xlabel='step $t$ along $v$ (normalized units)', ylabel='$H_b(t)\\,/\\,H_b(0)$')
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f'Response by muon region along each direction ({var}); lines: surrogate'
                 + (', squares: simulator' if SIM else ''))
    save(fig, f'{var}_bins_walks.png')

    mc = b['mc']
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    edges = np.logspace(np.log10(max(mc.min(), 1e-8) * 0.9), np.log10(mc.max() * 1.2), 50)
    for k in range(B):
        ax.hist(np.clip(mc[:, k], edges[0], edges[-1]), bins=edges, histtype='step', lw=1.6, color=COL[k], label=SHORT[k])
    ax.axvline(1.0, color='k', lw=1.2, ls='--', label='nominal')
    ax.set(xscale='log', xlabel='$H_b/H_b(0)$', ylabel='designs', title=f'{len(mc):,} designs uniform in the $\\pm$10% box, by region')
    ax.legend(fontsize=8, ncol=2)
    save(fig, f'{var}_bins_mc.png')


# =============================================================================
# tolerance_boxes.npz (+ _eval.npz)
# =============================================================================
def tolerance_figures(t, ev):
    names, labels = [str(s) for s in t['names']], list(t['labels'])
    pred, pred_se, sim, S0 = ev['pred'], ev['pred_se'], ev['sim'], float(ev['sim0'])
    SIM = np.isfinite(sim)

    # --- the worst design in each box against the scaled worst direction
    pcts = t['pcts']
    bi = [names.index(f'box{x:g}%') for x in pcts]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.errorbar(pcts, pred[bi], yerr=pred_se[bi], fmt='o-', color=OK, ms=6, capsize=3, label='box worst case: surrogate')
    if SIM[bi].any():
        sb = [i for i in bi if SIM[i]]
        sr, sr_se = poisson_ratio(sim[sb], S0)
        ax.errorbar(pcts[SIM[bi]], sr, yerr=sr_se, fmt='s', color=BAD, ms=7, capsize=3, label='box worst case: simulator')
        for x, s in zip(pcts[SIM[bi]], sr):
            ax.annotate(f'x{s:.3g}', (x, s), textcoords='offset points', xytext=(6, -12), fontsize=8)
    if len(ev['wc_pcts']):
        ax.plot(ev['wc_pcts'], ev['wc'][:, 0], 'o--', color=OK, mfc='white', ms=6, label='scaled worst direction: surrogate')
        ax.plot(ev['wc_pcts'], ev['wc'][:, 1], 's--', color=BAD, mfc='white', ms=6, label='scaled worst direction: simulator')
    ax.axhline(1, color='k', lw=0.8, ls=':')
    ax.set(yscale='log', xlabel='tolerance X (every parameter within $\\pm$X%)', ylabel='worst $H/H_0$',
           title='Worst case for a given tolerance')
    ax.legend(fontsize=8)
    save(fig, 'tolerance_boxes.png')

    # --- the allocations: per-parameter tolerance for each target, and the greedy path
    alloc = [nm for nm in names if nm.startswith('alloc')]
    if not alloc:
        return
    path_e, path_h = t['path_e'], np.exp(t['path_h'])
    steps = [int(np.argmax(path_h <= float(nm[5:]))) for nm in alloc]
    tight = np.flatnonzero(path_e[max(steps)] < 1)
    tight = tight[np.argsort(path_e[max(steps)][tight])]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 0.28 * len(tight) + 2.5), constrained_layout=True,
                                 gridspec_kw=dict(width_ratios=[2, 1]))
    y = np.arange(len(tight))[::-1]
    h = 0.8 / len(alloc)
    cmap = plt.get_cmap('viridis')
    for j, (nm, st) in enumerate(zip(alloc, steps)):
        i = names.index(nm)
        lab = f'worst $\\leq$ {nm[5:]} $H_0$ (surr. {pred[i]:.2f}' + (f', sim. {sim[i] / S0:.2f})' if SIM[i] else ')')
        ax.barh(y + 0.4 - (j + 0.5) * h, 10 * path_e[st][tight], h, color=cmap(j / max(1, len(alloc) - 1)), label=lab)
    ax.axvline(10, color='k', lw=0.8, ls=':')
    ax.set_yticks(y, [labels[k] for k in tight], fontsize=8)
    ax.set(xlabel='allowed tolerance ($\\pm$%); every other parameter $\\pm$10%', xlim=(0, 10.5),
           title='Tolerance allocation (greedy, surrogate)')
    ax.legend(fontsize=8, loc='lower right')
    bx.plot((path_e < 1).sum(1), path_h, '-', color=OK)
    for j, (nm, st) in enumerate(zip(alloc, steps)):
        bx.plot((path_e[st] < 1).sum(), path_h[st], 'o', color=cmap(j / max(1, len(alloc) - 1)), ms=7)
    bx.set(yscale='log', xlabel='parameters tightened', ylabel='worst $H/H_0$ in the box (surrogate)',
           title='Greedy path')
    save(fig, 'tolerance_allocation.png')


path = os.path.join(args.run, '{}.npz')
if os.path.exists(path.format('robustness')):
    robustness_figures(np.load(path.format('robustness')))
if os.path.exists(path.format('worst_cases')):
    worst_case_figures(np.load(path.format('worst_cases')))
for var in ('pz', 'pt', 'p', 'charge'):
    if os.path.exists(path.format(f'{var}_bins')):
        bin_figures(np.load(path.format(f'{var}_bins')))
if os.path.exists(path.format('tolerance_boxes_eval')):
    tolerance_figures(np.load(path.format('tolerance_boxes')), np.load(path.format('tolerance_boxes_eval')))
