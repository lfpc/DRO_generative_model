"""The previous surrogate against the current one, on the same axes, from their saved results
(no GPU, no simulation).

    hits_<ref>.png         predicted vs. simulated hits of the reference designs (validation/validate.py),
                           old | new side by side; the same designs and muons for both
    calibration_<ref>.png  per muon, simulated hit rate against predicted p(hit), both models overlaid
    walked_designs.png     surrogate vs. simulator Y/Y_0 of every design the analysis walked and
                           simulated (analysis/robustness.py), old | new; each model walked its own designs

    python -m validation.compare
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cli
from paths import FIGURES, RESULTS, save_figure
from validation import references
from validation.validate import REFS, merge_bins

parser = cli.parser(__doc__)
parser.add_argument('--old', default='v1', help='suffix of the previous results: results/{validation,robustness}_<old>')
parser.add_argument('--old_label', default='previous surrogate (lcso_v1)')
parser.add_argument('--new_label', default='current surrogate (lcso)')
parser.add_argument('--refs', default=','.join(REFS))
parser.add_argument('--out', default=os.path.join(FIGURES, 'comparison'))
args = parser.parse_args()

plt.rcParams.update({'font.size': 11})
OLD, NEW, GREY = '#7f7f7f', '#e34948', '0.45'
pred = {k: np.load(os.path.join(RESULTS, d, 'predictions.npz'))
        for k, d in (('old', f'validation_{args.old}'), ('new', 'validation'))}
labels = {'old': args.old_label, 'new': args.new_label}


def save(fig, name):
    save_figure(fig, args.out, name, dpi=170)
    plt.close(fig)
    print(f'saved {args.out}/{name}.png')


def mad(r):
    return np.median(np.abs(np.log(r)))


for tag in args.refs.split(','):
    names, U, sim, n, _ = references.load(tag)
    U = np.asarray(U)
    dist = np.linalg.norm(U, axis=1)
    se = np.sqrt(sim)
    o = np.argsort(-dist)
    ratios = {k: pred[k][f'pred_{tag}'] / sim for k in pred}
    lim = np.array([sim.min() * 0.75, sim.max() * 1.3])
    all_r = np.concatenate(list(ratios.values()))
    rlim = (min(0.85, all_r.min() * 0.9), max(1.15, all_r.max() * 1.1))
    box = dist > 0.75 * dist.max()

    # --- predicted vs. simulated hits, old | new
    fig = plt.figure(figsize=(12, 6.2), layout='constrained')
    axs = fig.subplots(2, 2, sharex=True, sharey='row', height_ratios=(3, 1))
    norm = matplotlib.colors.Normalize(0, dist.max())
    for j, k in enumerate(('old', 'new')):
        ax, rx, r = axs[0, j], axs[1, j], ratios[k]
        ax.fill_between(lim, lim * 0.9, lim * 1.1, color='0.93', lw=0, zorder=0, label=r'$\pm10\%$')
        ax.plot(lim, lim, color=GREY, lw=1, zorder=1)
        rx.axhspan(0.9, 1.1, color='0.93', lw=0, zorder=0)
        rx.axhline(1, color=GREY, lw=1, zorder=1)
        for a, y in ((ax, r * sim), (rx, r)):
            a.errorbar(sim, y, xerr=se, fmt='none', ecolor='0.7', elinewidth=0.7, zorder=2)
            sc = a.scatter(sim[o], y[o], c=dist[o], cmap='viridis', norm=norm, s=30,
                           edgecolors='white', linewidths=0.6, zorder=3)
        ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim, title=labels[k])
        rx.set(yscale='log', ylim=rlim, xlabel='simulated hits')
        ax.text(0.04, 0.96, f'{REFS.get(tag, tag)}, {len(sim)} designs\n'
                f'median |log pred/sim|: all {mad(r):.3f}, outer quarter {mad(r[box]):.3f}\n'
                f'nominal: pred/sim = {np.mean(r[dist == 0]):.3f}',
                transform=ax.transAxes, va='top', fontsize=9.5)
        ax.legend(loc='lower right', fontsize=9, frameon=False)
    axs[0, 0].set_ylabel('predicted hits')
    axs[1, 0].set_ylabel('pred. / sim.')
    for rx in axs[1]:
        rx.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        rx.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        rx.set_yticks([t for t in (0.3, 0.5, 0.7, 1, 1.4, 2, 3) if rlim[0] <= t <= rlim[1]])
    fig.colorbar(sc, ax=axs, shrink=0.8, label=r'distance from nominal $\|u\|_2$  ($u_i = \pm1$: $\pm10\%$)')
    save(fig, f'hits_{tag}')

    # --- per-muon calibration, both overlaid
    fig = plt.figure(figsize=(6.6, 6.2), layout='constrained')
    ax, rx = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    curves = {k: merge_bins(*pred[k][f'cal_{tag}']) for k in pred}
    pmin = min(c[0].min() for c in curves.values())
    lim2 = np.array([pmin / 2, 1])
    ax.plot(lim2, lim2, color=GREY, lw=1)
    rx.axhline(1, color=GREY, lw=1)
    for k, c in (('old', OLD), ('new', NEW)):
        p, rate, err = curves[k]
        ax.errorbar(p, rate, yerr=err, fmt='o', ms=4, color=c, mec='white', mew=0.5, elinewidth=1, label=labels[k])
        rx.errorbar(p, rate / p, yerr=err / p, fmt='o', ms=4, color=c, mec='white', mew=0.5, elinewidth=1)
    ax.set(xscale='log', yscale='log', xlim=lim2, ylim=lim2, ylabel='simulated hit rate',
           title=f'{REFS.get(tag, tag)}, {len(sim)} designs')
    rx.set(xlabel='predicted $p$(hit) of a muon', ylabel='sim. / pred.')
    ax.legend(loc='lower right', fontsize=9)
    save(fig, f'calibration_{tag}')

# --- every walked and simulated design of the analysis, old | new
runs = {'old': np.load(os.path.join(RESULTS, f'robustness_{args.old}', 'robustness.npz'), allow_pickle=True),
        'new': np.load(os.path.join(RESULTS, 'robustness', 'robustness.npz'), allow_pickle=True)}
pts = {}
for k, r in runs.items():
    sim = r['sim']
    rows = []
    for j, (nm, idx, ts) in enumerate(zip(r['dir_names'], r['dir_idx'], r['dir_t'])):
        for i, t in zip(idx, ts):
            if t != 0 and sim[i] > 0:
                s = sim[i] / sim[0]
                rows.append((j, s, s * np.sqrt(1 / sim[i] + 1 / sim[0]), r['ratio'][i], r['ratio_se'][i]))
    pts[k] = np.array(rows)
v = np.concatenate([p[:, [1, 3]].ravel() for p in pts.values()])
lo, hi = 0.5 * v.min(), 2 * v.max()
fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), layout='constrained', sharex=True, sharey=True)
names = [str(s) for s in runs['new']['dir_names']]
for ax, k in zip(axs, ('old', 'new')):
    p, r = pts[k], runs[k]
    ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2], color='0.93', lw=0, zorder=0, label='within $\\times2$')
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
    for j, nm in enumerate(names):
        q = p[p[:, 0] == j]
        if len(q):
            ax.errorbar(q[:, 1], q[:, 3], xerr=q[:, 2], yerr=q[:, 4], fmt='o', ms=5, capsize=2,
                        color=plt.cm.tab10(j % 10), label=nm)
    lr = np.abs(np.log(p[:, 3] / p[:, 1]))
    ax.text(0.04, 0.96, f'{len(p)} designs; median |log surr./sim.| {np.median(lr):.3f}\n'
            f'within $\\times2$: {(lr < np.log(2)).sum()}/{len(p)}\n'
            f'nominal: surrogate {r["pred"][0]:.0f}, simulator {r["sim"][0]:.0f} hits',
            transform=ax.transAxes, va='top', fontsize=9.5)
    ax.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo, hi), xlabel='simulator $Y/Y_0$', title=labels[k])
axs[0].set_ylabel('surrogate $Y/Y_0$')
axs[1].legend(fontsize=8, loc='lower right')
save(fig, 'walked_designs')
