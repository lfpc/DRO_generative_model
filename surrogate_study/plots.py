"""Every figure and summary table of the surrogate study, from the saved results.

    python plots.py            # -> outputs/surrogate_study/figs/*.png, *_summary.txt

Each curve is the median over seeds, the band the seed range; one setting varies, the others
stay at the defaults of run_toys.py / run_shield.py.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

import run_shield
import run_toys
import shield_ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'outputs/surrogate_study/figs')
# categorical slots in fixed order; the baselines are grey
COLOR = {'deeponet': '#2a78d6', 'taylor': '#eb6834', 'taylor:8': '#1baf7a', 'taylor-sub:4': '#eda100',
         'taylor-res': '#e87ba4', 'quad': '#008300', 'rsm': '#6250d6', 'fd': '0.45', 'existing': 'k',
         'final': '#e34948'}
LABEL = {'deeponet': 'DeepONet', 'taylor': 'Taylor, full rank', 'taylor:8': 'Taylor, rank 8',
         'taylor-sub:4': 'Taylor on a 4-d subspace', 'taylor-res': 'Taylor + residual',
         'quad': 'full quadratic', 'rsm': 'response surface (no x)', 'fd': 'finite differences',
         'existing': 'lcso_deeponet.pt (earlier)', 'final': 'lcso.pt (final: 4 x 128/32, all pools)'}
TOY_METRICS = [('pred_box', r'$H$ in the box: median $|\log \hat H / H|$', 'log'),
               ('grad_err', r'gradient: $|\hat g - g| / |g|$', 'log'),
               ('hess_err', r'Hessian: $|\hat H - H|_F / |H|_F$', 'log'),
               ('worst', 'worst case found (true rise / max)', 'linear')]
PROBLEMS = list(run_toys.toys.PROBLEMS)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5})


def matching(rows, target, free):
    """Rows whose config equals target on every key except `free` and the seed."""
    keys = [k for k in target if k not in (free, 'seed')]
    return [r for r in rows if all(r['config'].get(k) == target[k] for k in keys)]


def curve(rows, defaults, free, metric, **over):
    """(x, median, min, max) of metric as `free` varies, the rest at defaults + over."""
    sel = matching(rows, dict(defaults, **over), free)
    xs = sorted({r['config'][free] for r in sel if metric in r and np.isfinite(r[metric])}, key=lambda v: (str(type(v)), v))
    stats = [np.array([r[metric] for r in sel if r['config'][free] == x and metric in r]) for x in xs]
    return (np.array(xs, dtype=object), np.array([np.median(s) for s in stats]),
            np.array([s.min() for s in stats]), np.array([s.max() for s in stats]))


def draw(ax, x, med, lo, hi, color, label=None, ls='-', marker='o'):
    if not len(x):
        return
    x = np.asarray(x, dtype=float)
    ax.plot(x, med, ls, color=color, lw=2, marker=marker, ms=4, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)


def grid(n_rows, n_cols, w=3.2, h=2.5):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w * n_cols, h * n_rows), constrained_layout=True, squeeze=False)
    return fig, axes


def save(fig, name):
    for ax in fig.axes:                              # no labels on minor log ticks: they collide
        if ax.get_xscale() == 'log':
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 3.0) if name.startswith(('toy_delta', 'toy_latent')) else (1.0,)))
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name), dpi=150)
    plt.close(fig)
    print(f'  {name}')


def finish(axes, metrics, problems, xlabel, xscale='log'):
    for i, (m, ylabel, yscale) in enumerate(metrics):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            if ax.has_data():                    # (a partial run may leave a panel empty)
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
            if m == 'worst':
                ax.set_ylim(0, 1.08)
                ax.axhline(1, color='0.7', lw=0.8, zorder=0)
            if i == 0 and problems:
                ax.set_title(problems[j])
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(metrics) - 1:
                ax.set_xlabel(xlabel)


# ---------------------------------------------------------------------------
# Toys
# ---------------------------------------------------------------------------
def renormalize_worst(rows):
    """'worst' relative to the largest true rise any search found for that problem: on two
    moons the multi-start search of the truth misses the global maximum by a few percent."""
    top = {}
    for r in rows:
        if 'worst' in r:
            p = r['config']['problem']
            top[p] = max(top.get(p, 1.0), r['worst'])
    for r in rows:
        if 'worst' in r:
            r['worst'] /= top[r['config']['problem']]
    return top


def toy_figures(rows):
    D = dict(run_toys.DEFAULTS)

    def per_problem(name, free, kinds, xlabel, metrics=TOY_METRICS, xscale='log', over=None, styles=None):
        fig, axes = grid(len(metrics), len(PROBLEMS))
        for j, prob in enumerate(PROBLEMS):
            for i, (m, _, _) in enumerate(metrics):
                for k in kinds:
                    for tag, extra, ls in (styles or [('', {}, '-')]):
                        x, med, lo, hi = curve(rows, D, free, m, problem=prob, kind=k, **(over or {}), **extra)
                        if len(x):
                            draw(axes[i, j], x, med, lo, hi, COLOR[k], f'{LABEL[k]}{tag}', ls)
        finish(axes, metrics, PROBLEMS, xlabel, xscale)
        handles = {l: h for ax in axes.flat for h, l in zip(*ax.get_legend_handles_labels())}
        axes[0, 0].legend(handles.values(), handles.keys(), fontsize=7)
        return fig, axes

    fig, _ = per_problem('budget', 'budget', run_toys.KINDS, 'simulated samples (budget)')
    fig.suptitle('Convergence with the simulation budget (1000 samples per design, region = the box)')
    save(fig, 'toy_budget.png')

    fig, _ = per_problem('scheme', 'budget', ['deeponet', 'taylor'], 'simulated samples (budget)', TOY_METRICS[:3],
                         styles=[(' (fixed)', dict(scheme='fixed'), '-'), (' (stream)', dict(scheme='stream'), '--')])
    fig.suptitle('All designs at once (fixed) vs. a rolling buffer of 64 designs (stream), same data and steps')
    save(fig, 'toy_scheme.png')

    fig, _ = per_problem('n', 'n', ['deeponet', 'taylor', 'taylor:8'], 'samples per design (budget 1e6 fixed)')
    fig.suptitle('Many designs with few samples, or few designs with many: budget fixed at 1e6')
    save(fig, 'toy_n.png')

    fig, axes = per_problem('delta', 'delta', ['deeponet', 'taylor', 'taylor:8', 'quad'],
                            r'design region half-width $\delta$ (1 = the tolerance box)')
    for j, prob in enumerate(PROBLEMS):
        for i, (m, _, _) in enumerate(TOY_METRICS):
            for k in ('deeponet', 'taylor'):
                sel = [r[m] for r in matching(rows, dict(D, problem=prob, kind=k, radial=False), None) if m in r]
                if sel:
                    axes[i, j].plot([1.0], [np.median(sel)], 'D', mfc='white', mec=COLOR[k], ms=7, mew=1.5,
                                    label=f'{LABEL[k]}, no radial pull' if i == j == 0 else None)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(r'Size of the region the training designs are drawn from (worst case: ball $|u| \leq 1$)')
    save(fig, 'toy_delta.png')

    # DeepONet shape: width x depth heatmaps, then the latent size and the activation
    widths, depths = (16, 32, 64, 128, 256), (1, 2, 3, 4)
    fig, axes = grid(2, len(PROBLEMS), 3.2, 2.8)
    for j, prob in enumerate(PROBLEMS):
        for i, m in enumerate(('grad_err', 'hess_err')):
            Z = np.full((len(depths), len(widths)), np.nan)
            for a, d in enumerate(depths):
                for b, w in enumerate(widths):
                    sel = [r[m] for r in matching(rows, dict(D, problem=prob, hidden=w, depth=d), None)]
                    Z[a, b] = np.median(sel) if sel else np.nan
            ax = axes[i, j]
            ax.grid(False)
            ax.imshow(np.log10(Z), cmap='Blues', origin='lower', aspect='auto')
            for a in range(len(depths)):
                for b in range(len(widths)):
                    if np.isfinite(Z[a, b]):
                        ax.text(b, a, f'{Z[a, b]:.2f}', ha='center', va='center', fontsize=7,
                                color='white' if np.log10(Z[a, b]) > np.nanmean(np.log10(Z)) else '0.15')
            ax.set_xticks(range(len(widths)), widths)
            ax.set_yticks(range(len(depths)), depths)
            ax.set_title(f'{prob}: {"gradient" if i == 0 else "Hessian"} error', fontsize=8)
            if i == 1:
                ax.set_xlabel('width (hidden units)')
            if j == 0:
                ax.set_ylabel('depth (hidden layers)')
    fig.suptitle('DeepONet width x depth (p = 16, budget 1e6): relative errors at u = 0, median of 3 seeds')
    save(fig, 'toy_shape.png')

    acts = ('relu', 'silu', 'gelu', 'tanh', 'softplus')
    fig, axes = grid(2, len(PROBLEMS))
    for j, prob in enumerate(PROBLEMS):
        for m, c in (('grad_err', '#2a78d6'), ('hess_err', '#eb6834')):
            x, med, lo, hi = curve(rows, D, 'p', m, problem=prob, kind='deeponet')
            draw(axes[0, j], x, med, lo, hi, c, 'gradient' if m == 'grad_err' else 'Hessian')
            meds = [np.median([r[m] for r in matching(rows, dict(D, problem=prob, act=a), None)] or [np.nan]) for a in acts]
            off = -0.2 if m == 'grad_err' else 0.2
            if np.isfinite(meds).any():
                axes[1, j].bar(np.arange(len(acts)) + off, meds, 0.38, color=c, label='gradient' if m == 'grad_err' else 'Hessian')
                axes[1, j].set_yscale('log')
        if axes[0, j].has_data():
            axes[0, j].set(xscale='log', yscale='log')
        axes[0, j].set(title=prob, xlabel='latent size p')
        axes[1, j].set(xticks=range(len(acts)), xticklabels=acts, xlabel='branch activation')
    axes[0, 0].set_ylabel('relative error at u = 0')
    axes[1, 0].set_ylabel('relative error at u = 0')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('DeepONet latent size p and branch activation (ReLU has no curvature of its own)')
    save(fig, 'toy_latent_act.png')

    fig, axes = grid(len(TOY_METRICS), len(PROBLEMS))
    for j, prob in enumerate(PROBLEMS):
        P = run_toys.toys.load(prob).D
        for i, (m, _, _) in enumerate(TOY_METRICS):
            for base, col in (('taylor', COLOR['taylor']), ('taylor-sub', COLOR['taylor-sub:4'])):
                pts = []
                for r_ in (1, 2, 4, 8, 16, P):
                    k = f'{base}:{r_}' if r_ != P else base
                    sel = [r[m] for r in matching(rows, dict(D, problem=prob, kind=k), None) if m in r]
                    if r_ <= P and sel:
                        pts.append((min(r_, P), np.median(sel), min(sel), max(sel)))
                if pts:
                    pts = np.array(sorted(set(pts)))
                    draw(axes[i, j], pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], col,
                         'Taylor: full gradient, rank-r curvature' if base == 'taylor' else 'Taylor on an r-d subspace')
    finish(axes, TOY_METRICS, PROBLEMS, 'rank r (rightmost: r = D)')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Rank of the Taylor surrogates (synthetic: the truth acts through 4 directions + a weak full part)')
    save(fig, 'toy_rank.png')


def toy_summary(rows):
    D = dict(run_toys.DEFAULTS)
    cols = ['h0_err', 'pred_in', 'pred_box', 'grad_err', 'grad_cos', 'hess_err', 'eig_cos', 'worst', 'steps', 'seconds']
    lines = [f'Toy problems at the defaults ({json.dumps({k: v for k, v in D.items() if k != "seed"})}),'
             ' budgets 1e6 and 1e7; median over seeds']
    for prob in PROBLEMS:
        lines.append(f'\n{prob}')
        lines.append(f'  {"surrogate":26s} {"budget":>7s} ' + ' '.join(f'{c:>9s}' for c in cols))
        for b in (1_000_000, 10_000_000):
            for k in run_toys.KINDS:
                sel = matching(rows, dict(D, problem=prob, kind=k, budget=b), None)
                if sel:
                    vals = [np.median([r[c] for r in sel]) if c in sel[0] else np.nan for c in cols]
                    lines.append(f'  {LABEL[k]:26s} {b:7.0e} ' + ' '.join(f'{v:9.3g}' for v in vals))
    with open(os.path.join(FIGS, 'toys_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('  toys_summary.txt')


# ---------------------------------------------------------------------------
# Muon shield
# ---------------------------------------------------------------------------
def noise_floors(refs):
    """What the metrics read for a perfect model, from the reference's own Poisson noise
    (expected values under Gaussian noise of the simulated counts)."""
    out, rng = {}, np.random.default_rng(0)
    for tag, (names, U, hits, n, _) in refs.items():
        names = np.array(names)
        for grp in ('half', 'box'):
            h = hits[np.char.startswith(names, grp)]
            out[f'pred_{grp}_{tag}'] = float(np.median(np.abs(np.log1p(rng.standard_normal((2000, len(h))) / np.sqrt(h)))))
        if any(n_.startswith('oat') for n_ in names):
            idx = {n_: i for i, n_ in enumerate(names)}
            Dm = U.shape[1]
            hp = np.array([hits[idx[f'oat+{i}']] for i in range(Dm)])
            hm = np.array([hits[idx[f'oat-{i}']] for i in range(Dm)])
            h0 = hits[np.char.startswith(names, 'nominal')]
            s0, v0 = h0.mean(), h0.mean() / len(h0)
            out['oat_slope_err'] = float(np.sqrt(((hp + hm) / 4).sum()) / np.linalg.norm((hp - hm) / 2))
            out['oat_curv_err'] = float(np.sqrt((hp + hm + 4 * v0).sum()) / np.linalg.norm(hp + hm - 2 * s0))
    return out


SHIELD_METRICS = [('pred_box_T1', 'T = 1: median $|\\log$ pred / sim$|$ in the box', 'log'),
                  ('pred_box_T1.5', 'T = 1.5: median $|\\log$ pred / sim$|$ in the box', 'log'),
                  ('oat_slope_err', 'one-at-a-time secants: rel. error', 'log'),
                  ('oat_curv_err', 'one-at-a-time curvatures: rel. error', 'log')]


def shield_figures(rows, refs):
    D = dict(run_shield.DEFAULTS)
    floor = noise_floors(refs)
    existing = next((r for r in rows if r['config']['kind'] == 'existing'), None)
    final = next((r for r in rows if r['config']['kind'] == 'final'), None)

    def reference_lines(ax, m):
        for r, ls in ((existing, '--'), (final, '-')):
            if r and m in r:
                ax.axhline(r[m], color=COLOR[r['config']['kind']], ls=ls, lw=1.4, label=LABEL[r['config']['kind']])
        if m in floor:
            ax.axhline(floor[m], color='0.55', ls=':', lw=1.2, label='simulation noise')

    fig, axes = grid(1, len(SHIELD_METRICS), 3.4, 3.0)
    for j, (m, ylabel, yscale) in enumerate(SHIELD_METRICS):
        ax = axes[0, j]
        for k in run_shield.KINDS:
            x, med, lo, hi = curve(rows, D, 'budget', m, kind=k)
            if len(x):
                draw(ax, x, med, lo, hi, COLOR[k], LABEL[k])
        reference_lines(ax, m)
        ax.set(xscale='log', yscale=yscale, xlabel='simulated muons at T = 1.5 (1e7 per design)', title=ylabel)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Muon shield: convergence with the simulation budget, scored on the reference designs')
    save(fig, 'shield_budget.png')

    # convergence of the derivatives at u = 0: seed-to-seed agreement, and agreement with the
    # largest-budget DeepONet
    arrays = {r['key']: np.load(os.path.join(run_shield.OUT, 'arrays', f'{r["key"]}.npz')) for r in rows}
    top = [r for r in matching(rows, dict(D, kind='deeponet', budget=max(run_shield.BUDGETS)), None)]
    g_ref = np.mean([arrays[r['key']]['g'] for r in top], 0) if top else None
    fig, axes = grid(1, 3, 3.6, 3.0)
    for k in run_shield.KINDS:
        xs, sg, sh, cg = [], [], [], []
        for b in run_shield.BUDGETS:
            sel = matching(rows, dict(D, kind=k, budget=b), None)
            if len(sel) < 2:
                continue
            g = [arrays[r['key']]['g'] for r in sel]
            H = [arrays[r['key']]['H'] for r in sel]
            xs.append(b)
            sg.append(np.linalg.norm(g[0] - g[1]) / np.linalg.norm(np.mean(g, 0)))
            sh.append(np.linalg.norm(H[0] - H[1]) / np.linalg.norm(np.mean(H, 0)))
            gm = np.mean(g, 0)
            cg.append(gm @ g_ref / np.linalg.norm(gm) / np.linalg.norm(g_ref) if g_ref is not None else np.nan)
        if xs:
            for ax, y in zip(axes[0], (sg, sh, cg)):
                ax.plot(xs, y, '-o', color=COLOR[k], lw=2, ms=4, label=LABEL[k])
    axes[0, 0].set(xscale='log', yscale='log', title='gradient at u = 0: seed 0 vs seed 1', ylabel='relative difference')
    axes[0, 1].set(xscale='log', yscale='log', title='Hessian at u = 0: seed 0 vs seed 1')
    axes[0, 2].set(xscale='log', title='cos(gradient, largest DeepONet)', ylim=(0, 1.02))
    for ax in axes[0]:
        ax.set_xlabel('simulated muons at T = 1.5')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Muon shield: do the gradient and Hessian at u = 0 converge? (on the 1e9 reference muons at T = 1)')
    save(fig, 'shield_derivatives.png')

    # one setting at a time: (setting, panel title, fixed overrides, series [(label, color, kind or None)])
    rank_kinds = [f'taylor:{r}' for r in (2, 4, 8, 16)] + ['taylor']
    sub_kinds = [f'taylor-sub:{r}' for r in (2, 4, 8, 16)]
    two = [(LABEL['deeponet'], COLOR['deeponet'], 'deeponet'), (LABEL['taylor'], COLOR['taylor'], 'taylor')]
    settings = [('n', 'muons per design (budget 2e9)', dict(budget=2_000_000_000), two),
                ('pool', 'pool: T and region (budget 3e9)', dict(budget=3_000_000_000), two),
                ('keep_neg', 'fraction of misses kept', {}, two),
                ('scheme', 'fixed vs. stream (budget 1.8e10)', dict(budget=max(run_shield.BUDGETS)), two),
                ('pool', 'hot pools, 1e6 muons/design (9e9)', dict(n=1_000_000, budget=9_000_000_000),
                 [(LABEL[k], COLOR[k], k) for k in ('deeponet', 'taylor-sub:4', 'taylor:8')]),
                ('kind', 'Taylor rank r (r = 43: full)', {}, [('Taylor, rank r', COLOR['taylor'], rank_kinds),
                                                            ('Taylor, r-d subspace', COLOR['taylor-sub:4'], sub_kinds)]),
                ('p', 'DeepONet latent p', {}, two[:1]), ('act', 'branch activation', {}, two[:1]),
                ('hidden', 'DeepONet width (depth 2)', {}, two[:1])]

    def xval(free, v):
        return int(v.split(':')[1]) if free == 'kind' and ':' in v else (43 if free == 'kind' else v)

    def series_rows(free, over, kind):
        if free == 'kind':                       # the series is a list of kinds, at the defaults
            return [r for r in rows if r['config']['kind'] in kind
                    and all(r['config'].get(k) == v for k, v in D.items() if k not in ('kind', 'seed'))]
        return matching(rows, dict(D, kind=kind, **over), free)

    fig, axes = grid(2, len(settings), 2.6, 2.8)
    for j, (free, title, over, series) in enumerate(settings):
        for i, m in enumerate(('pred_box_T1.5', 'oat_slope_err')):
            ax = axes[i, j]
            sels = [series_rows(free, over, kind) for _, _, kind in series]
            allv = {xval(free, r['config'][free]) for sel in sels if len({r['config'][free] for r in sel}) > 1 for r in sel}
            if not allv:
                continue
            order = sorted(allv, key=lambda q: (isinstance(q, str), q))
            pos = {q: n for n, q in enumerate(order)}
            for si, ((lab, col, kind), sel) in enumerate(zip(series, sels)):
                vals = sorted({r['config'][free] for r in sel}, key=lambda v: (isinstance(v, str), xval(free, v)))
                if len(vals) < 2:
                    continue
                off = (si - (len(series) - 1) / 2) * 0.25
                xs = [pos[xval(free, v)] + off for v in vals]
                ys = [[r[m] for r in sel if r['config'][free] == v] for v in vals]
                ax.plot(xs, [np.median(y) for y in ys], 'o', color=col, ms=6, label=lab)
                ax.vlines(xs, [min(y) for y in ys], [max(y) for y in ys], color=col, lw=1.5)
            ax.set_xticks(range(len(order)), [f'{q:g}' if isinstance(q, (int, float)) else q.replace('_', ' ') for q in order],
                          rotation=30, fontsize=7)
            ax.set_yscale('log')
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(['T = 1.5 box prediction error', 'one-at-a-time secant error'][i])
            if i == 0 and len(series) > 1 and (j == 0 or free == 'kind' or len(series) > 2):
                ax.legend(fontsize=6.5)
    fig.suptitle('Muon shield: one training setting at a time (budget 4e9 unless labelled; dots: median of 2 seeds, bars: range)')
    save(fig, 'shield_settings.png')

    # calibration on the reference designs
    best = {k: min(matching(rows, dict(D, kind=k, budget=max(run_shield.BUDGETS)), None) or [None],
                   key=lambda r: r['pred_box_T1.5'] if r else 0) for k in ('deeponet', 'taylor')}
    fig, axes = grid(1, 2, 4.2, 3.9)
    for j, tag in enumerate(('T1', 'T1.5')):
        names, U, hits, n, _ = refs[tag]
        ax = axes[0, j]
        for k, r in [('deeponet', best['deeponet']), ('existing', existing), ('final', final)]:
            if r is None:
                continue
            pred = np.load(os.path.join(run_shield.OUT, 'arrays', f'{r["key"]}.npz'))[f'pred_{tag}']
            ax.errorbar(hits, pred, xerr=np.sqrt(hits), fmt='o', ms=3.5, color=COLOR[k], elinewidth=0.6,
                        label=LABEL[k], alpha=0.85)
        lim = [hits.min() * 0.8, hits.max() * 1.2]
        ax.plot(lim, lim, color='0.6', lw=1, zorder=0)
        ax.set(xscale='log', yscale='log', xlabel='simulated hits', ylabel='predicted hits',
               title=f'reference at {tag.replace("T", "T = ")} ({n:.0e} muons)')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Muon shield: predicted vs. simulated hits on the reference designs, same muons')
    save(fig, 'shield_calibration.png')

    # one-at-a-time secants of the 43 parameters
    names, U, hits, n, _ = refs['T1.5']
    idx = {n_: i for i, n_ in enumerate(names)}
    Dm = U.shape[1]
    sim = np.array([(hits[idx[f'oat+{i}']] - hits[idx[f'oat-{i}']]) / 2 for i in range(Dm)])
    err = np.array([np.sqrt(hits[idx[f'oat+{i}']] + hits[idx[f'oat-{i}']]) / 2 for i in range(Dm)])
    order = np.argsort(-np.abs(sim))
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    ax.errorbar(np.arange(Dm), sim[order], yerr=err[order], fmt='s', color='0.25', ms=4, label='simulator (1 s.d.)')
    from shield import MuonShield
    labels = MuonShield().labels
    for k, r in [('deeponet', best['deeponet']), ('existing', existing), ('final', final)]:
        if r is None:
            continue
        pred = np.load(os.path.join(run_shield.OUT, 'arrays', f'{r["key"]}.npz'))['pred_T1.5']
        s = np.array([(pred[idx[f'oat+{i}']] - pred[idx[f'oat-{i}']]) / 2 for i in range(Dm)])
        ax.plot(np.arange(Dm), s[order], 'o', mfc='none', mec=COLOR[k], ms=6, mew=1.5, label=LABEL[k])
    ax.axhline(0, color='0.7', lw=0.8)
    ax.set_xticks(range(Dm), [labels[i] for i in order], rotation=90, fontsize=6)
    ax.set_ylabel('(H(+10%) - H(-10%)) / 2  [hits]')
    ax.legend(fontsize=7)
    ax.set_title(f'One parameter at a time at T = 1.5 ({n:.0e} muons): simulator vs. surrogates')
    save(fig, 'shield_oat.png')
    return floor


def shield_summary(rows, floor):
    D = dict(run_shield.DEFAULTS)
    cols = ['h0_err_T1', 'pred_half_T1', 'pred_box_T1', 'h0_err_T1.5', 'pred_box_T1.5', 'oat_slope_err',
            'oat_slope_cos', 'oat_curv_err', 'oat_curv_cos', 'steps', 'seconds']
    lines = ['Muon shield, by budget (DeepONet etc. at the defaults); median over seeds',
             'noise floors: ' + ', '.join(f'{k} {v:.3g}' for k, v in floor.items()),
             f'  {"surrogate":26s} {"budget":>7s} ' + ' '.join(f'{c:>13s}' for c in cols)]
    for k in run_shield.KINDS:
        for b in run_shield.BUDGETS:
            sel = matching(rows, dict(D, kind=k, budget=b), None)
            if sel:
                lines.append(f'  {LABEL[k]:26s} {b:7.0e} ' + ' '.join(f'{np.median([r[c] for r in sel]):13.3g}' for c in cols))
    for kind in run_shield.PRODUCTION:
        ex = [r for r in rows if r['config']['kind'] == kind]
        if ex:
            lines.append(f'  {LABEL[kind][:26]:26s} {"":7s} ' + ' '.join(f'{ex[0].get(c, np.nan):13.3g}' for c in cols))
    with open(os.path.join(FIGS, 'shield_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('  shield_summary.txt')


if __name__ == '__main__':
    os.makedirs(FIGS, exist_ok=True)
    toys_rows = run_toys.load_results()
    if toys_rows:
        print(f'toys: {len(toys_rows)} results; worst-case rescaled by ' + str(renormalize_worst(toys_rows)))
        toy_figures(toys_rows)
        toy_summary(toys_rows)
    shield_rows = run_shield.load_results()
    if shield_rows:
        print(f'shield: {len(shield_rows)} results')
        refs = {t: shield_ref.load(t) for t in run_shield.REFS}
        shield_summary(shield_rows, shield_figures(shield_rows, refs))
