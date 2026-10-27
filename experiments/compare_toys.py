"""DeepONet vs. Taylor surrogates on the toy problems: how the errors of the gradient and the
Hessian at u = 0 (against the exact ones) fall with the number of simulated samples, and how many
samples each surrogate needs to reach a given error (log-log interpolation of the median over the
seeds; '>' when it does not get there within the largest budget).

    python experiments/compare_toys.py      # -> figs/compare_toys.png, figs/compare_toys.txt
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import run_toys

import cli
from paths import EXPERIMENTS, figures

FIGS = figures(EXPERIMENTS)
MODELS = [('deeponet', 0, 'DeepONet', '#e34948'),
          ('taylor-res4:8', 0.3, 'Taylor (full, weighted)', '#2a78d6'),
          ('taylor-res4:8', 0, 'Taylor (full)', '#1baf7a'),
          ('taylor-sub-res4:4', 0, 'Taylor (reduced)', '#eb6834')]
METRICS = [('grad_err', 'gradient error'), ('hess_err', 'Hessian error')]
THRESHOLDS = {'grad_err': (0.3, 0.1), 'hess_err': (1.0, 0.5)}


def curves(rows, problem, kind, near):
    """budget -> per-seed results of one model at the defaults (step cap aside)."""
    D = run_toys.DEFAULTS
    out = {}
    for r in rows:
        c = r['config']
        if (c['problem'], c['kind'], c.get('near', 0)) != (problem, kind, near) or 'grad_err' not in r:
            continue
        if any(c.get(k) != D[k] for k in ('n', 'delta', 'radial', 'scheme', 'p', 'hidden', 'depth', 'act', 'lr', 'batch')):
            continue
        out.setdefault(c['budget'], []).append(r)
    return dict(sorted(out.items()))


def needed(b, med, thr):
    """Smallest budget at which the median error falls below thr (log-log interpolation)."""
    for i in range(len(b)):
        if med[i] <= thr:
            if i == 0:
                return f'<={b[0]:.0e}'
            x = np.interp(np.log(thr), np.log([med[i], med[i - 1]]), np.log([b[i], b[i - 1]]))
            return f'{np.exp(x):.1e}'
    return f'>{b[-1]:.0e}'


if __name__ == '__main__':
    parser = cli.parser(__doc__)
    args = parser.parse_args()
    rows = run_toys.load_results()
    problems = sorted({r['config']['problem'] for r in rows})
    fig, axes = plt.subplots(len(METRICS), len(problems), figsize=(3.6 * len(problems), 3.1 * len(METRICS)),
                             constrained_layout=True, squeeze=False)
    lines = ['median over the seeds of the error at u = 0 against the exact derivatives; samples needed to reach '
             + ', '.join(f'{m} {t}' for m, t in THRESHOLDS.items()), '']
    for j, p in enumerate(problems):
        lines.append(f'== {p}')
        for i, (m, mlabel) in enumerate(METRICS):
            ax = axes[i, j]
            for kind, near, label, color in MODELS:
                c = curves(rows, p, kind, near)
                if not c:
                    continue
                b = np.array(list(c))
                vals = [np.array([r[m] for r in c[x]]) for x in b]
                med = np.array([np.median(v) for v in vals])
                ax.plot(b, med, '-o', color=color, ms=4, lw=1.8, label=label)
                for x, v in zip(b, vals):
                    ax.plot([x] * len(v), v, '.', color=color, ms=3, alpha=0.4)
                need = ', '.join(f'{t}: {needed(b, med, t)}' for t in THRESHOLDS[m])
                lines.append(f'  {m:9s} {label:24s} ' + ' '.join(f'{x:.0e}:{v:.3f}' for x, v in zip(b, med))
                             + f'   samples for {need}')
            ax.set(xscale='log', yscale='log', xlabel='simulated samples', title=f'{p}: {mlabel}')
        lines.append('')
    axes[0, 0].legend(fontsize=7)
    fig.savefig(os.path.join(FIGS, 'compare_toys.png'), dpi=150)
    with open(os.path.join(EXPERIMENTS, 'compare_toys.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
