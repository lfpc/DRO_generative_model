"""LCSO on one toy problem: every surrogate, both sampling schemes, several seeds.

    python3 optb/compare_lcso.py --problem short_column
    python3 optb/compare_lcso.py --problem beam_shield --seeds 1 --budget 500000   # quick
    python3 optb/compare_lcso.py --problem two_moons --plot_only

Same wrapper and settings as the problem's train_lcso.py, for the surrogates
  taylor               rank-8 quadratic, trust-region Newton step
  taylor-full          full-rank quadratic re-centred on the incumbent every round
  taylor-res           taylor-full + gated residual network
  taylor-res:0.1:fold  taylor-res, residual penalty 0.1, residual folded into the
                       polynomial on every move of the centre
  mlp                  BranchTrunk, first-order step of trust-region length
under resample and nominal, against the exact-gradient reference of that script, all at
the same sample budget. Reported: final true objective of the believed-best design, and
the % of the improvement (initial -> exact gradient) it captures (>100% if the reference
has not converged).
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
sys.path.insert(0, ROOT)
from optb.lcso import Budget, lcso                                # noqa: E402

# the wrapper class each problem's train_lcso.py defines
WRAPPER = {'short_column': 'ForLCSO', 'beam_shield': 'ForLCSO', 'newsvendor': 'ForLCSO',
           'portfolio_optimization': 'PortfolioForLCSO', 'two_moons': 'MoonsForLCSO'}
# surrogate -> (step, rank); rank None = full rank
SURROGATES = {'taylor': ('tr-newton', 8), 'taylor-full': ('tr-newton', None),
              'taylor-res': ('tr-newton', None), 'taylor-res:0.1:fold': ('tr-newton', None),
              'mlp': ('gradient', 8)}
SCHEMES = ('resample', 'nominal')
COLOR = {'taylor': '#2a78d6', 'mlp': '#eb6834', 'taylor-res': '#1baf7a',
         'taylor-full': '#eda100', 'taylor-res:0.1:fold': '#7b4fc9'}
STYLE = {'resample': '-', 'nominal': '--'}


def load(name):
    """The problem's train_lcso module, by path (it imports its own problem.py)."""
    d = os.path.join(ROOT, name)
    sys.path.insert(0, d)
    s = importlib.util.spec_from_file_location(f'{name}_train_lcso',
                                               os.path.join(d, 'train_lcso.py'))
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


def on_grid(trace, grid):
    """Best-so-far true objective at each budget on the grid (step function)."""
    s, v = map(np.asarray, zip(*trace))
    idx = np.searchsorted(s, grid, side='right') - 1
    return np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)


def plot(res, path):
    grid = np.asarray(res['grid'])
    init, ref = res['init'], res['ref']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    ax = axes[0]
    for s in SURROGATES:
        for k in SCHEMES:
            runs = [r for r in res['runs'] if r['surrogate'] == s and r['scheme'] == k]
            if not runs:
                continue
            curves = np.array([on_grid(r['trace'], grid) for r in runs])
            ax.plot(grid, np.nanmedian(curves, 0), STYLE[k], color=COLOR[s], lw=2,
                    label=f'{s} ({k})')
            ax.fill_between(grid, np.nanmin(curves, 0), np.nanmax(curves, 0),
                            color=COLOR[s], alpha=0.10, lw=0)
    s_, v = map(np.asarray, zip(*res['ref_trace']))
    ax.plot(s_, v, ':', color='0.35', lw=1.6, label='exact gradient')
    ax.axhline(init, color='0.6', lw=1, ls=(0, (6, 3)))
    ax.set(xlabel='samples simulated', ylabel='true objective of the best design',
           title='convergence (median and range over seeds)')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    labels, xs = [], 0
    for s in SURROGATES:
        for k in SCHEMES:
            runs = [r for r in res['runs'] if r['surrogate'] == s and r['scheme'] == k]
            if not runs:
                continue
            cap = [100 * (r['final'] - init) / (ref - init) for r in runs]
            ax.bar(xs, np.median(cap), color=COLOR[s], alpha=1.0 if k == 'resample' else 0.45,
                   width=0.7)
            ax.plot([xs] * len(cap), cap, 'o', color='0.15', ms=4, zorder=3)
            labels.append(f'{s}\n{k}')
            xs += 1
    ax.axhline(100, color='0.35', lw=1, ls=':')
    ax.set(xticks=range(len(labels)), ylabel='% of the exact-gradient improvement',
           title='final design (bar: median; dots: seeds)')
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.5, axis='y')
    fig.suptitle(f'{res["problem"]} (D = {res["dim"]}), budget {res["budget"]:,} samples',
                 fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  wrote {path}')


def table(res):
    init, ref = res['init'], res['ref']
    print(f'\n  {res["problem"]}: initial {init:.4f}, exact gradient {ref:.4f}')
    print('  %-24s %10s %10s %12s %8s' % ('', 'final', '(range)', '% captured', 'min/run'))
    for s in SURROGATES:
        for k in SCHEMES:
            runs = [r for r in res['runs'] if r['surrogate'] == s and r['scheme'] == k]
            if not runs:
                continue
            f = np.array([r['final'] for r in runs])
            cap = 100 * (np.median(f) - init) / (ref - init)
            print('  %-24s %10.4f %10s %11.0f%% %8.1f' % (
                f'{s} ({k})', np.median(f), f'{f.min():.3f}-{f.max():.3f}', cap,
                np.median([r['seconds'] for r in runs]) / 60))


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter,
                   argparse.RawDescriptionHelpFormatter):
    pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--problem', default='short_column', choices=sorted(WRAPPER))
    ap.add_argument('--surrogates', nargs='+', default=list(SURROGATES),
                    choices=list(SURROGATES))
    ap.add_argument('--budget', type=int, default=2_000_000, help='simulated samples per run')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--merge', action='store_true',
                    help='keep saved runs of surrogates not run now')
    ap.add_argument('--plot_only', action='store_true', help='re-plot the saved JSON')
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    js = os.path.join(OUT, f'compare_lcso_{a.problem}.json')
    fig = os.path.join(OUT, f'compare_lcso_{a.problem}.png')
    if a.plot_only:
        res = json.load(open(js))
        table(res)
        plot(res, fig)
        sys.exit()

    mod = load(a.problem)
    problem = getattr(mod, WRAPPER[a.problem])()
    print(f'{a.problem}: design dimension {problem.dim};  budget {a.budget:,} samples')
    init = problem.true_objective(problem.init_design())
    phi_ref, ref_trace = mod.exact_gradient_descent(problem, a.budget)
    ref = problem.true_objective(phi_ref)
    print(f'  initial {init:.4f}   exact gradient {ref:.4f}')

    kept = []
    if a.merge and os.path.exists(js):
        kept = [r for r in json.load(open(js))['runs'] if r['surrogate'] not in a.surrogates]
    runs = []
    for seed in range(a.seeds):
        for s in a.surrogates:
            step, rank = SURROGATES[s]
            for k in SCHEMES:
                t0 = time.time()
                b = Budget(a.budget)
                # the settings of the problems' own train_lcso.py
                phi, final = lcso(problem, b, scheme=k, seed=seed, n_samples=800,
                                  min_per_design=100, n_designs=4 * problem.dim,
                                  surrogate=s, step=step, sampler='sobol', epochs=40,
                                  rank=rank)
                runs.append(dict(surrogate=s, scheme=k, seed=seed, final=final,
                                 designs=b.designs, trace=b.trace,
                                 seconds=time.time() - t0))
                print(f'  seed {seed}  {s:10s} {k:8s} final {final:.4f}  captures '
                      f'{100 * (final - init) / (ref - init):5.0f}%  ({b.designs} designs, '
                      f'{(time.time() - t0) / 60:.1f} min)', flush=True)
            res = dict(problem=a.problem, dim=problem.dim, budget=a.budget, init=init,
                       ref=ref, ref_trace=ref_trace,
                       grid=np.linspace(a.budget / 100, a.budget, 100).tolist(), runs=kept + runs)
            json.dump(res, open(js, 'w'))               # after every surrogate
    table(res)
    plot(res, fig)
