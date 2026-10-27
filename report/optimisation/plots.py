"""Figures of the optimisation report: best f against the muons simulated (in full simulations), median over the
seeds and the range of the seeds, for the local LCSO runs of the MuonShieldChallenge and its baselines.

    python plots.py --challenge ~/projects/MuonShieldChallenge
    # -> figs/easy_curves.png, figs/easy_kinds.png, figs/hard_curves.png
"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIGS = Path(__file__).resolve().parent / 'figs'
GRAY = ['#9a9a9a', '#6f6f6f', '#c2c2c2', '#4a4a4a', '#b0b0b0', '#858585']
COLORS = ['#e34948', '#2a78d6', '#1baf7a', '#eb6834', '#8a5cc2']


def curves(path, tag=None, grid=None):
    """{name: (seeds, len(grid)) best f} of a results file; name is the method, or tag for a lcso_local set."""
    out = defaultdict(list)
    for key, run in json.loads(Path(path).read_text()).items():
        n_samples = run['config']['n_samples']
        used = np.cumsum([h['n_muons'] for h in run['history']]) / n_samples
        f = np.array([h['f'] if h.get('f') is not None else np.inf for h in run['history']])
        best = np.minimum.accumulate(f)
        k = np.searchsorted(used, grid, side='right') - 1     # the last simulation finished by each budget
        c = np.where(k >= 0, best[np.maximum(k, 0)], np.nan)
        c[grid > used[-1] + 1e-6] = np.nan
        c[~np.isfinite(c)] = np.nan
        out[tag or re.sub(r'_seed\d+$', '', key)].append(c)
    return {k: np.array(v) for k, v in out.items()}


def panel(ax, groups, grid, styles, title):
    for name, (label, color, ls) in styles.items():
        if name not in groups:
            continue
        a = groups[name]
        med = np.nanmedian(a, 0)
        ax.plot(grid, med, color=color, ls=ls, lw=2 if ls == '-' else 1.4, label=f'{label} ({len(a)})')
        if ls == '-':
            ax.fill_between(grid, np.nanmin(a, 0), np.nanmax(a, 0), color=color, alpha=0.12, lw=0)
    ax.set(xscale='log', yscale='log', xlabel='muons simulated [full simulations]', ylabel='best hits', title=title)
    ax.grid(True, which='both', color='0.92', lw=0.6)
    ax.legend(fontsize=7, ncol=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=type(
        'F', (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter), {}))
    ap.add_argument('--challenge', default=os.path.expanduser('~/projects/MuonShieldChallenge'))
    args = ap.parse_args()
    root = Path(args.challenge) / 'results'
    FIGS.mkdir(exist_ok=True)

    def local(task, sets, grid):
        g = {}
        for d, name in sets:
            p = root / d / name / f'{task}.json'
            if p.exists():
                g.update(curves(p, name, grid))
        return g

    # easy: the final local LCSO against the baselines
    grid = np.geomspace(2, 400, 300)
    g = curves(root / 'baselines' / 'easy.json', grid=grid)
    g.update(local('easy', [('lcso_local3', 'R_taylor'), ('lcso_local3', 'R_deeponet'), ('lcso_local3', 'R'),
                            ('lcso_local6', 'C_follow2'), ('lcso_local', 'base')], grid))
    base_styles = {'rl_model_based': ('RL, model-based', GRAY[3], '--'),
                   'slice_model_based': ('slice, model-based', GRAY[1], ':'),
                   'turbo': ('TuRBO', GRAY[0], '--'), 'cmaes': ('CMA-ES', GRAY[2], '-.'),
                   'ga': ('GA', GRAY[4], ':'), 'lcso_newton': ('LCSO global (old), Newton', GRAY[5], '-.')}
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    panel(ax, g, grid, {**base_styles,
                        'C_follow2': ('local LCSO, full Taylor (final)', COLORS[0], '-'),
                        'R_deeponet': ('local LCSO, DeepONet', COLORS[1], '-'),
                        'base': ('local LCSO, first version', COLORS[3], '-')},
          'Easy (30 parameters, 1e7 muons per simulation)')
    fig.savefig(FIGS / 'easy_curves.png', dpi=150)

    # easy: the surrogate kinds, same optimiser
    g = local('easy', [('lcso_local3', n) for n in ('R_taylor', 'R_deeponet', 'R', 'R_tsr16', 'R_tsub8')], grid)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    panel(ax, g, grid, {'R_taylor': ('full Taylor', COLORS[0], '-'), 'R_deeponet': ('DeepONet', COLORS[1], '-'),
                        'R': ('reduced Taylor r=8 + residual', COLORS[2], '-'),
                        'R_tsr16': ('reduced Taylor r=16 + residual', COLORS[3], '-'),
                        'R_tsub8': ('reduced Taylor r=8', COLORS[4], '-')},
          'Easy: the surrogate kinds in the same local optimiser')
    fig.savefig(FIGS / 'easy_kinds.png', dpi=150)

    # hard
    grid = np.geomspace(2, 500, 300)
    g = curves(root / 'baselines' / 'hard.json', grid=grid)
    g.update(local('hard', [('lcso_local_hard', n) for n in ('H', 'HF_g025', 'H_deeponet')], grid))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    panel(ax, g, grid, {'turbo': ('TuRBO', GRAY[0], '--'), 'bo': ('BO', GRAY[1], ':'),
                        'cmaes': ('CMA-ES', GRAY[2], '-.'), 'ga': ('GA', GRAY[4], ':'),
                        'lcso_newton': ('LCSO global (old), Newton', GRAY[5], '-.'),
                        'HF_g025': ('local LCSO, full Taylor, far steps', COLORS[0], '-'),
                        'H_deeponet': ('local LCSO, DeepONet', COLORS[1], '-'),
                        'H': ('local LCSO, reduced Taylor r=8 + residual', COLORS[2], '-')},
          'Hard (72 parameters, 5e8 muons per simulation, noisy)')
    fig.savefig(FIGS / 'hard_curves.png', dpi=150)
    print('saved', *sorted(p.name for p in FIGS.glob('*.png')))


if __name__ == '__main__':
    main()
