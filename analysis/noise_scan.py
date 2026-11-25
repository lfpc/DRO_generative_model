"""The simulator's noise on one parameter: the design walked from -range to +range of one
parameter's nominal value in small steps, each point simulated once on the same 1e9 muons. The
transport is stochastic, so the hits jump from point to point by about their Poisson error,
more than a small change of the parameter moves them.

    python -m analysis.noise_scan --param M1:dXYokeIn              # simulate, then plot
    python -m analysis.noise_scan --param M1:dXYokeIn --plot_only

Workers split the designs, so several GPUs can share the work (then --plot_only merges them):

    CUDA_VISIBLE_DEVICES=0 python -m analysis.noise_scan --worker 0 --n_workers 2 &
    CUDA_VISIBLE_DEVICES=1 python -m analysis.noise_scan --worker 1 --n_workers 2 &
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import cli
from paths import RUN, figures, save_figure
from shield import pretty_text

parser = cli.add_common(cli.parser(__doc__))
parser.add_argument('--param', default='M1:dXYokeIn', help='the parameter walked (label M<i>:<column>)')
parser.add_argument('--range', type=float, default=20, help='walk from -range%% to +range%% of the nominal value')
parser.add_argument('--step', type=float, default=1, help='step in %% of the nominal value')
parser.add_argument('--n_muons', type=int, default=int(1e9))
parser.add_argument('--block', type=int, default=int(2e7), help='sampled muons held in memory')
parser.add_argument('--worker', type=int, default=0)
parser.add_argument('--n_workers', type=int, default=1)
parser.add_argument('--plot_only', action='store_true', help='plot the saved scan')
args = parser.parse_args()
stem = os.path.join(RUN, f'noise_scan_{args.param.replace(":", "_")}')

if not args.plot_only:
    from analysis.core import muon_blocks, one_group, setup, simulate
    args.fixed = ''
    shield, model = setup(args)
    pct = np.round(np.arange(-args.range, args.range + args.step / 2, args.step), 6)[args.worker::args.n_workers]
    U = np.zeros((len(pct), model.dim))
    U[:, shield.labels.index(args.param)] = pct / (100 * model.delta)
    print(f'{args.param}: {len(pct)} designs over {args.n_muons:,} muons')
    hits = simulate(model, shield, muon_blocks(shield, args.n_muons, args.block), U, one_group, 1)[:, 0]
    out = f'{stem}_{args.worker}.npz'
    np.savez(out, param=args.param, pct=pct, hits=hits, n_muons=args.n_muons)
    print(f'saved {out}')
    if args.n_workers > 1:
        raise SystemExit('plot with --plot_only once every worker is done')

import glob
parts = [np.load(f) for f in sorted(glob.glob(f'{stem}_*.npz'))]
r = parts[0]
pct, y = (np.concatenate([q[k] for q in parts]) for k in ('pct', 'hits'))
o = np.argsort(pct)
pct, y = pct[o], y[o]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.axvline(0, color='red', ls='--', lw=1.2, label='nominal')
ax.errorbar(pct, y, yerr=np.sqrt(y), fmt='o', ms=4, color='k', ecolor='0.6', elinewidth=1, capsize=0,
            label=r'simulated, $\pm\sqrt{Y}$')
ax.set_xlabel(f'change of {pretty_text(r["param"])} [%]')
ax.set_ylabel(fr'$Y$ per ${r["n_muons"] / 1e9:g}\times10^9$ muons')
ax.grid(alpha=0.3)
ax.legend(frameon=False, loc='best')
fig.tight_layout()
save_figure(fig, figures(RUN), 'noise_scan', 'noise_scan')
print(f'Y: mean {y.mean():.0f}, rms {y.std():.0f} ({y.std() / y.mean():.1%}); Poisson {np.sqrt(y.mean()):.0f}; '
      f'largest jump between neighbours {np.abs(np.diff(y)).max():.0f}')
