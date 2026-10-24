"""The reference-design figures for any production surrogates: one-at-a-time secants on the hot
check sample against the simulator, and predicted vs. simulated hits of every reference design.
Predictions are cached in figs/ref_pred_<name>.npz (the slow part: 1.4e9 stored muons).

    python ref_figures.py --models lcso,tr4n30 --labels DeepONet,Taylor --device cpu
    # -> figs/refs_<names>_oat.png, figs/refs_<names>_calibration.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import run_shield
import shield_ref

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # after this folder: both have a plots.py
from shield import MuonShield  # noqa: E402
from lcso.surrogate import load_surrogate  # noqa: E402

FIGS = os.path.join(run_shield.ROOT, 'outputs/surrogate_study/figs')
COLORS = ['#e34948', '#2a78d6', '#1baf7a', '#eb6834']


def predictions(name, device, refs):
    path = os.path.join(FIGS, f'ref_pred_{name}.npz')
    ckpt = os.path.join(run_shield.ROOT, 'outputs', f'{name}.pt')
    if os.path.exists(path) and os.path.getmtime(path) > os.path.getmtime(ckpt):
        return dict(np.load(path))
    model = run_shield.as_study(load_surrogate(ckpt, MuonShield().phi_0, device))
    out = {t: refs()[t].predict(model) for t in run_shield.REFS}
    np.savez(path, **out)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=type(
        'F', (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter), {}))
    parser.add_argument('--models', default='lcso', help='outputs/<name>.pt, comma-separated')
    parser.add_argument('--labels', default=None, help='legend labels (default: the names)')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    names = args.models.split(',')
    labels = args.labels.split(',') if args.labels else names
    loaded = {}

    def refs():
        if not loaded:
            loaded.update({t: run_shield.Reference(t) for t in run_shield.REFS})
        return loaded

    preds = {n: predictions(n, args.device, refs) for n in names}
    tag = '_'.join(names)

    # one-at-a-time secants on the hot check sample
    rn, U, hits, n, _ = shield_ref.load('T1.5')
    idx = {x: i for i, x in enumerate(rn)}
    D = U.shape[1]
    sim = np.array([(hits[idx[f'oat+{i}']] - hits[idx[f'oat-{i}']]) / 2 for i in range(D)])
    err = np.array([np.sqrt(hits[idx[f'oat+{i}']] + hits[idx[f'oat-{i}']]) / 2 for i in range(D)])
    order = np.argsort(-np.abs(sim))
    plabels = MuonShield().labels
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    ax.errorbar(np.arange(D), sim[order], yerr=err[order], fmt='s', color='0.25', ms=4, label='simulator (1 s.d.)')
    for j, (nm, lab) in enumerate(zip(names, labels)):
        p = preds[nm]['T1.5']
        s = np.array([(p[idx[f'oat+{i}']] - p[idx[f'oat-{i}']]) / 2 for i in range(D)])
        ax.plot(np.arange(D), s[order], 'o', mfc='none', mec=COLORS[j], ms=6, mew=1.5, label=lab)
    ax.axhline(0, color='0.7', lw=0.8)
    ax.set_xticks(range(D), [plabels[i] for i in order], rotation=90, fontsize=6)
    ax.set_ylabel('(H(+10%) - H(-10%)) / 2  [hits]')
    ax.legend(fontsize=7)
    ax.set_title(f'One parameter at a time, hot check sample ({n:.0e} muons): simulator vs. surrogate')
    fig.savefig(os.path.join(FIGS, f'refs_{tag}_oat.png'), dpi=150)

    # predicted vs. simulated hits of every reference design
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), constrained_layout=True)
    for ax, t in zip(axes, run_shield.REFS):
        rn, U, hits, n, _ = shield_ref.load(t)
        for j, (nm, lab) in enumerate(zip(names, labels)):
            ax.errorbar(hits, preds[nm][t], xerr=np.sqrt(hits), fmt='o', ms=3.5, color=COLORS[j], elinewidth=0.6,
                        label=lab, alpha=0.85)
        lim = [hits.min() * 0.8, hits.max() * 1.2]
        ax.plot(lim, lim, color='0.6', lw=1, zorder=0)
        ax.set(xscale='log', yscale='log', xlabel='simulated hits', ylabel='predicted hits',
               title=f'{run_shield.REF_LABEL[t]} ({n:.0e} muons)')
    axes[0].legend(fontsize=7)
    fig.savefig(os.path.join(FIGS, f'refs_{tag}_calibration.png'), dpi=150)
    print(f'saved refs_{tag}_oat.png, refs_{tag}_calibration.png')
