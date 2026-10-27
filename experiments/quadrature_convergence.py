"""Monte Carlo vs. the quadrature grid, as the analysis reports them: running estimates.

For four quantities of the robustness analysis (final surrogate, per 1e9 nominal muons):
  H_0                   the nominal hits
  H_0, pz > 300 GeV     the nominal hits of the sparse high-momentum muons
  M7:B at -10%          H/H_0 with the field of M7 alone lowered by 10%
  worst, |u| = 1        H/H_0 one tolerance unit along the worst direction
the running Monte Carlo estimate as muons accumulate (several seeds), and the quadrature grid at
increasing resolution. --earlier also draws the values a 1e9-muon Monte Carlo robustness run
(--run) reported, with its jackknife errors.

    python experiments/quadrature_convergence.py    # -> outputs/experiments/figs/quadrature_convergence.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from quadrature import FIGS, grid_points, mc_points

import cli
from lcso.surrogate import load_surrogate
from paths import RUN
from shield import MuonShield

QUANTITIES = ['H_0', 'H_0, pz > 300 GeV', 'M7:B at -10%', 'worst direction, |u| = 1']


@torch.no_grad()
def sums(model, points, U, record_every=None):
    """Weighted sums over points of p(hit) at designs U (m, D), and the same for pz > 300 GeV.
    With record_every: the running values after every that many points."""
    b = model.branch_net(torch.as_tensor(U, dtype=torch.float32, device=model.bias.device))
    S = torch.zeros(len(U) + 1, dtype=torch.float64, device=b.device)
    W, seen, rec = 0.0, 0, []
    for x, w in points:
        p = torch.sigmoid(model.features(x) @ b.T + model.bias).double()      # (n, m)
        S[:-1] += (p * w[:, None]).sum(0)
        S[-1] += (p[:, 0] * w * (x[:, 2] > 300)).sum()
        W += float(w.sum())
        seen += len(x)
        if record_every and seen % record_every == 0:
            rec.append((seen, values(S, W)))
    return (rec if record_every else values(S, W))


def values(S, W):
    s = (S / W * 1e9).cpu().numpy()
    return [s[0], s[-1], s[1] / s[0], s[2] / s[0]]


if __name__ == '__main__':
    parser = cli.add_common(cli.parser(__doc__))
    parser.add_argument('--run', default=RUN,
                        help='robustness run whose worst direction is used (and, with --earlier, its values)')
    parser.add_argument('--earlier', action='store_true', help='draw --run\'s Monte Carlo values (a sampled run)')
    parser.add_argument('--n_max', type=float, default=1e9, help='muons per Monte Carlo run')
    parser.add_argument('--seeds', type=int, default=6)
    parser.add_argument('--grids', default='51x4,101x8,201x16,401x32,801x64,1601x64')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    shield = MuonShield(args.muons)
    model = load_surrogate(args.model, shield.phi_0, args.device)
    r = np.load(os.path.join(args.run, 'robustness.npz'), allow_pickle=True)
    labels, names = [str(s) for s in r['labels']], [str(s) for s in r['dir_names']]
    k = labels.index('M7:NI')
    v = r['dir_vecs'][names.index('worst')]
    U = np.stack([np.zeros(shield.dim), -np.eye(shield.dim)[k], np.clip(v, -1, 1)])
    i_w = r['dir_idx'][names.index('worst')][-1]
    pz = np.load(os.path.join(args.run, 'pz_bins.npz'), allow_pickle=True) if args.earlier else None
    earlier = None if not args.earlier else [(float(r['H_0']), 3.608),               # the jackknife error its report gives
               (float(pz['H_b'][-1]), np.nan),         # (the bins had no errors)
               (1 + float(r['oat_minus'][k]), float(r['oat_minus_se'][k])),
               (float(r['ratio'][i_w]), float(r['ratio_se'][i_w]))]

    step = 2 ** 21                                    # record the running value every batch
    n_max = int(args.n_max) // step * step
    runs = []
    for seed in range(args.seeds):
        runs.append(sums(model, mc_points(shield, n_max, 100 + seed, batch=step), U, record_every=step))
        print(f'monte carlo seed {seed}: {runs[-1][-1][1]}', flush=True)
    grids = []
    for spec in args.grids.split(','):
        nz, nphi = map(int, spec.split('x'))
        grids.append((nz * nz * nphi * 2, spec, sums(model, grid_points(shield, spec), U)))
        print(f'grid {spec}: {grids[-1][2]}', flush=True)
    with open(os.path.join(FIGS, 'quadrature_convergence.json'), 'w') as f:
        json.dump(dict(runs=[[(n, [float(x) for x in val]) for n, val in run] for run in runs],
                       grids=[(n, s, [float(x) for x in val]) for n, s, val in grids], earlier=earlier), f)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), constrained_layout=True)
    ref = grids[-1][2]
    for q, ax in enumerate(axes):
        for j, run in enumerate(runs):
            ax.plot([n for n, _ in run], [val[q] for _, val in run], color='#2a78d6', lw=1, alpha=0.6,
                    label='Monte Carlo, running (6 seeds)' if j == 0 else None)
        ns = [n for n, _ in runs[0]]
        allv = np.array([[val[q] for _, val in run] for run in runs])
        ax.fill_between(ns, allv.mean(0) - allv.std(0), allv.mean(0) + allv.std(0), color='#2a78d6', alpha=0.12, lw=0)
        ax.plot([g[0] for g in grids], [g[2][q] for g in grids], '-o', color='#1baf7a', lw=2, ms=5,
                label='quadrature grid')
        ax.axhline(ref[q], color='#1baf7a', ls=':', lw=1)
        if earlier:
            e, se = earlier[q]
            ax.errorbar([1e9], [e], yerr=[0 if np.isnan(se) else se], fmt='s', color='k', ms=6, capsize=3,
                        label='Monte Carlo run (1e9, jackknife)')
        ax.set(xscale='log', xlabel='muons / grid points', title=QUANTITIES[q])
        lo, hi = ref[q] * 0.75, ref[q] * 1.25
        ax.set_ylim(lo, hi)
    axes[0].set_ylabel('per 1e9 muons, or H/H_0')
    axes[0].legend(fontsize=7.5, loc='lower right')
    fig.suptitle('The surrogate integrated over the muon flow: running Monte Carlo vs. the quadrature grid '
                 '(dotted: its finest value)')
    fig.savefig(os.path.join(FIGS, 'quadrature_convergence.png'), dpi=150)
    print('saved quadrature_convergence.png')
