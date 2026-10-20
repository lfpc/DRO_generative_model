"""Monte Carlo vs. the quadrature grid, as the analysis reports them: running estimates.

For four quantities of the robustness analysis (final surrogate, T = 1, per 1e9 muons):
  H_0                   the nominal hits
  H_0, pz > 300 GeV     the nominal hits of the sparse high-momentum muons
  M7:B at -10%          H/H_0 with the field of M7 alone lowered by 10%
  worst, |u| = 1        H/H_0 one tolerance unit along the worst direction
the running Monte Carlo estimate as muons accumulate (several seeds, the way the analysis
sampled them until now), the quadrature grid at increasing resolution, and the value the
earlier 1e9-muon Monte Carlo analysis reported, with its jackknife error.

    python quadrature_convergence.py    # -> outputs/surrogate_study/figs/quadrature_convergence.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from quadrature import FIGS, ROOT, Muons, grid_points, mc_points
from shield import MuonShield
from surrogate import load_surrogate

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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=type(
        'F', (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter), {}))
    parser.add_argument('--model', default=os.path.join(ROOT, 'outputs/lcso.pt'))
    parser.add_argument('--run', default=os.path.join(ROOT, 'outputs/archive/robustness_v2_mc'),
                        help='the Monte Carlo robustness run to compare with (its directions and values)')
    parser.add_argument('--n_max', type=float, default=1e9, help='muons per Monte Carlo run')
    parser.add_argument('--seeds', type=int, default=6)
    parser.add_argument('--grids', default='51x4,101x8,201x16,401x32,801x64,1601x64')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    shield = MuonShield()
    model = load_surrogate(args.model, shield.phi_0, 'cuda')
    M = Muons(shield, 'cuda')
    r = np.load(os.path.join(args.run, 'robustness.npz'), allow_pickle=True)
    labels, names = [str(s) for s in r['labels']], [str(s) for s in r['dir_names']]
    k = labels.index('M7:NI')
    v = r['dir_vecs'][names.index('worst')]
    U = np.stack([np.zeros(43), -np.eye(43)[k], np.clip(v, -1, 1)])
    i_w = r['dir_idx'][names.index('worst')][-1]
    pz = np.load(os.path.join(args.run, 'pz_bins.npz'), allow_pickle=True)
    earlier = [(float(r['H_0']), 3.608),               # the jackknife error its report gives
               (float(pz['H_b'][-1]), np.nan),         # (the bins had no errors)
               (1 + float(r['oat_minus'][k]), float(r['oat_minus_se'][k])),
               (float(r['ratio'][i_w]), float(r['ratio_se'][i_w]))]

    step = 2 ** 21                                    # record the running value every batch
    n_max = int(args.n_max) // step * step
    runs = []
    for seed in range(args.seeds):
        runs.append(sums(model, mc_points(M, n_max, 100 + seed, batch=2 ** 21), U, record_every=step))
        print(f'monte carlo seed {seed}: {runs[-1][-1][1]}', flush=True)
    grids = []
    for spec in args.grids.split(','):
        nz, nphi = map(int, spec.split('x'))
        grids.append((nz * nz * nphi * 2, spec, sums(model, grid_points(M, nz, nphi), U)))
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
        e, se = earlier[q]
        ax.errorbar([1e9], [e], yerr=[0 if np.isnan(se) else se], fmt='s', color='k', ms=6, capsize=3,
                    label='earlier analysis (1e9 MC, jackknife)')
        ax.set(xscale='log', xlabel='muons / grid points', title=QUANTITIES[q])
        lo, hi = ref[q] * 0.75, ref[q] * 1.25
        ax.set_ylim(lo, hi)
    axes[0].set_ylabel('per 1e9 muons, or H/H_0')
    axes[0].legend(fontsize=7.5, loc='lower right')
    fig.suptitle('The surrogate integrated over the muon flow: running Monte Carlo (as the analysis sampled until now) '
                 'vs. the quadrature grid (dotted: its finest value)')
    fig.savefig(os.path.join(FIGS, 'quadrature_convergence.png'), dpi=150)
    print('saved quadrature_convergence.png')
