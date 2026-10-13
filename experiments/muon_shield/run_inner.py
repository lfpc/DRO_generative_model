"""Sweep each adversary's radius and record the worst-case risk of the fixed shield.

    python3 run_inner.py                      # after run_inner_cost.py

Left panel: risk vs each method's own radius. Right panel: risk vs the mean log-density of the
adversarial muons under the true generator, the one axis on which the methods are comparable.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inner                                                       # noqa: E402
from problem import MuonShield                                     # noqa: E402
from run_inner_cost import load_surrogate                          # noqa: E402
from train_generator import Generator                              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
KL_RHOS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
W_EPS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]          # standardised feature units
STYLE = {'KL-DRO': ('C2', 'o'), 'Wasserstein (surrogate)': ('C3', 's'),
         'Latent-DRO (surrogate)': ('C0', 'D')}


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def sweep(problem, flow, sur, phi, x, y):
    nom = inner.nominal(problem, phi, x, y)
    lp = problem.log_prob(x, y)
    rows = {k: [] for k in STYLE}
    for rho in KL_RHOS:
        risk, w = inner.kl_dro(problem, phi, x, y, rho)
        rows['KL-DRO'].append((rho, risk, float((w * lp).sum())))
    for eps in W_EPS:
        risk, xa = inner.wasserstein_dro(problem, phi, x, y, eps, loss_fn=sur)
        rows['Wasserstein (surrogate)'].append((eps, risk, inner.plausibility(problem, xa, y)))
    for rho in KL_RHOS:
        risk, xa, ya = inner.latent_dro(problem, flow, phi, rho, n=x.shape[0], loss_fn=sur)
        rows['Latent-DRO (surrogate)'].append((rho, risk, inner.plausibility(problem, xa, ya)))
    return nom, float(lp.mean()), rows


def plot(nom, lp_nom, rows):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    for name, (c, m) in STYLE.items():
        r, v, lp = zip(*rows[name])
        axes[0].plot(r, v, color=c, marker=m, ms=4.5, mfc='none', lw=1.7, label=name)
        axes[1].plot(lp, v, color=c, marker=m, ms=4.5, mfc='none', lw=1.7, label=name)
    for ax in axes:
        ax.axhline(nom, color='.5', lw=1, ls=(0, (6, 3)))
        ax.text(.99, nom, 'nominal ', color='.5', fontsize=7, ha='right', va='bottom',
                transform=ax.get_yaxis_transform())
        ax.set_ylabel('worst-case risk (re-simulated)')
        ax.legend(fontsize=8)
    axes[0].set(xscale='log', xlabel="each method's own radius",
                title='damage vs the radius you ask for')
    axes[1].axvline(lp_nom, color='.5', lw=1, ls=':')
    axes[1].text(lp_nom, .02, ' real muons', color='.5', fontsize=7, rotation=90,
                 va='bottom', transform=axes[1].get_xaxis_transform())
    axes[1].set(xlabel='mean log-density of the adversarial muons (true generator)',
                title='damage vs how implausible the muons are')
    axes[1].invert_xaxis()
    fig.savefig(os.path.join(OUT, 'inner.png'), dpi=150)
    print(f'  wrote {OUT}/inner.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--n', type=float, default=1e8, help='nominal muons')
    a = ap.parse_args()
    a.n = int(a.n)
    problem = MuonShield()
    x, y = problem.sample(a.n, torch.Generator().manual_seed(0))
    phi = torch.load(os.path.join(MODELS, 'design.pt'), weights_only=False)
    flow = Generator.load(os.path.join(MODELS, 'generator.pt')).flow
    sur = load_surrogate(os.path.join(MODELS, 'surrogate.pt'), problem)
    nom, lp_nom, rows = sweep(problem, flow, sur, phi, x, y)
    print(f'nominal risk {nom:.4g}   hit rate {problem.hit_rate(phi, x, y):.4g}   '
          f'mean log-density of real muons {lp_nom:.3f}')
    for name in STYLE:
        print(f'\n  {name}')
        print('    %8s %12s %12s' % ('radius', 'worst risk', 'log-density'))
        for r, v, lp in rows[name]:
            print('    %8.3f %12.4g %12.3f' % (r, v, lp))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'inner.json'), 'w') as f:
        json.dump({'nominal': nom, 'lp_nominal': lp_nom, 'rows': rows}, f)
    plot(nom, lp_nom, rows)
