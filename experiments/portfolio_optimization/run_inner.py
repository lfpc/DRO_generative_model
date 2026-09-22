"""How much can each ambiguity set degrade a fixed portfolio?

    python3 run_inner.py

Left: worst-case risk against each method's own radius -- honest, but the three radii are
in different units. Right: the same risk against the mean log-density of the returns the
adversary certifies against, which is comparable and is the question that matters -- a
worst case is only worth defending against if the returns behind it can occur.
"""
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
from problem import Portfolio                                      # noqa: E402
from train_generator import train                                  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
RHOS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
EPS = [0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.22]
STYLE = {'KL-DRO': ('C2', 'o'), 'Wasserstein': ('C1', 'v'), 'Latent-DRO': ('C0', 'D')}


def sweep(problem, flow, phi, x):
    rows = {k: [] for k in STYLE}
    lp = problem.true_log_prob(x)
    for rho in RHOS:
        v, w = inner.kl_dro(problem, phi, x, rho)
        rows['KL-DRO'].append((rho, v, float((w * lp).sum())))
    for eps in EPS:
        v, xa = inner.wasserstein_dro(problem, phi, x, eps)
        rows['Wasserstein'].append((eps, v, inner.plausibility(problem, xa)))
    for rho in RHOS:
        v, xa = inner.latent_dro(problem, flow, phi, rho)
        rows['Latent-DRO'].append((rho, v, inner.plausibility(problem, xa)))
    return rows


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
        ax.set_ylabel('worst-case mean-CVaR risk')
        ax.legend(fontsize=8)
    axes[0].set(xscale='log', xlabel="each method's own radius",
                title='damage vs the radius you ask for')
    axes[1].axvline(lp_nom, color='.5', lw=1, ls=':')
    axes[1].text(lp_nom, .02, ' real returns', color='.5', fontsize=7, rotation=90,
                 va='bottom', transform=axes[1].get_xaxis_transform())
    axes[1].set(xlabel='mean log-density of the adversarial returns (true model)',
                title='damage vs how implausible the returns are')
    axes[1].invert_xaxis()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'inner.png'), dpi=150)
    print(f'  wrote {OUT}/inner.png')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    problem = Portfolio()
    x = problem.sample(20000, torch.Generator().manual_seed(0))
    phi = problem.solve(x)
    flow, _ = train(problem, n_train=4000, epochs=800, seed=0)
    nom, lp_nom = inner.nominal(problem, phi, x), inner.plausibility(problem, x)
    rows = sweep(problem, flow, phi, x)
    print(f'nominal risk {nom:.4f}   log-density of real returns {lp_nom:.3f}')
    for name in STYLE:
        print(f'\n  {name}')
        for r, v, lp in rows[name]:
            print('    %8.3f  risk %8.4f  log-density %8.3f' % (r, v, lp))
    with open(os.path.join(OUT, 'inner.json'), 'w') as f:
        json.dump({'nominal': nom, 'lp_nominal': lp_nom, 'rows': rows}, f)
    plot(nom, lp_nom, rows)
