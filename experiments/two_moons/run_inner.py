"""How much can each ambiguity set degrade a fixed classifier?

    python3 run_inner.py

Sweeps each adversary's radius and records what it costs the classifier. Two views,
because the radii are in different units and only one of them is comparable:

  left    worst-case risk against each method's own radius. Honest but not a like-for-like
          comparison: a reweighting radius is a divergence between weightings of one
          sample, a transport radius is a distance in input units, a latent radius is a
          data-space divergence.
  right   worst-case risk against the mean log-density of the points the adversary
          certifies against, under the TRUE model. This is comparable, and it is the
          question that matters for a physical design: damage is only meaningful if the
          inputs that cause it can occur.
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
from problem import TwoMoons                                       # noqa: E402
from train_generator import train                                  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
KL_RHOS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
W_EPS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45]
STYLE = {'KL-DRO': ('C2', 'o'), 'Wasserstein': ('C1', 'v'), 'Latent-DRO': ('C0', 'D')}


def sweep(problem, flows, phi, x, y):
    nom = inner.nominal(problem, phi, x, y)
    lp_nom = inner.plausibility(problem, x, y)
    rows = {k: [] for k in STYLE}
    for rho in KL_RHOS:
        risk, w = inner.kl_dro(problem, phi, x, y, rho)
        # reweighting moves no points: its samples are the observed ones, reweighted
        lp = float((w * problem.log_prob(x, y)).sum())
        rows['KL-DRO'].append((rho, risk, lp))
    for eps in W_EPS:
        risk, xa = inner.wasserstein_dro(problem, phi, x, y, eps)
        rows['Wasserstein'].append((eps, risk, inner.plausibility(problem, xa, y)))
    for rho in KL_RHOS:
        risk, xa, ya = inner.latent_dro(problem, flows, phi, rho)
        rows['Latent-DRO'].append((rho, risk, inner.plausibility(problem, xa, ya)))
    return nom, lp_nom, rows


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
        ax.set_ylabel('worst-case risk')
        ax.legend(fontsize=8)
    axes[0].set(xscale='log', xlabel="each method's own radius",
                title='damage vs the radius you ask for')
    axes[1].axvline(lp_nom, color='.5', lw=1, ls=':')
    axes[1].text(lp_nom, .02, ' real data', color='.5', fontsize=7, rotation=90,
                 va='bottom', transform=axes[1].get_xaxis_transform())
    axes[1].set(xlabel='mean log-density of the adversarial points (true model)',
                title='damage vs how implausible the inputs are')
    axes[1].invert_xaxis()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'inner.png'), dpi=150)
    print(f'  wrote {OUT}/inner.png')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    problem = TwoMoons(kind='poly', degree=5)
    x, y = problem.sample(20000, torch.Generator().manual_seed(0))
    phi = problem.fit(x, y, steps=3000)
    flows, _ = train(problem, n_train=4000, epochs=1500, seed=0)
    nom, lp_nom, rows = sweep(problem, flows, phi, x, y)
    print(f'nominal risk {nom:.4f}   accuracy {problem.accuracy(phi, x, y):.4f}   '
          f'mean log-density of real data {lp_nom:.3f}')
    for name in STYLE:
        print(f'\n  {name}')
        print('    %8s %10s %12s' % ('radius', 'worst risk', 'log-density'))
        for r, v, lp in rows[name]:
            print('    %8.3f %10.4f %12.3f' % (r, v, lp))
    with open(os.path.join(OUT, 'inner.json'), 'w') as f:
        json.dump({'nominal': nom, 'lp_nominal': lp_nom, 'rows': rows}, f)
    plot(nom, lp_nom, rows)
