"""What does each adversary think the world looks like, at equal damage?

    python3 run_inner_samples.py

The three radii are in different units, so comparing the sets at "the same radius" compares
nothing. Instead each adversary is calibrated by bisection to inflict the SAME worst-case
risk, and then we look at the distribution it used to do it. That is the question a
certificate answers: to believe a worst-case number you have to believe the inputs behind
it, so the honest comparison holds the number fixed and inspects the inputs.

KL-DRO moves no points -- it reweights the ones observed -- so its panel shows the sample
with marker area proportional to weight.
"""
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
TARGET = 0.35                    # worst-case risk every method is calibrated to reach


def calibrate(fn, target, lo, hi, iters=12):
    """Smallest radius reaching `target`, by bisection. Damage increases with radius."""
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if fn(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    return hi, fn(hi)


def boundary(problem, phi, ax):
    g = torch.linspace(-2.2, 3.2, 200)
    h = torch.linspace(-1.6, 1.9, 200)
    X, Y = torch.meshgrid(g, h, indexing='ij')
    with torch.no_grad():
        z = problem.logits(phi, torch.stack([X.reshape(-1), Y.reshape(-1)], 1))
    ax.contour(X, Y, z.reshape(200, 200), levels=[0], colors='k', linewidths=1.1)
    ax.set(xlim=(-2.2, 3.2), ylim=(-1.6, 1.9), xticks=[], yticks=[])


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    os.makedirs(OUT, exist_ok=True)
    problem = TwoMoons(kind='poly', degree=5)
    x, y = problem.sample(20000, torch.Generator().manual_seed(0))
    phi = problem.fit(x, y, steps=3000)
    flows, _ = train(problem, n_train=4000, epochs=1500, seed=0)
    nom = inner.nominal(problem, phi, x, y)
    lp_real = inner.plausibility(problem, x, y)
    print(f'nominal risk {nom:.4f};  calibrating every method to worst-case risk {TARGET}')

    r_kl, (v_kl, w) = calibrate(lambda r: inner.kl_dro(problem, phi, x, y, r),
                                TARGET, 1e-3, 5.0)
    r_w, (v_w, x_w) = calibrate(lambda e: inner.wasserstein_dro(problem, phi, x, y, e),
                                TARGET, 1e-3, 1.0)
    r_l, (v_l, x_l, y_l) = calibrate(lambda r: inner.latent_dro(problem, flows, phi, r),
                                     TARGET, 1e-3, 5.0)
    lp_kl = float((w * problem.log_prob(x, y)).sum())
    panels = [('nominal data', x, y, None, nom, lp_real, ''),
              ('KL-DRO', x, y, w, v_kl, lp_kl, f'$\\rho$={r_kl:.2f}'),
              ('Wasserstein', x_w, y, None, v_w, inner.plausibility(problem, x_w, y),
               f'$\\epsilon$={r_w:.2f}'),
              ('Latent-DRO', x_l, y_l, None, v_l,
               inner.plausibility(problem, x_l, y_l), f'$\\rho$={r_l:.2f}')]
    for name, _, _, _, v, lp, r in panels:
        print(f'  {name:14s} {r:10s} risk {v:.4f}   log-density {lp:7.3f}')

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), constrained_layout=True)
    for j, (name, xa, ya, wt, v, lp, r) in enumerate(panels):
        ax = axes[0, j]
        for s, c in ((1.0, 'C0'), (-1.0, 'C1')):
            m = ya == s
            size = 3.0 if wt is None else 4000 * wt[m]
            ax.scatter(xa[m, 0][:6000], xa[m, 1][:6000], s=(size if wt is None
                       else size[:6000]), alpha=.35, c=c, linewidths=0)
        boundary(problem, phi, ax)
        ax.set_title(f'{name}  {r}\nrisk {v:.3f},  log-density {lp:.2f}', fontsize=10)

        ax = axes[1, j]
        bins = torch.linspace(-2.2, 3.2, 70)
        for k, (lab, col) in enumerate((('$x_1$', 'C3'), ('$x_2$', 'C4'))):
            ax.hist(x[:, k], bins=bins, density=True, histtype='stepfilled', alpha=.25,
                    color=col)
            ax.hist(xa[:, k], bins=bins, density=True, histtype='step', lw=1.6,
                    color=col, weights=(None if wt is None else wt * len(wt)),
                    label=f'{lab} adversarial')
        ax.set(yticks=[], xlabel='coordinate value')
        if j == 0:
            ax.set_ylabel('density (filled: real)')
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(OUT, 'inner_samples.png'), dpi=150)
    print(f'  wrote {OUT}/inner_samples.png')
