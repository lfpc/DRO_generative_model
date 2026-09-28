"""What does each adversary think returns look like, at equal damage?

    python3 run_inner_samples.py

Every method is calibrated by bisection to the SAME worst-case risk, then we look at the
returns it used. The loss sees the ten assets only through r = w'xi and penalises the lower
tail of r, so the return distribution is the object to inspect -- a ten-dimensional scatter
would hide exactly what matters.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inner                                                       # noqa: E402
from problem import Portfolio                                      # noqa: E402
from train_generator import train                                  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
TARGET = -0.55                   # worst-case risk every method is calibrated to reach


def calibrate(fn, target, lo, hi, iters=12):
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if fn(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    return hi, fn(hi)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    os.makedirs(OUT, exist_ok=True)
    problem = Portfolio()
    x = problem.sample(20000, torch.Generator().manual_seed(0))
    phi = problem.solve(x)
    w_opt, tau = phi[:-1], float(phi[-1])
    flow, _ = train(problem, n_train=4000, epochs=800, seed=0)
    nom = inner.nominal(problem, phi, x)
    lp = problem.true_log_prob(x)
    print(f'nominal risk {nom:.4f};  calibrating every method to {TARGET}')

    r_kl, (v_kl, wt) = calibrate(lambda r: inner.kl_dro(problem, phi, x, r),
                                 TARGET, 1e-3, 5.0)
    r_w, (v_w, x_w) = calibrate(lambda e: inner.wasserstein_dro(problem, phi, x, e),
                                TARGET, 1e-3, 1.0)
    r_l, (v_l, x_l) = calibrate(lambda r: inner.latent_dro(problem, flow, phi, r),
                                TARGET, 1e-3, 5.0)
    panels = [('real returns', x, None, nom, float(lp.mean()), ''),
              ('KL-DRO', x, wt, v_kl, float((wt * lp).sum()), f'$\\rho$={r_kl:.2f}'),
              ('Wasserstein', x_w, None, v_w, inner.plausibility(problem, x_w),
               f'$\\epsilon$={r_w:.3f}'),
              ('Latent-DRO', x_l, None, v_l, inner.plausibility(problem, x_l),
               f'$\\rho$={r_l:.2f}')]
    for n_, _, _, v, l_, r in panels:
        print(f'  {n_:14s} {r:12s} risk {v:8.4f}   log-density {l_:7.3f}')

    r_real = (x @ w_opt).numpy()
    bins = np.linspace(-0.25, 0.6, 90)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True)
    for j, (name, xa, wt_, v, l_, r) in enumerate(panels):
        ra = (xa @ w_opt).numpy()
        ww = None if wt_ is None else wt_.numpy() * len(wt_)
        ax = axes[0, j]
        ax.hist(r_real, bins=bins, density=True, histtype='stepfilled', alpha=.3,
                color='C7', label='real')
        ax.hist(ra, bins=bins, density=True, weights=ww, histtype='step', lw=1.7,
                color='C0', label='adversarial')
        ax.axvline(-tau, color='C3', lw=1.2, ls='--')
        ax.text(-tau, .96, ' VaR level', color='C3', fontsize=7, rotation=90, va='top',
                transform=ax.get_xaxis_transform())
        ax.set(title=f'{name}  {r}\nrisk {v:.3f},  log-density {l_:.2f}',
               xlabel="portfolio return  $w^\\top\\xi$", yticks=[])
        ax.legend(fontsize=7)

        ax = axes[1, j]                       # where the damage comes from: the lower tail
        q = np.linspace(0.001, 0.30, 120)
        ax.plot(q, np.quantile(r_real, q), color='C7', lw=2.0, label='real')
        if ww is None:
            ax.plot(q, np.quantile(ra, q), color='C0', lw=1.6, ls='--', label='adversarial')
        else:
            o = np.argsort(ra)
            cw = np.cumsum(ww[o]) / ww.sum()
            ax.plot(q, np.interp(q, cw, ra[o]), color='C0', lw=1.6, ls='--',
                    label='adversarial')
        ax.set(xlabel='lower-tail quantile', ylabel='portfolio return' if j == 0 else None)
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(OUT, 'inner_samples.png'), dpi=150)
    print(f'  wrote {OUT}/inner_samples.png')
