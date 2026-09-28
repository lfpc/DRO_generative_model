"""What does each adversary think the inputs look like, at equal damage?

    python3 run_inner_samples.py

Every method is calibrated by bisection to the SAME worst-case risk, so the comparison is
about the shape of the distribution each one reaches rather than about radii, which are not
commensurable across ambiguity sets. The cost reaches the twelve sensor readings only through the system limit state, the
minimum of the five column limit states, and the structure fails where that is
negative. That minimum is therefore the scalar to look at.

The generator is loaded, not refitted: this script compares adversaries, and refitting the
flow inside it would put a second source of variation into that comparison.
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
from problem import ShortColumn                                        # noqa: E402
from train_generator import Generator                              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
TARGET = 2.10              # worst-case risk every method is calibrated to reach
TAIL = 'lower'                # which tail of the projection carries the damage


def calibrate(fn, target, lo, hi, iters=10):
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if fn(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    return hi, fn(hi)


def project(problem, phi, x):
    """System limit state, min over columns -- the scalar the loss actually responds to."""
    return (problem.limit_state(phi, torch.as_tensor(x)).min(-1).values).detach().numpy()


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    os.makedirs(OUT, exist_ok=True)
    problem = ShortColumn()
    x = problem.sample(20000, torch.Generator().manual_seed(0))
    phi = torch.load(os.path.join(MODELS, 'design.pt'), weights_only=False)
    flow = Generator.load(os.path.join(MODELS, 'generator.pt')).flow
    nom = inner.nominal(problem, phi, x)
    lp = problem.true_log_prob(x)
    print(f'nominal risk {nom:.4f};  calibrating every method to {TARGET}')

    r_kl, (v_kl, wt) = calibrate(lambda r: inner.kl_dro(problem, phi, x, r),
                                 TARGET, 1e-3, 5.0)
    r_w, (v_w, x_w) = calibrate(lambda e: inner.wasserstein_dro(problem, phi, x, e),
                                TARGET, 1e-4, 1.0)
    r_l, (v_l, x_l) = calibrate(lambda r: inner.latent_dro(problem, flow, phi, r),
                                TARGET, 1e-3, 5.0)
    panels = [('nominal', x, None, nom, float(lp.mean()), ''),
              ('KL-DRO', x, wt, v_kl, float((wt * lp).sum()), f'$\\rho$={r_kl:.3g}'),
              ('Wasserstein', x_w, None, v_w, inner.plausibility(problem, x_w),
               f'$\\epsilon$={r_w:.3g}'),
              ('Latent-DRO', x_l, None, v_l, inner.plausibility(problem, x_l),
               f'$\\rho$={r_l:.3g}')]
    for n_, _, _, v, l_, r in panels:
        print(f'  {n_:14s} {r:14s} risk {v:9.4f}   log-density {l_:8.3f}')

    s_real = project(problem, phi, x)
    lo, hi = np.quantile(s_real, [.002, .998])
    pad = 0.35 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 90)
    q = np.linspace(0.001, 0.30, 120) if TAIL == 'lower' else np.linspace(0.70, 0.999, 120)

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True)
    for j, (name, xa, wt_, v, l_, r) in enumerate(panels):
        sa = project(problem, phi, xa)
        ww = None if wt_ is None else wt_.numpy() * len(wt_)
        ax = axes[0, j]
        ax.hist(s_real, bins=bins, density=True, histtype='stepfilled', alpha=.3,
                color='C7', label='nominal')
        ax.hist(sa, bins=bins, density=True, weights=ww, histtype='step', lw=1.7,
                color='C0', label='adversarial')
        ax.axvline(0.0, color='C3', lw=1.2, ls='--')
        ax.text(0.0, .96, ' failure', color='C3', fontsize=7, rotation=90,
                va='top', transform=ax.get_xaxis_transform())
        ax.set(title=f'{name}  {r}\nrisk {v:.3f},  log-density {l_:.2f}',
               xlabel='system limit state  $\\min_j g_j$', yticks=[])
        ax.legend(fontsize=7)

        ax = axes[1, j]                    # where the damage comes from: the loaded tail
        ax.plot(q, np.quantile(s_real, q), color='C7', lw=2.0, label='nominal')
        if ww is None:
            ax.plot(q, np.quantile(sa, q), color='C0', lw=1.6, ls='--', label='adversarial')
        else:
            o = np.argsort(sa)
            cw = np.cumsum(ww[o]) / ww.sum()
            ax.plot(q, np.interp(q, cw, sa[o]), color='C0', lw=1.6, ls='--',
                    label='adversarial')
        ax.set(xlabel=f'{TAIL}-tail quantile',
               ylabel='system limit state  $\\min_j g_j$' if j == 0 else None)
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(OUT, 'inner_samples.png'), dpi=150)
    print(f'  wrote {OUT}/inner_samples.png')
