"""What flux does each adversary reach, at equal damage?

    python3 run_inner_samples.py              # after run_inner_cost.py

Every method is calibrated by bisection to the same worst-case risk, --ratio x nominal, and
the |p| and p_T spectra of its muons are compared to the nominal ones.
"""
import argparse
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
from problem import MuonShield                                     # noqa: E402
from run_inner_cost import load_surrogate                          # noqa: E402
from train_generator import Generator                              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def calibrate(fn, target, lo, hi, iters=10):
    """Smallest radius reaching `target`, by geometric bisection. Damage grows with radius."""
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if fn(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    out = fn(hi)
    if out[0] < target:
        print(f'    (target not reached at the largest radius {hi:.3g}: {out[0]:.4g})')
    return hi, out


def momenta(problem, x, y):
    mu = problem.to_muons(torch.as_tensor(x), torch.as_tensor(y))
    return mu[:, :3].norm(dim=1).numpy(), mu[:, :2].norm(dim=1).numpy()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--n', type=float, default=1e8, help='nominal muons')
    ap.add_argument('--ratio', type=float, default=1.25, help='target risk / nominal risk')
    ap.add_argument('--iters', type=int, default=10, help='bisection steps per method')
    a = ap.parse_args()
    a.n = int(a.n)
    os.makedirs(OUT, exist_ok=True)
    problem = MuonShield()
    x, y = problem.sample(a.n, torch.Generator().manual_seed(0))
    phi = torch.load(os.path.join(MODELS, 'design.pt'), weights_only=False)
    flow = Generator.load(os.path.join(MODELS, 'generator.pt')).flow
    sur = load_surrogate(os.path.join(MODELS, 'surrogate.pt'), problem)
    nom = inner.nominal(problem, phi, x, y)
    lp = problem.log_prob(x, y)
    target = a.ratio * nom
    print(f'nominal risk {nom:.4g};  calibrating every method to {target:.4g}')

    r_kl, (v_kl, wt) = calibrate(lambda r: inner.kl_dro(problem, phi, x, y, r),
                                 target, 1e-4, 5.0, a.iters)
    r_w, (v_w, x_w) = calibrate(lambda e: inner.wasserstein_dro(problem, phi, x, y, e,
                                                                loss_fn=sur),
                                target, 1e-3, 5.0, a.iters)
    r_l, (v_l, x_l, y_l) = calibrate(lambda r: inner.latent_dro(problem, flow, phi, r, n=a.n,
                                                                loss_fn=sur),
                                     target, 1e-3, 5.0, a.iters)
    panels = [('nominal', x, y, None, nom, float(lp.mean()), ''),
              ('KL-DRO', x, y, wt, v_kl, float((wt * lp).sum()), f'$\\rho$={r_kl:.3g}'),
              ('Wasserstein (sur.)', x_w, y, None, v_w, inner.plausibility(problem, x_w, y),
               f'$\\epsilon$={r_w:.3g}'),
              ('Latent-DRO (sur.)', x_l, y_l, None, v_l, inner.plausibility(problem, x_l, y_l),
               f'$\\rho$={r_l:.3g}')]
    for n_, _, _, _, v, l_, r in panels:
        print(f'  {n_:20s} {r:16s} risk {v:10.4g}   log-density {l_:8.3f}')

    p0, pt0 = momenta(problem, x, y)
    bp = np.logspace(np.log10(p0.min()), np.log10(400), 80)
    bt = np.linspace(0, np.quantile(pt0, .999) * 1.5, 80)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True)
    for j, (name, xa, ya, wt_, v, l_, r) in enumerate(panels):
        pa, pta = momenta(problem, xa, ya)
        ww = None if wt_ is None else wt_.numpy() * len(wt_)
        for ax, real, adv, bins, lab in ((axes[0, j], p0, pa, bp, '|p| (GeV)'),
                                         (axes[1, j], pt0, pta, bt, '$p_T$ (GeV)')):
            ax.hist(real, bins=bins, density=True, histtype='stepfilled', alpha=.3, color='C7',
                    label='nominal')
            ax.hist(adv, bins=bins, density=True, weights=ww, histtype='step', lw=1.7,
                    color='C0', label='adversarial')
            ax.set(xlabel=lab, yscale='log', xscale='log' if bins is bp else 'linear')
            ax.legend(fontsize=7)
        axes[0, j].set_title(f'{name}  {r}\nrisk {v:.4g},  log-density {l_:.2f}', fontsize=9)
    fig.savefig(os.path.join(OUT, 'inner_samples.png'), dpi=150)
    print(f'  wrote {OUT}/inner_samples.png')
