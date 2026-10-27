"""Four adversaries calibrated to the same worst-case risk at a fixed design; for each,
measure its cost (true-loss evaluations), the surrogate's error at its points, and their
plausibility (log-density under the true model).

    python3 run_compare.py --problem short_column
    python3 run_compare.py --problem muon_shield --n 100000000

Writes <problem>.json (read by combine.py) and figs/compare_<problem>.png.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'figs')
sys.path.insert(0, os.path.join(ROOT, 'outer'))
from run_outer import PROBLEMS, Adapter, draw_from_weights, setup  # noqa: E402

METHODS = ('KL-DRO (true loss)', 'Wasserstein (true loss)',
           'Wasserstein (surrogate)', 'Latent-DRO (surrogate)')
COLOR = {m: c for m, c in zip(METHODS, ('C2', 'C1', 'C3', 'C0'))}


def calibrate(fn, target, lo, hi, iters=10):
    """Geometric bisection on the radius until fn(radius)[0] reaches target."""
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if fn(mid)[0] < target else (lo, mid)
    return hi, fn(hi)


def solve(method, ad, problem, inner, flows, sur, phi, x, target, g, steps=100):
    """Calibrate one adversary to the target risk; return everything measured about it."""
    a = ad._split(x)
    c = inner.Counter(problem)
    if method.startswith('KL'):
        r, (_, w) = calibrate(lambda t: inner.kl_dro(problem, phi, *a, t, loss_fn=c.loss),
                              target, 1e-4, 50.0)
        xa = x[draw_from_weights(w, x.shape[0], g)]
    elif method.startswith('Wasserstein'):
        fn = c.loss if 'true' in method else sur
        r, out = calibrate(lambda e: inner.wasserstein_dro(problem, phi, *a, e, steps=steps,
                                                           loss_fn=fn), target, 1e-4, 50.0)
        xa = ad.pack((out[1], a[1])) if ad.labelled else out[1]
    else:
        r, out = calibrate(lambda t: inner.latent_dro(problem, flows, phi, t, n=x.shape[0],
                                                      steps=steps, loss_fn=sur),
                           target, 1e-4, 50.0)
        xa = ad.pack(out[1:3]) if ad.labelled else out[1]
    b = ad._split(xa)
    with torch.no_grad():
        truth = float(ad.risk(phi, xa))
        claim = float(sur(phi, *b).mean())
    return dict(radius=float(r), evals=c.n, risk=truth, claim=claim,
                error=claim - truth, plaus=float(inner.plausibility(problem, *b)))


if __name__ == '__main__':
    class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter,
                       argparse.RawDescriptionHelpFormatter):
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--problem', default='short_column', choices=sorted(PROBLEMS))
    ap.add_argument('--ratio', type=float, default=1.25)
    ap.add_argument('--n', type=int, default=None, help="default: problem's n_test, else 20000")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)

    spec, problem, inner, flows = setup(a.problem)
    a.n = a.n or spec.get('n_test', 20000)
    # a black-box simulator has no input gradient: no transport adversary on the truth
    methods = [m for m in METHODS
               if not (spec.get('black_box') and m == 'Wasserstein (true loss)')]
    ad = Adapter(problem, spec['labelled'], spec['bounds'])
    from run_inner_cost import load_surrogate      # the problem folder is on the path now
    sur = load_surrogate(os.path.join(ROOT, a.problem, 'models', 'surrogate.pt'), problem)
    phi = torch.load(os.path.join(ROOT, a.problem, 'models', 'design.pt'),
                     weights_only=False)
    g = torch.Generator().manual_seed(0)
    x = ad.sample(a.n, g)
    nom = float(ad.risk(phi, x))
    target = nom + (a.ratio - 1.0) * abs(nom)
    base_plaus = float(inner.plausibility(problem, *ad._split(x)))
    base_err = float(sur(phi, *ad._split(x)).mean()) - nom
    print(f'{a.problem}: nominal risk {nom:.4f} -> target {target:.4f};  '
          f'on nominal data the surrogate errs by {base_err:+.4f} and log-density is '
          f'{base_plaus:.3f}\n')

    rows = {}
    for m in methods:
        rows[m] = solve(m, ad, problem, inner, flows, sur, phi, x, target, g)
        r = rows[m]
        print(f'  {m:26s} r={r["radius"]:<9.4g} risk {r["risk"]:8.4f}  '
              f'evals {r["evals"]:>11,d}  surrogate error {r["error"]:+8.4f}  '
              f'log-density {r["plaus"]:9.3f}')

    with open(os.path.join(HERE, f'{a.problem}.json'), 'w') as f:
        json.dump({'problem': a.problem, 'nominal': nom, 'base_plaus': base_plaus,
                   'base_err': base_err, 'rows': rows}, f, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    names = methods
    xs = np.arange(len(names))
    axes[0].bar(xs, [max(rows[m]['evals'], 0.5) for m in names],
                color=[COLOR[m] for m in names])
    axes[0].set(yscale='log', ylabel='true-loss evaluations', title='what it costs',
                xticks=xs, xticklabels=[])
    axes[1].bar(xs, [rows[m]['error'] for m in names], color=[COLOR[m] for m in names])
    axes[1].axhline(base_err, color='.4', lw=1, ls=':')
    axes[1].set(ylabel='surrogate claim - truth', xticks=xs, xticklabels=[],
                title='does the surrogate still hold there?')
    axes[2].bar(xs, [base_plaus - rows[m]['plaus'] for m in names],
                color=[COLOR[m] for m in names])
    axes[2].set(ylabel='nats below the real data', xticks=xs, xticklabels=[],
                title='how plausible are its points?')
    for ax in axes:
        ax.set_xticklabels(names, rotation=18, ha='right', fontsize=7)
    fig.suptitle(f'{a.problem} -- every adversary calibrated to risk {target:.4f}',
                 fontsize=10)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f'compare_{a.problem}.png'), dpi=150)
    print(f'  wrote {OUT}/compare_{a.problem}.png')
