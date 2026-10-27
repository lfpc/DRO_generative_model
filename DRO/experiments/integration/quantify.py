"""Is randomised QMC better than plain MC on the generator, and by how much?

Reports, against the converged value: the equal-N efficiency gain (RMSE_MC / RMSE_QMC)^2
with a bootstrap CI (= factor fewer loss evaluations), the convergence rate (slope of
log RMSE vs log N; MC is -1/2), the coverage of nominal 95% t-intervals from 8
randomisations, and the bias in standard errors.

    python3 quantify.py --problem two_moons
    python3 quantify.py --problem muon_shield --loss surrogate --sizes 1024 4096 16384
"""
import argparse
import importlib.util
import os

import numpy as np
import torch

# quasi-MC.py has a hyphen in its name, so it is loaded by path
q = importlib.util.spec_from_file_location(
    'q', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quasi-MC.py'))
qmc = importlib.util.module_from_spec(q)
q.loader.exec_module(qmc)


def replicates(sampler, sizes, reps, loss):
    """est[i, r]: the estimate at sizes[i] from randomisation r. Only `loss` is evaluated."""
    return np.array([[qmc.estimate(n, sampler, seed=100003 * r + n, losses=(loss,))[loss]
                      for r in range(reps)] for n in sizes])


def rmse(est, gold):
    return np.sqrt(((est - gold) ** 2).mean(1))


def fit_rate(sizes, est, gold, lo):
    """Slope of log RMSE vs log N, over the sizes in the asymptotic range."""
    m = np.asarray(sizes) >= lo
    return np.polyfit(np.log(np.asarray(sizes)[m]), np.log(rmse(est, gold)[m]), 1)[0]


# Student-t 0.975 quantiles, keyed by k (k-1 degrees of freedom)
T975 = {4: 3.1824, 6: 2.5706, 8: 2.3646, 12: 2.2010, 16: 2.1314, 24: 2.0687}


def coverage(est, gold, k=8, draws=4000, seed=0):
    """Per size, how often mean +- t * se over k resampled randomisations contains gold."""
    rng = np.random.default_rng(seed)
    t = T975[k]
    out = []
    for row in est:
        idx = rng.integers(0, row.shape[0], size=(draws, k))
        s = row[idx]
        half = t * s.std(1, ddof=1) / np.sqrt(k)
        out.append(float((np.abs(s.mean(1) - gold) <= half).mean()))
    return np.array(out)


if __name__ == '__main__':
    class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter,
                       argparse.RawDescriptionHelpFormatter):
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--problem', default='two_moons', choices=sorted(qmc.PROBLEMS))
    ap.add_argument('--sizes', nargs='+', type=int, default=None,
                    help="default: problem's, else 256 1024 4096 16384 65536")
    ap.add_argument('--reps', type=int, default=48, help='independent randomisations')
    ap.add_argument('--boot', type=int, default=4000, help='bootstrap resamples')
    ap.add_argument('--loss', default='true', choices=list(qmc.STYLE))
    ap.add_argument('--gold_n', type=int, default=None,
                    help="Sobol points per scrambling of the reference; default: problem's, "
                         "else 2**19")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)

    spec = qmc.PROBLEMS[a.problem]
    a.sizes = a.sizes or spec.get('sizes', [256, 1024, 4096, 16384, 65536])
    models, _ = qmc.setup(a.problem)
    model = qmc.Generator.load(os.path.join(models, 'generator.pt'))
    gold = qmc.converged(model, n=a.gold_n or spec.get('gold_n', 2 ** 19),
                         losses=(a.loss,))[a.loss]
    print(f'{a.problem}: latent dim {model.latent_dim}, {a.reps} randomisations, '
          f'{a.loss} loss, converged value {gold:.6f}\n')

    methods = {
        qmc.MC_MODEL: lambda k, generator=None: qmc.sample_from_model(model, k, generator),
        qmc.QMC: lambda k, generator=None: qmc.sample_from_model(model, k, generator,
                                                                 qmc=True),
    }
    est = {m: replicates(s, a.sizes, a.reps, a.loss) for m, s in methods.items()}

    hdr = ' '.join('%11d' % n for n in a.sizes)
    print('%-28s %s' % ('', hdr))
    for m, e in est.items():
        print('%-28s %s' % (f'{m} RMSE', ' '.join('%11.2e' % v for v in rmse(e, gold))))

    # equal-N efficiency, with a bootstrap over the randomisations of both methods
    mc, qm = est[qmc.MC_MODEL], est[qmc.QMC]
    gain = (rmse(mc, gold) / rmse(qm, gold)) ** 2
    rng = np.random.default_rng(1)
    bi = rng.integers(0, a.reps, size=(a.boot, a.reps))
    bg = np.array([(rmse(mc[:, i], gold) / rmse(qm[:, i], gold)) ** 2 for i in bi])
    lo, hi = np.percentile(bg, [2.5, 97.5], axis=0)
    print('%-28s %s' % ('efficiency gain (x)',
                        ' '.join('%11.1f' % v for v in gain)))
    print('%-28s %s' % ('  95% bootstrap CI',
                        ' '.join('%11s' % f'[{l:.0f},{h:.0f}]' for l, h in zip(lo, hi))))

    print()
    cut = a.sizes[len(a.sizes) // 2]           # fit the rate on the asymptotic half only
    for m, e in est.items():
        r = fit_rate(a.sizes, e, gold, cut)
        br = np.array([fit_rate(a.sizes, e[:, i], gold, cut) for i in bi])
        cov = coverage(e, gold)
        b = np.abs(e.mean(1) - gold) / (e.std(1, ddof=1) / np.sqrt(a.reps))
        print(f'{m:28s} rate {r:+.2f} [{np.percentile(br, 2.5):+.2f}, '
              f'{np.percentile(br, 97.5):+.2f}]   coverage {cov.mean():.3f}   '
              f'max |bias|/se {b.max():.1f}')
    print('\n  rate is fitted for N >= %d; plain MC is pinned at -0.50 by the CLT.' % cut)
    print('  coverage should be 0.95; bias/se above ~2.5 at any N means the mean is off.')
