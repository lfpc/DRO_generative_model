"""Do the surrogate's design gradient and Hessian converge to the true ones?

    python3 optb/gradcheck.py --problem short_column
    python3 optb/gradcheck.py --problem beam_shield --quick
    python3 optb/gradcheck.py --problem two_moons --plot_only

At one analysis design (the initial design moved by OFFSET), on toy problems with a
differentiable loss:
  truth      autograd gradient of the mean loss on a large fixed sample; Hessian by central
             differences of it on the same sample (checked against twice the step).
  surrogate  fitted to per-sample losses at designs around the analysis design, as one
             optb.lcso round does, and differentiated there.
All in u = (phi - phi_0) / half-width of the design box.

Swept one at a time: budget (total samples), delta (half-width of the design cloud, in u),
factor (resample: designs x f, samples / f). Arms: {taylor, taylor-full, taylor-res,
taylor-res:0.1, deeponet} x {resample, nominal}, same optimiser steps and rows per step.
'pathwise MC' = the same budget spent at phi_0 with dloss/dphi available (a floor).

Metrics: grad err ||g_hat - g|| / ||g||, hess err (Frobenius), eig align |cos| of the top
|eigenvalue| eigenvectors, worst = true rise along the surrogate's trust-region worst
direction at --radius over the true worst rise.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
sys.path[:0] = [ROOT, os.path.join(ROOT, 'outer')]
from dro.method import trust_region_max                           # noqa: E402
from optb.lcso import _feasible, _fit, _make_surrogate            # noqa: E402
from optb.samplers import draw_designs                            # noqa: E402
from run_outer import PROBLEMS, Adapter                           # noqa: E402

SURROGATES = ('taylor', 'taylor-full', 'taylor-res', 'taylor-res:0.1', 'deeponet')
SCHEMES = ('resample', 'nominal')
METRICS = (('grad_err', 'gradient rel. error', 'log'),
           ('hess_err', 'Hessian rel. error (Frobenius)', 'log'),
           ('eig_align', r'top-eigenvector $|\cos|$', 'linear'),
           ('worst', 'worst-case captured (true)', 'linear'))
# differentiable loss and an interior start (not the simplex portfolio or the newsvendor)
CHOICES = ('short_column', 'beam_shield', 'two_moons')
# Analysis design = initial design + OFFSET x a fixed random direction (box half-widths).
# Short column: at its initial design all columns are equal and the expected loss has a kink
# (the minimum limit state switches), so there is no Hessian; at 0.4 it is smooth.
OFFSET = {'short_column': 0.4, 'beam_shield': 0.05, 'two_moons': 0.05}


def load(name):
    """The problem behind run_outer's Adapter, loaded by file path (every folder has a problem.py)."""
    spec = PROBLEMS[name]
    s = importlib.util.spec_from_file_location(f'{name}_problem',
                                               os.path.join(ROOT, name, 'problem.py'))
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return Adapter(getattr(mod, spec['cls'])(**spec['kwargs']), spec['labelled'],
                   spec['bounds'])


def analysis_design(ad, offset, seed=0):
    """The initial design moved by `offset` half-widths in a fixed random direction (two moons
    starts at exactly zero, a symmetric special point)."""
    g = torch.Generator().manual_seed(seed)
    half = (ad.hi - ad.lo) / 2
    u = torch.rand(ad.dim, generator=g) * 2 - 1
    return ad.project(ad.init_design() + offset * u * half)


# ---------------------------------------------------------------------------
# The truth
# ---------------------------------------------------------------------------
def mean_loss(ad, phi, x, chunk=200_000):
    with torch.no_grad():
        return sum(float(ad.loss(phi, x[i:i + chunk]).double().sum())
                   for i in range(0, x.shape[0], chunk)) / x.shape[0]


def pathwise_grad_u(ad, phi, x, half, chunk=200_000):
    """d/du of the sample-mean loss, by autograd."""
    p = phi.detach().clone().requires_grad_(True)
    g = torch.zeros(ad.dim, dtype=torch.float64)
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        gb, = torch.autograd.grad(ad.loss(p, xb).sum(), p)
        g += gb.double()
    return (g / x.shape[0]) * half.double()


def pathwise_derivatives(ad, phi, x, half, h):
    """(g_u, H_u) on the sample x; Hessian by central differences of the gradient, symmetrised."""
    g = pathwise_grad_u(ad, phi, x, half)
    cols = []
    for i in range(ad.dim):
        e = torch.zeros(ad.dim, dtype=phi.dtype)
        e[i] = h * half[i]
        cols.append((pathwise_grad_u(ad, phi + e, x, half)
                     - pathwise_grad_u(ad, phi - e, x, half)) / (2 * h))
    H = torch.stack(cols, 1)
    return g, 0.5 * (H + H.T)


# ---------------------------------------------------------------------------
# The surrogate
# ---------------------------------------------------------------------------
def surrogate_derivatives(model, phi, x, half, chunk=65536):
    """(g_u, H_u) of the surrogate's predicted objective, mean over x, at phi.

    Taylor models: design_derivatives, chunked over x. Branch-trunk (identity link): the
    prediction is <b(phi), mean t(x)> + beta, so autograd through the branch alone."""
    k = model.y_scale
    n = x.shape[0]
    with torch.no_grad():
        if hasattr(model, 'Q_raw'):
            g = torch.zeros(model.D, dtype=torch.float64)
            H = torch.zeros(model.D, model.D, dtype=torch.float64)
            for i in range(0, n, chunk):
                xb = x[i:i + chunk]
                _, gb, Hb = model.design_derivatives(phi, xb)
                g += gb.double() * xb.shape[0] / n
                H += Hb.double() * xb.shape[0] / n
        else:
            tbar = sum(model.trunk((x[i:i + chunk] - model.input_loc) / model.input_scale)
                       .double().sum(0) for i in range(0, n, chunk)) / n
    if not hasattr(model, 'Q_raw'):
        tbar = tbar.to(phi.dtype)

        def f(p):
            return model.branch((p - model.design_loc) / model.design_scale) @ tbar

        p = phi.detach().clone()
        g = torch.autograd.functional.jacobian(f, p).double()
        H = torch.autograd.functional.hessian(f, p).double()
    hd = half.double()
    return k * g * hd, k * H * hd[:, None] * hd[None, :]


def simulate(ad, designs, m, g):
    """m fresh inputs per design, and their per-sample losses."""
    X = torch.stack([ad.sample(m, g) for _ in range(designs.shape[0])])
    with torch.no_grad():
        Y = torch.stack([ad.loss(designs[i], X[i]) for i in range(designs.shape[0])])
    return X, Y


def fit_local(ad, phi, surrogate, scheme, budget, delta, n_samples, factor, steps, seed,
              rank=8, sampler='sobol', refresh_frac=0.5, rows=65536, min_per_design=10):
    """One surrogate fitted around phi on `budget` simulated samples -> (model, designs).

    nominal: budget / n_samples designs, fitted once. resample: rounds of 4D designs with
    n_samples / factor each, in optb.lcso's rotating buffer."""
    g = torch.Generator().manual_seed(1000 + seed)
    D, lo, hi = ad.dim, ad.lo, ad.hi
    half = (hi - lo) / 2
    # the centred Taylor models are full rank: the rank-8 floor is taylor's alone
    rank = None if surrogate.split(':')[0] in ('taylor-full', 'taylor-res') else rank
    model, opt = _make_surrogate(ad, surrogate, seed, rank=rank), None

    def draw(k):
        u = draw_designs(sampler, k, D, g, radial=True).to(phi.dtype)
        d = _feasible(ad, phi + delta * u * half, lo, hi).detach()
        d[0] = phi                                   # pin the analysis design, as lcso does
        return d

    def batch(M, N):
        # same number of (design, sample) rows per optimiser step for every arm
        return max(8, min(N, rows // M))

    if scheme == 'nominal':
        M = max(2, budget // n_samples)
        designs = draw(M)
        X, Y = simulate(ad, designs, n_samples, g)
        model, _ = _fit(model, designs, X, Y, steps, seed, batch=batch(M, n_samples),
                        link=ad.link)
        return model, M

    m = max(min_per_design, n_samples // factor)
    k = max(2, min(4 * D, budget // m))
    rounds = max(1, budget // (k * m))
    per_round = max(1, steps // rounds)
    bd = bx = by = None
    for r in range(rounds):
        nd = draw(k)
        nx, ny = simulate(ad, nd, m, g)
        if bd is None:
            bd, bx, by = nd, nx, ny
        else:
            keep = int((1.0 - refresh_frac) * bd.shape[0])
            bd, bx, by = (torch.cat([t[-keep:], n]) if keep else n
                          for t, n in ((bd, nd), (bx, nx), (by, ny)))
        model, opt = _fit(model, bd, bx, by, per_round, seed + r,
                          batch=batch(bd.shape[0], m), link=ad.link, opt=opt)
    return model, rounds * k


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def worst_direction(g, H, radius):
    """argmax over ||v|| <= radius of g'v + v'Hv/2."""
    v, _, _ = trust_region_max(g, H, radius)
    return v


def score(g_hat, H_hat, truth, ad, phi, half, x_true, radius):
    g, H, v_true, rise_true = truth['g'], truth['H'], truth['v'], truth['rise']
    ev, Q = torch.linalg.eigh(H)
    ev_h, Q_h = torch.linalg.eigh(0.5 * (H_hat + H_hat.T))
    top, top_h = Q[:, ev.abs().argmax()], Q_h[:, ev_h.abs().argmax()]
    v = worst_direction(g_hat, H_hat, radius)
    rise = mean_loss(ad, phi + v.to(phi.dtype) * half, x_true) - truth['f0']
    return dict(grad_err=float((g_hat - g).norm() / g.norm()),
                grad_cos=float(g_hat @ g / (g_hat.norm() * g.norm()).clamp_min(1e-300)),
                hess_err=float((H_hat - H).norm() / H.norm()),
                eig_align=float((top @ top_h).abs()),
                worst=rise / rise_true,
                worst_dir_cos=float(v @ v_true / (v.norm() * v_true.norm()).clamp_min(1e-300)))


def reference(ad, phi, half, x_true, radius, h):
    """The truth, with a check of the FD Hessian against twice the step."""
    g, H = pathwise_derivatives(ad, phi, x_true, half, h)
    _, H2 = pathwise_derivatives(ad, phi, x_true, half, 2 * h)
    f0 = mean_loss(ad, phi, x_true)
    v = worst_direction(g, H, radius)
    rise = mean_loss(ad, phi + v.to(phi.dtype) * half, x_true) - f0
    ev = torch.linalg.eigvalsh(H)
    check = float((H - H2).norm() / H.norm())
    print(f'  truth: |g_u| {float(g.norm()):.4g}  |H_u|_F {float(H.norm()):.4g}  '
          f'eig [{float(ev[0]):+.3g}, {float(ev[-1]):+.3g}]  '
          f'FD step check {check:.2e}  '
          f'worst-case rise at r={radius}: {rise:+.4g} (quadratic model '
          f'{float(g @ v + 0.5 * v @ H @ v):+.4g})')
    if check > 1e-2:
        print(f'  WARNING: the reference Hessian moves by {check:.1%} when the FD step '
              f'doubles; the expected loss is probably not smooth here (see OFFSET).')
    return dict(g=g, H=H, v=v, rise=rise, f0=f0, check=check)


def run_point(ad, phi, half, truth, x_true, x_int, a, surrogate, scheme, budget, delta,
              factor, seed):
    t0 = time.time()
    model, n_des = fit_local(ad, phi, surrogate, scheme, budget, delta, a.n_samples,
                             factor, a.steps, seed, rank=a.rank)
    g_hat, H_hat = surrogate_derivatives(model, phi, x_int, half)
    out = score(g_hat, H_hat, truth, ad, phi, half, x_true, a.radius)
    out.update(designs=n_des, seconds=time.time() - t0)
    return out


def oracle_point(ad, phi, half, truth, x_true, a, budget, seed):
    g = torch.Generator().manual_seed(5000 + seed)
    x = ad.sample(budget, g)
    gh, Hh = pathwise_derivatives(ad, phi, x, half, a.h)
    return score(gh, Hh, truth, ad, phi, half, x_true, a.radius)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COLOR = {'taylor': '#2a78d6', 'deeponet': '#eb6834', 'taylor-res': '#1baf7a',
         'taylor-full': '#eda100', 'taylor-res:0.1': '#7b4fc9'}
STYLE = {'resample': '-', 'nominal': '--'}
SWEEPS = (('budget', 'simulated samples', 'log'),
          ('delta', r'trust-region half-width $\delta$ (normalised)', 'log'),
          ('factor', 'resample factor (designs x f, samples / f)', 'log'))


def summarise(rows, sweep, arm, metric):
    sel = [r for r in rows if r['sweep'] == sweep and r['arm'] == arm]
    xs = sorted({r[sweep] for r in sel})
    med, lo, hi = [], [], []
    for x in xs:
        v = np.array([r[metric] for r in sel if r[sweep] == x])
        med.append(np.median(v))
        lo.append(v.min())
        hi.append(v.max())
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)


def arms_of(rows):
    """(surrogate, scheme) pairs present in rows, in SURROGATES order."""
    have = {r['arm'] for r in rows}
    return [(s, k) for s in SURROGATES for k in SCHEMES if f'{s}/{k}' in have]


def plot(res, path):
    rows, d = res['rows'], res['defaults']
    ARMS = arms_of(rows)
    fig, axes = plt.subplots(len(METRICS), len(SWEEPS), figsize=(13, 11.5),
                             constrained_layout=True, sharex='col')
    for j, (sweep, xlabel, xscale) in enumerate(SWEEPS):
        for i, (metric, ylabel, yscale) in enumerate(METRICS):
            ax = axes[i, j]
            for s, k in ARMS:
                arm = f'{s}/{k}'
                if sweep == 'factor' and k == 'nominal':
                    continue
                xs, med, lo, hi = summarise(rows, sweep, arm, metric)
                if not len(xs):
                    continue
                ax.plot(xs, med, STYLE[k], color=COLOR[s], lw=2, marker='o', ms=4,
                        label=f'{s} ({k})')
                ax.fill_between(xs, lo, hi, color=COLOR[s], alpha=0.12, lw=0)
            xs, med, lo, hi = summarise(rows, sweep, 'pathwise MC', metric)
            if len(xs) > 1:
                ax.plot(xs, med, ':', color='0.35', lw=1.6, label='pathwise MC (oracle)')
            elif len(xs):
                ax.axhline(med[0], ls=':', color='0.35', lw=1.2, label='pathwise MC (oracle)')
            if metric in ('eig_align', 'worst'):
                ax.axhline(1.0, color='0.75', lw=0.8, zorder=0)
            ax.axvline(d[sweep], color='0.85', lw=0.8, zorder=0)
            ax.set(xscale=xscale, yscale=yscale)
            ax.grid(alpha=0.25, lw=0.5)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(f'varying {sweep}', fontsize=10)
            if i == len(METRICS) - 1:
                ax.set_xlabel(xlabel)
    axes[0, 0].legend(fontsize=7.5, loc='best')
    fig.suptitle(f'{res["problem"]} (D = {res["dim"]}): surrogate derivatives at a fixed '
                 f'design vs the truth.  Defaults: budget {d["budget"]:,}, '
                 f'delta {d["delta"]}, factor {d["factor"]}, n_samples {res["n_samples"]}; '
                 f'median and range over {res["seeds"]} seeds', fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  wrote {path}')


def table(res):
    """Every arm and metric at the default point."""
    rows, d = res['rows'], res['defaults']
    ARMS = arms_of(rows)
    print(f'\n  at the defaults (budget {d["budget"]:,}, delta {d["delta"]}, factor '
          f'{d["factor"]}); median over seeds')
    print('  %-22s %9s %9s %9s %9s %9s %8s' % ('', 'grad err', 'grad cos', 'hess err',
                                             'eig |cos|', 'worst', 'designs'))
    for arm in [f'{s}/{k}' for s, k in ARMS] + ['pathwise MC']:
        sel = [r for r in rows if r['sweep'] == 'budget' and r['arm'] == arm
               and r['budget'] == d['budget']]
        if not sel:
            continue
        m = {k: float(np.median([r[k] for r in sel])) for k in
             ('grad_err', 'grad_cos', 'hess_err', 'eig_align', 'worst')}
        des = int(np.median([r.get('designs', 0) for r in sel]))
        print('  %-22s %9.3f %9.3f %9.3f %9.3f %9.3f %8d' % (
            arm, m['grad_err'], m['grad_cos'], m['hess_err'], m['eig_align'], m['worst'],
            des))


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter,
                   argparse.RawDescriptionHelpFormatter):
    pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--problem', default='short_column', choices=CHOICES)
    ap.add_argument('--budgets', nargs='+', type=int,
                    default=[25_000, 100_000, 400_000, 1_600_000], help='budget sweep')
    ap.add_argument('--budget', type=int, default=400_000,
                    help='fixed budget of the other sweeps (must be in --budgets)')
    ap.add_argument('--deltas', nargs='+', type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument('--delta', type=float, default=0.3, help='fixed delta of the other sweeps')
    ap.add_argument('--factors', nargs='+', type=int, default=[1, 2, 8, 32, 128])
    ap.add_argument('--factor', type=int, default=8, help='fixed factor of the other sweeps')
    ap.add_argument('--n_samples', type=int, default=2000,
                    help='samples per design (nominal); resample uses n_samples / factor')
    ap.add_argument('--steps', type=int, default=1500, help='optimiser steps per fit')
    ap.add_argument('--rank', type=int, default=8, help='rank of taylor')
    ap.add_argument('--surrogates', nargs='+', default=list(SURROGATES), choices=SURROGATES)
    ap.add_argument('--merge', action='store_true',
                    help='keep saved rows of arms not run now (and the oracle rows)')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--radius', type=float, default=0.1, help='worst-case radius, in u')
    ap.add_argument('--offset', type=float, default=None, help='default: OFFSET[problem]')
    ap.add_argument('--n_true', type=int, default=1_000_000, help='inputs for the truth')
    ap.add_argument('--n_int', type=int, default=200_000,
                    help='inputs the surrogate is averaged over')
    ap.add_argument('--h', type=float, default=2e-3, help='FD step of the true Hessian, in u')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--quick', action='store_true', help='smoke test')
    ap.add_argument('--plot_only', action='store_true', help='re-plot the saved JSON')
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    tag = a.problem + ('_quick' if a.quick else '')
    js, fig = (os.path.join(OUT, f'gradcheck_{tag}.{e}') for e in ('json', 'png'))

    if a.plot_only:
        res = json.load(open(js))
        table(res)
        plot(res, fig)
        sys.exit()
    if a.quick:
        a.seeds, a.steps, a.n_true, a.n_int = 1, 400, 200_000, 50_000
        a.budgets, a.deltas, a.factors = [25_000, 200_000], [0.1, 1.0], [1, 32]
        a.budget = 200_000

    ad = load(a.problem)
    a.offset = OFFSET[a.problem] if a.offset is None else a.offset
    phi = analysis_design(ad, a.offset)
    half = (ad.hi - ad.lo) / 2
    x_true = ad.sample(a.n_true, torch.Generator().manual_seed(7))
    x_int = x_true[:a.n_int]
    print(f'{a.problem}: D = {ad.dim}, truth on {a.n_true:,} inputs')
    truth = reference(ad, phi, half, x_true, a.radius, a.h)

    points = ([('budget', b, a.delta, a.factor) for b in a.budgets]
              + [('delta', a.budget, dl, a.factor) for dl in a.deltas]
              + [('factor', a.budget, a.delta, f) for f in a.factors])
    ARMS = [(s, k) for s in a.surrogates for k in SCHEMES]
    run = {f'{s}/{k}' for s, k in ARMS}
    kept = []
    if a.merge and os.path.exists(js):
        kept = [r for r in json.load(open(js))['rows'] if r['arm'] not in run]
        print(f'  merging: keeping {len(kept)} saved rows of '
              f'{sorted({r["arm"] for r in kept})}')
    have_oracle = any(r['arm'] == 'pathwise MC' for r in kept)
    rows = []
    for seed in range(a.seeds):
        for b in ([] if have_oracle else a.budgets):
            r = oracle_point(ad, phi, half, truth, x_true, a, b, seed)
            rows.append(dict(sweep='budget', arm='pathwise MC', budget=b, delta=a.delta,
                             factor=a.factor, seed=seed, **r))
        for sweep, b, dl, f in points:
            for s, k in ARMS:
                if sweep == 'factor' and k == 'nominal':
                    continue              # the factor does not enter the nominal scheme
                r = run_point(ad, phi, half, truth, x_true, x_int, a, s, k, b, dl, f, seed)
                rows.append(dict(sweep=sweep, arm=f'{s}/{k}', budget=b, delta=dl, factor=f,
                                 seed=seed, **r))
                print(f'  seed {seed} {sweep:6s} B={b:>9,} delta={dl:<5g} f={f:<4d} '
                      f'{s:11s} {k:8s} designs {r["designs"]:5d}  grad err '
                      f'{r["grad_err"]:7.3f}  hess err {r["hess_err"]:7.3f}  eig '
                      f'{r["eig_align"]:5.2f}  worst {r["worst"]:+6.2f}  '
                      f'({r["seconds"]:.0f}s)', flush=True)
        all_rows = kept + rows
        res = dict(problem=a.problem, dim=ad.dim, n_samples=a.n_samples, seeds=seed + 1,
                   radius=a.radius, offset=a.offset,
                   defaults=dict(budget=a.budget, delta=a.delta, factor=a.factor),
                   truth=dict(grad_norm=float(truth['g'].norm()),
                              hess_eigs=torch.linalg.eigvalsh(truth['H']).tolist(),
                              worst_rise=truth['rise'], fd_check=truth['check']),
                   rows=all_rows)
        json.dump(res, open(js, 'w'), indent=1)       # after every seed
    table(res)
    plot(res, fig)
