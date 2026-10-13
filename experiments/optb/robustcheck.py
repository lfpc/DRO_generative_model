"""Does the robustness analysis read off a surrogate reach the right conclusions?

    python3 optb/robustcheck.py --problem short_column
    python3 optb/robustcheck.py --problem beam_shield --quick
    python3 optb/robustcheck.py --problem two_moons --plot_only

Each question robustness.py answers from the surrogate is answered from every surrogate
and checked against the simulator on a large common sample:

  ranking     one-at-a-time rise per parameter (to a face of the tolerance box): Spearman
              rho and top-3 overlap with the simulated ranking.
  worst case  simulated rise along ascent (+grad), worst (trust-region maximiser at
              ||w|| = 1), hess1 (top Hessian eigenvector), act1 (top active-subspace
              direction), random, over the best simulated rise on the unit ball; corner
              (sign(grad)) and pga (ascent on the surrogate) over the best rise in the box.
              cal = predicted rise / simulated rise.
  Hessian     H gain = capture(worst) - capture(ascent); curv top/bot = true curvature along
              the surrogate's extreme eigenvectors over the true extreme eigenvalue.
  active sub. share of the true C = E[grad l grad l'] top-k energy in the surrogate's top-k.
  box         Spearman rho and 95th-percentile ratio over designs uniform in the box.

Points: init (initial design + OFFSET, gradient-dominated) and opt (the nominal optimum,
curvature-dominated). Frame: w = (phi - phi_a) / (delta * half-width), box = [-1, 1]^D,
which is also the surrogate's training region. 'pathwise MC' = gradient, Hessian and
per-sample gradients from the same budget at phi_a with dloss/dphi (a floor).
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradcheck import (OFFSET, OUT, analysis_design, fit_local, load,   # noqa: E402
                       mean_loss, pathwise_derivatives, surrogate_derivatives)
from dro.method import trust_region_max                                  # noqa: E402
from run_outer import PROBLEMS                                           # noqa: E402

SURROGATES = ('taylor', 'taylor-full', 'taylor-res', 'taylor-res:0.1', 'deeponet')
ORACLE = 'pathwise MC'
POINTS = ('init', 'opt')
BALL_DIRS = ('ascent', 'worst', 'hess1', 'act1', 'random')
BOX_DIRS = ('corner', 'pga')
COLOR = {'taylor': '#2a78d6', 'taylor-full': '#eda100', 'taylor-res': '#1baf7a',
         'taylor-res:0.1': '#7b4fc9', 'deeponet': '#eb6834', ORACLE: '0.35'}
CHOICES = ('short_column', 'beam_shield', 'two_moons', 'newsvendor')
OFFSET = dict(OFFSET, newsvendor=0.05)


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter,
                   argparse.RawDescriptionHelpFormatter):
    pass


def unit(v):
    return v / v.norm().clamp_min(1e-30)


# ---------------------------------------------------------------------------
# The analysis point and the frame
# ---------------------------------------------------------------------------
def optimum(ad, n=200_000):
    """The problem's own nominal optimum, as run_outer calibrates at."""
    g = torch.Generator().manual_seed(11)
    p = ad.p
    if hasattr(p, 'solve'):
        phi = p.solve(p.sample(n, g))
    else:
        phi = p.fit(*p.sample(n, g))
    return ad.project(phi.detach().float())


class Frame:
    """w <-> phi around the analysis design; the tolerance box is w in [-1, 1]^D."""

    def __init__(self, ad, phi, delta):
        self.ad, self.phi = ad, phi
        self.half = (ad.hi - ad.lo) / 2
        self.s = delta * self.half

    def __call__(self, w):
        """Designs for w (D,) or (M, D); clamped to the problem's bounds, differentiably."""
        p = self.phi + w * self.s
        return torch.maximum(torch.minimum(p, self.ad.hi), self.ad.lo)

    def to_w(self, g_u, H_u, delta):
        """gradcheck's u-frame (u = (phi - phi_a) / half) to w = u / delta."""
        return g_u * delta, H_u * delta ** 2


def true_values(ad, F, W, x):
    return np.array([mean_loss(ad, F(w), x) for w in W])


def per_sample_grads(fn, phi, x, s, chunk=20_000):
    """(N, D) per-sample gradients in w of fn(phi, x) -> (N,). Forward mode: memory D x N,
    where reverse mode over N outputs is N x N."""
    out = []
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        J = torch.autograd.functional.jacobian(lambda p: fn(p, xb), phi.detach(),
                                               vectorize=True, strategy='forward-mode')
        out.append(J.detach().double())
    return torch.cat(out) * s.double()


def ascend(f, F, w0s, box, radius=None, steps=150, lr=0.05):
    """Projected ascent of f(design) over w in the box or in the ball ||w|| <= radius,
    from several starts; returns the best w found."""
    best, best_v = None, -float('inf')
    for w0 in w0s:
        w = w0.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(steps):
            v = f(F(w))
            opt.zero_grad()
            (-v).backward()
            opt.step()
            with torch.no_grad():
                if box:
                    w.clamp_(-1, 1)
                else:
                    w.mul_(min(1.0, radius / max(float(w.norm()), 1e-12)))
        with torch.no_grad():
            v = float(f(F(w)))
        if v > best_v:
            best, best_v = w.detach().clone(), v
    return best


# ---------------------------------------------------------------------------
# The truth at one analysis point
# ---------------------------------------------------------------------------
def truth_at(ad, F, a, delta):
    D = ad.dim
    x_true = ad.sample(a.n_true, torch.Generator().manual_seed(7))
    x_mid = x_true[:a.n_mid]
    g_u, H_u = pathwise_derivatives(ad, F.phi, x_true, F.half, a.h)
    _, H2 = pathwise_derivatives(ad, F.phi, x_true, F.half, 2 * a.h)
    g, H = F.to_w(g_u, H_u, delta)
    f0 = mean_loss(ad, F.phi, x_true)
    C = per_sample_grads(ad.loss, F.phi, x_mid[:a.n_as], F.s)
    C = C.T @ C / C.shape[0]
    # one parameter at a time, to each face
    E = torch.eye(D)
    oat = true_values(ad, F, torch.cat([E, -E]), x_true) - f0
    oat = np.maximum(oat[:D], oat[D:])
    # worst on the unit ball and in the box, by ascent on the simulator itself
    f_mid = lambda p: ad.loss(p, x_mid).double().mean()                 # noqa: E731
    gen = torch.Generator().manual_seed(3)
    v_tr = unit(trust_region_max(g, H, 1.0)[0].float())
    starts = [v_tr, unit(g.float())] + [unit(torch.randn(D, generator=gen)) for _ in range(3)]
    w_ball = ascend(f_mid, F, starts, box=False, radius=1.0, steps=300, lr=0.03)
    w_box = ascend(f_mid, F, [torch.sign(g.float()), v_tr * D ** 0.5]
                   + [torch.rand(D, generator=gen) * 2 - 1 for _ in range(3)], box=True)
    ball, box = true_values(ad, F, torch.stack([w_ball, w_box]), x_true) - f0
    # the ascent may lose to the plain TR direction or corner: keep the best of all
    tr_rise, corner_rise = true_values(ad, F, torch.stack([v_tr, torch.sign(g.float())]),
                                       x_true) - f0
    W_mc = torch.rand(a.n_mc, D, generator=torch.Generator().manual_seed(5)) * 2 - 1
    mc = true_values(ad, F, W_mc, x_mid) - mean_loss(ad, F.phi, x_mid)
    ev = torch.linalg.eigvalsh(H)
    check = float((H_u - H2).norm() / H_u.norm())
    print(f'  truth: f0 {f0:.5g}  |g_w| {float(g.norm()):.4g}  eig H_w [{float(ev[0]):+.3g}, '
          f'{float(ev[-1]):+.3g}]  FD check {check:.1e}  worst rise: ball {max(ball, tr_rise):.4g} '
          f'(ascent {ball:.4g}, TR {tr_rise:.4g})  box {max(box, corner_rise):.4g}  '
          f'mc 95% {np.percentile(mc, 95):.4g}', flush=True)
    return dict(x_true=x_true, x_mid=x_mid, g=g, H=H, C=C, f0=f0, oat=oat,
                ball=max(ball, tr_rise), box=max(box, corner_rise), W_mc=W_mc, mc=mc,
                check=check)


# ---------------------------------------------------------------------------
# One analysis from one model
# ---------------------------------------------------------------------------
def analyse(ad, F, T, g, H, C, predict, a, seed):
    """Every robustness.py product from (g, H, C, predict), scored against the truth T.
    predict(W) = predicted f at F(W) minus predicted f at the centre."""
    D = ad.dim
    out = {}
    # ranking
    E = torch.eye(D)
    oat_hat = predict(torch.cat([E, -E]))
    oat_hat = np.maximum(oat_hat[:D], oat_hat[D:])
    rk = lambda v: np.argsort(np.argsort(-v))                            # noqa: E731
    out['rank_spearman'] = float(np.corrcoef(rk(oat_hat), rk(T['oat']))[0, 1])
    out['rank_top3'] = len(set(np.argsort(-oat_hat)[:3]) & set(np.argsort(-T['oat'])[:3])) / 3
    # directions
    gf, Hf = g.float(), H.float()
    ev, Q = torch.linalg.eigh(Hf)
    evC, QC = torch.linalg.eigh(C.float())
    up = lambda v: v if float(v @ gf) >= 0 else -v                       # noqa: E731
    gen = torch.Generator().manual_seed(100 + seed)
    dirs = {'ascent': unit(gf), 'worst': unit(trust_region_max(g, H, 1.0)[0].float()),
            'hess1': up(Q[:, -1]), 'act1': up(QC[:, -1]),
            'random': unit(torch.randn(D, generator=gen)),
            'corner': torch.sign(gf)}
    f_pred = lambda p: torch.as_tensor(predict(p.unsqueeze(0) if p.dim() == 1 else p,   # noqa: E731
                                               designs=True)).sum()
    dirs['pga'] = ascend(f_pred, F, [torch.sign(gf), dirs['worst'] * D ** 0.5,
                                     torch.zeros(D)], box=True)
    names = list(dirs)
    W = torch.stack([dirs[k] for k in names])
    true_rise = true_values(ad, F, W, T['x_true']) - T['f0']
    pred_rise = predict(W)
    for k, tr, pr in zip(names, true_rise, pred_rise):
        out[f'rise_{k}'], out[f'pred_{k}'] = float(tr), float(pr)
    # true curvature along the surrogate's extreme eigenvectors / true extreme eigenvalue,
    # only where that eigenvalue is not negligible
    evt = torch.linalg.eigvalsh(T['H'])
    big = float(evt.abs().max())
    out['curv_top'] = (float(Q[:, -1].double() @ T['H'] @ Q[:, -1].double() / evt[-1])
                       if evt[-1] > 0.05 * big else float('nan'))
    out['curv_bot'] = (float(Q[:, 0].double() @ T['H'] @ Q[:, 0].double() / evt[0])
                       if evt[0] < -0.05 * big else float('nan'))
    # active subspace
    evt_c, _ = torch.linalg.eigh(T['C'])
    for k in (1, 2, 3):
        V = QC[:, -k:].double()
        out[f'as{k}'] = float(torch.trace(V.T @ T['C'] @ V) / evt_c[-k:].sum())
    # the box
    mc_hat = predict(T['W_mc'])
    out['mc_spearman'] = float(np.corrcoef(rk(mc_hat), rk(T['mc']))[0, 1])
    out['mc_q95'] = float(np.percentile(mc_hat, 95) / np.percentile(T['mc'], 95))
    return out


def surrogate_predictor(model, F, x_int):
    k = model.y_scale

    def raw(P):
        return model(P, x_int).double().mean(-1) * k

    with torch.no_grad():
        c = float(raw(F.phi.unsqueeze(0))[0])

    def predict(W, designs=False):
        if designs:                                   # already designs, keep the graph
            return raw(W) - c
        with torch.no_grad():
            return (raw(F(W)) - c).numpy()
    return predict


def quadratic_predictor(g, H, F):
    def predict(W, designs=False):
        if designs:
            W = (W - F.phi) / F.s
        W = W.double()
        v = W @ g + 0.5 * ((W @ H) * W).sum(-1)
        return v if designs else v.detach().numpy()
    return predict


def run_surrogate(ad, F, T, a, surrogate, scheme, delta, seed):
    t0 = time.time()
    model, _ = fit_local(ad, F.phi, surrogate, scheme, a.budget, delta, a.n_samples,
                         a.factor, a.steps, seed)
    x_int = T['x_mid'][:a.n_int]
    g_u, H_u = surrogate_derivatives(model, F.phi, x_int, F.half)
    g, H = F.to_w(g_u, H_u, delta)
    k = model.y_scale
    C = per_sample_grads(lambda p, xb: model(p, xb) * k, F.phi, x_int[:a.n_as], F.s)
    C = C.T @ C / C.shape[0]
    out = analyse(ad, F, T, g, H, C, surrogate_predictor(model, F, x_int), a, seed)
    out['seconds'] = time.time() - t0
    return out


def run_oracle(ad, F, T, a, delta, seed):
    x = ad.sample(a.budget, torch.Generator().manual_seed(5000 + seed))
    g_u, H_u = pathwise_derivatives(ad, F.phi, x, F.half, a.h)
    g, H = F.to_w(g_u, H_u, delta)
    C = per_sample_grads(ad.loss, F.phi, x[:a.n_as], F.s)
    C = C.T @ C / C.shape[0]
    return analyse(ad, F, T, g, H, C, quadratic_predictor(g, H, F), a, seed)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
COLS = (('rank_spearman', 'rank rho'), ('rank_top3', 'top-3'),
        ('cap_ascent', 'ascent'), ('cap_worst', 'TR worst'), ('cap_hess1', 'hess1'),
        ('cap_act1', 'act1'), ('cap_random', 'random'), ('cap_corner', 'corner'),
        ('cap_pga', 'pga'), ('cal_worst', 'cal TR'), ('cal_pga', 'cal pga'),
        ('hess_gain', 'H gain'), ('curv_top', 'curv top'), ('curv_bot', 'curv bot'),
        ('as1', 'AS1'), ('as2', 'AS2'), ('as3', 'AS3'), ('mc_spearman', 'box rho'),
        ('mc_q95', 'box q95'))


def derive(res):
    """Captures against the best simulated rise found at that point by the simulator's own
    ascent or by any arm, and calibration against the simulated rise."""
    for pt, T in res['truth'].items():
        rows = [r for r in res['rows'] if r['point'] == pt]
        ball = max([T['ball']] + [r[f'rise_{k}'] for r in rows for k in BALL_DIRS])
        box = max([T['box'], ball] + [r[f'rise_{k}'] for r in rows for k in BOX_DIRS])
        T['ball_best'], T['box_best'] = ball, box
        for r in rows:
            for k in BALL_DIRS + BOX_DIRS:
                ref = box if k in BOX_DIRS else ball
                tr = r[f'rise_{k}']
                r[f'cap_{k}'] = tr / ref
                r[f'cal_{k}'] = r[f'pred_{k}'] / tr if abs(tr) > 0.05 * ref else float('nan')
            r['hess_gain'] = r['cap_worst'] - r['cap_ascent']
    return res


def arms(rows):
    seen = []
    for r in rows:
        if r['arm'] not in seen:
            seen.append(r['arm'])
    return seen


def med(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float('nan')


def table(res):
    derive(res)
    for pt in POINTS:
        rows = [r for r in res['rows'] if r['point'] == pt]
        if not rows:
            continue
        T = res['truth'][pt]
        print(f'\n  {res["problem"]} at {pt} (delta {res["delta"]}, budget {res["budget"]:,}); '
              f'median over seeds. Captures are simulated rises over the best found: '
              f'{T["ball_best"]:.4g} on the unit ball, {T["box_best"]:.4g} in the box. '
              f'|g_w| {T["g_norm"]:.3g}, eig H_w [{T["eig_min"]:+.3g}, {T["eig_max"]:+.3g}], '
              f'FD check {T["check"]:.1e}' + ('  (NOT SMOOTH: Hessian ill-defined)'
                                              if T['check'] > 1e-2 else ''))
        print('  %-26s' % '' + ''.join('%9s' % c for _, c in COLS))
        for arm in arms(rows):
            rr = [r for r in rows if r['arm'] == arm]
            print('  %-26s' % arm + ''.join('%9.2f' % med(rr, k) for k, _ in COLS))


def plot(res, path):
    derive(res)
    pts = [p for p in POINTS if any(r['point'] == p for r in res['rows'])]
    fig, axes = plt.subplots(len(pts), 3, figsize=(17, 4.3 * len(pts)), squeeze=False,
                             constrained_layout=True)
    groups = (('worst case found (simulated rise / simulated worst)',
               [f'cap_{k}' for k in BALL_DIRS + BOX_DIRS], (-0.6, 1.1)),
              ('Hessian and active subspace',
               ['hess_gain', 'curv_top', 'curv_bot', 'as1', 'as2', 'as3'], (-0.6, 1.1)),
              ('ranking, box and calibration',
               ['rank_spearman', 'rank_top3', 'mc_spearman', 'mc_q95', 'cal_worst',
                'cal_pga'], (-0.6, 2.0)))
    for i, pt in enumerate(pts):
        rows = [r for r in res['rows'] if r['point'] == pt]
        A = arms(rows)
        w = 0.8 / len(A)
        for j, (title, keys, ylim) in enumerate(groups):
            ax = axes[i, j]
            for n, arm in enumerate(A):
                rr = [r for r in rows if r['arm'] == arm]
                x = np.arange(len(keys)) + (n - (len(A) - 1) / 2) * w
                m = [med(rr, k) for k in keys]
                ax.bar(x, m, width=w, color=COLOR.get(arm.split('/')[0], '0.5'),
                       alpha=1.0 if arm.endswith('resample') or arm == ORACLE else 0.5,
                       label=arm)
                for k_, xx in zip(keys, x):
                    v = [r[k_] for r in rr if np.isfinite(r[k_])]
                    ax.plot([xx] * len(v), v, '.', color='0.15', ms=2.5)
            ax.axhline(1, color='0.4', lw=0.8, ls=':')
            ax.axhline(0, color='0.4', lw=0.8)
            ax.set_xticks(np.arange(len(keys)))
            ax.set_xticklabels([dict(COLS).get(k, k) for k in keys], fontsize=8)
            ax.set_ylim(*ylim)
            ax.set_title(f'{pt}: {title}', fontsize=10)
            ax.grid(alpha=0.25, lw=0.5, axis='y')
            if i == 0 and j == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f'{res["problem"]}: the robustness analysis from each surrogate, checked '
                 f'against the simulator (delta {res["delta"]}, budget {res["budget"]:,})',
                 fontsize=11)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'  wrote {path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--problem', default='short_column', choices=CHOICES)
    ap.add_argument('--points', nargs='+', default=list(POINTS), choices=POINTS)
    ap.add_argument('--surrogates', nargs='+', default=list(SURROGATES), choices=SURROGATES)
    ap.add_argument('--schemes', nargs='+', default=['resample'],
                    choices=['resample', 'nominal'])
    ap.add_argument('--budget', type=int, default=400_000, help='simulated samples per fit')
    ap.add_argument('--delta', type=float, default=0.3,
                    help='tolerance box half-width, in design half-widths')
    ap.add_argument('--n_samples', type=int, default=2000)
    ap.add_argument('--factor', type=int, default=8)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--n_true', type=int, default=1_000_000, help='inputs for the truth')
    ap.add_argument('--n_mid', type=int, default=200_000,
                    help='inputs for the simulator ascent and the box MC')
    ap.add_argument('--n_int', type=int, default=50_000,
                    help='inputs the surrogate is averaged over')
    ap.add_argument('--n_as', type=int, default=50_000, help='inputs for the active subspace')
    ap.add_argument('--n_mc', type=int, default=256, help='designs uniform in the box')
    ap.add_argument('--h', type=float, default=2e-3, help='FD step of the true Hessian')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--quick', action='store_true', help='smoke test')
    ap.add_argument('--merge', action='store_true',
                    help='keep saved rows of arms not run now')
    ap.add_argument('--plot_only', action='store_true', help='re-plot the saved JSON')
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(a.threads)
    tag = (a.problem + (f'_d{a.delta:g}' if a.delta != 0.3 else '')
           + ('_quick' if a.quick else ''))
    js, fig = (os.path.join(OUT, f'robustcheck_{tag}.{e}') for e in ('json', 'png'))
    if a.plot_only:
        res = json.load(open(js))
        table(res)
        plot(res, fig)
        sys.exit()
    if a.quick:
        a.seeds, a.steps, a.budget, a.n_true, a.n_mid = 1, 300, 100_000, 200_000, 50_000
        a.n_int, a.n_as, a.n_mc = 20_000, 10_000, 64

    ad = load(a.problem)
    arms_now = [f'{s}/{k}' for s in a.surrogates for k in a.schemes] + [ORACLE]
    kept = []
    if a.merge and os.path.exists(js):
        kept = [r for r in json.load(open(js))['rows'] if r['arm'] not in arms_now]
    rows, truth = [], {}
    if a.merge and os.path.exists(js):
        truth = json.load(open(js)).get('truth', {})
    for pt in a.points:
        phi = (analysis_design(ad, OFFSET[a.problem]) if pt == 'init'
               else optimum(ad))
        F = Frame(ad, phi, a.delta)
        print(f'{a.problem} at {pt}: D = {ad.dim}', flush=True)
        T = truth_at(ad, F, a, a.delta)
        ev = torch.linalg.eigvalsh(T['H'])
        truth[pt] = dict(ball=T['ball'], box=T['box'], check=T['check'], f0=T['f0'],
                         g_norm=float(T['g'].norm()), eig_min=float(ev[0]),
                         eig_max=float(ev[-1]))
        for seed in range(a.seeds):
            r = run_oracle(ad, F, T, a, a.delta, seed)
            rows.append(dict(point=pt, arm=ORACLE, seed=seed, **r))
            for s in a.surrogates:
                for k in a.schemes:
                    r = run_surrogate(ad, F, T, a, s, k, a.delta, seed)
                    rows.append(dict(point=pt, arm=f'{s}/{k}', seed=seed, **r))
                    print(f'  {pt} seed {seed} {s + "/" + k:24s} '
                          + ' '.join(f'{d}={r["rise_" + d] / T["ball"]:+.2f}'
                                     for d in BALL_DIRS + BOX_DIRS)
                          + f'  AS1={r["as1"]:.2f} rank={r["rank_spearman"]:+.2f} '
                            f'({r["seconds"]:.0f}s)', flush=True)
            res = dict(problem=a.problem, delta=a.delta, budget=a.budget, truth=truth,
                       rows=kept + rows)
            json.dump(res, open(js, 'w'))
    table(res)
    plot(res, fig)
