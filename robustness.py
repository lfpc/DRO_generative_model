"""Robustness of the nominal muon-shield design, read off the LCSO surrogate.

Per-parameter sensitivity (gradient, curvature, exact one-at-a-time), Hessian spectrum,
active subspace, worst directions (+grad, trust-region maximizer of the quadratic model,
projected-gradient ascent in the box) and walks along them, checked with the simulator.
Every number gets a jackknife error over --n_folds disjoint muon folds (muon i -> fold i % K).

Frame: u = 0 is the nominal design, u_i = +-1 puts parameter i on the face of the +-delta box.

The logit is z(u, x) = <b(u), t(x)> + bias, so one pass over the muons accumulating
    T = sum w s t,   A = sum w s (1 - 2p) t t^T,   S = sum w s^2 t t^T     (s = p(1-p))
gives grad = J^T T, hess = J^T A J + d2(b.T)/du2 and active = J^T S J / sum w, with J = db/du
from autograd on the branch net alone. The trunk features of --n_cache muons are cached, so
the hits of any design are then one matmul (scans, Monte Carlo, ascent).

    python robustness.py --no_sim --n_muons 20000000          # surrogate only
    python robustness.py --n_muons 50000000 --n_muons_sim 5000000
    python robustness.py --muons data --dirs ascent,worst,pga,corner,random
"""
import argparse
import os
import time

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import load_surrogate
from problems import ShipMuonShieldCuda


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter,
                   argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument('--model', default='outputs/lcso_model.pt')
parser.add_argument('--muons', default='gen', choices=['gen', 'data'],
                    help='gen: flow samples (unweighted); data: the real file (weighted)')
parser.add_argument('--temperature', type=float, default=1.0, help='flow temperature (gen)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--pz_min', type=float, default=None, help='reject gen muons below this p_z')
parser.add_argument('--pt_max', type=float, default=None, help='reject gen muons above this p_t')
parser.add_argument('--n_muons', type=int, default=int(2e7),
                    help='muons for the derivatives (0 = all)')
parser.add_argument('--n_muons_sim', type=int, default=0,
                    help='muons for the walks (0 = n_muons)')
parser.add_argument('--n_folds', type=int, default=8, help='jackknife folds (>= 2)')
parser.add_argument('--n_sigma', type=float, default=3.0, help='significance for "resolved"')
parser.add_argument('--n_cache', type=int, default=int(1e6),
                    help='muons whose trunk features are cached')
parser.add_argument('--dirs', default='ascent,worst,pga,random',
                    help='directions to walk: ascent, descent, worst, corner, pga, random, '
                         'hess<k>, act<k>')
parser.add_argument('--t_max', type=float, default=1.0, help='walk half-length (L2, normalized)')
parser.add_argument('--n_points', type=int, default=5, help='points per walk (made odd)')
parser.add_argument('--n_repeats', type=int, default=1, help='simulator runs per design')
parser.add_argument('--quad_tol', type=float, default=0.2,
                    help='relative gap where the quadratic model counts as failed')
parser.add_argument('--no_sim', dest='sim', action='store_false', help='skip the simulator')
parser.add_argument('--no_walk', dest='walk', action='store_false',
                    help='walks from the cache only (no second pass)')
parser.add_argument('--no_check', dest='check', action='store_false',
                    help='skip the autograd cross-check')
parser.add_argument('--n_mc', type=int, default=4096, help='random designs in the box')
parser.add_argument('--pga_restarts', type=int, default=4)
parser.add_argument('--pga_steps', type=int, default=200)
parser.add_argument('--pga_lr', type=float, default=0.05)
parser.add_argument('--top', type=int, default=15, help='rows in the printed tables')
parser.add_argument('--block', type=int, default=int(2e7), help='muons held in memory')
parser.add_argument('--chunk', type=int, default=2 ** 19, help='muon chunk on the device')
parser.add_argument('--out_dir', default='outputs')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

# The p x p accumulators are differences of large terms: TF32 is not precise enough.
torch.backends.cuda.matmul.allow_tf32 = False

DEVICE = torch.device(args.device)
FIGS = os.path.join(args.out_dir, 'figs')
os.makedirs(FIGS, exist_ok=True)
GEN = args.muons == 'gen'
K = args.n_folds
NS = args.n_sigma
OK, BAD, GREY = '#0072B2', '#D55E00', '0.6'

REPORT = []


def say(line=''):
    """Print and keep for robustness_report.txt."""
    print(line)
    REPORT.append(line)


def jack_se(loo):
    """Jackknife standard error from leave-one-out replicates loo (K, ...)."""
    loo = np.asarray(loo, dtype=np.float64)
    m = loo.mean(0)
    return np.sqrt((loo.shape[0] - 1) / loo.shape[0] * ((loo - m) ** 2).sum(0))


def pm(value, se, fmt='.3f'):
    return f'{value:{fmt}} +- {se:{fmt}}'


def resolved(value, se):
    return abs(value) > NS * se


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def angle_deg(a, b):
    """Sign-insensitive angle between two directions."""
    c = abs(float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(min(1.0, c))))


# =============================================================================
# Setup
# =============================================================================
for attr, val in (('PZ_MIN', args.pz_min), ('PT_MAX', args.pt_max)):
    if val is not None:
        setattr(ShipMuonShieldCuda, attr, float(val))
        say(f'kinematic envelope: {attr} = {val}')

shield = ShipMuonShieldCuda(n_samples=1, uniform_fields=True)
phi_0 = shield.initial_phi                                   # physical
model, cfg = load_surrogate(args.model, phi_0)
model.eval().to(DEVICE)
D = model.dim
labels = [f'M{m}:{shield.idx_mag[i].split("[")[0]}' for m, i in shield.params_idx.tolist()]
magnet = np.array([m for m, _ in shield.params_idx.tolist()])

if GEN:
    shield._load_flow()   # before any seeding: building the flow draws from the global RNG


def read_muons(lo, hi):
    """Muons [lo, hi) as (raw, weights). gen: seeded by lo so both passes see the same
    muons, weights 1. data: transverse origin reset as in _load_muons, file weights."""
    if GEN:
        torch.manual_seed(args.seed + lo)
        x = shield.sample_gen(n_samples=hi - lo, temperature=args.temperature)
        return x, torch.ones(x.shape[0])
    with h5py.File(shield.muons_file, 'r') as f:
        x = np.stack([np.asarray(f[k][lo:hi], dtype=np.float32)
                      for k in ['px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight']], axis=1)
    x[:, 3], x[:, 4], x[:, 5] = 0.0, 0.0, -1.0
    x = torch.from_numpy(x)
    return x, x[:, 7]


if GEN:
    n = args.n_muons if args.n_muons > 0 else int(5e7)
    say(f'muons: generative flow at temperature {args.temperature:g}, seed {args.seed} '
        f'(unweighted)')
else:
    with h5py.File(shield.muons_file, 'r') as f:
        n = f['px'].shape[0]
    n = n if args.n_muons <= 0 else min(args.n_muons, n)
    say(f'muons: {shield.muons_file} (weighted)')
n_sim = n if args.n_muons_sim <= 0 else min(args.n_muons_sim, n)
say(f'model: {os.path.basename(args.model)} {cfg["model_type"]}, p = {cfg["p"]}, '
    f'delta = {model.delta}, D = {D}')
say(f'statistics: {K} muon folds, resolved means > {NS:g} jackknife sigma')

# =============================================================================
# One pass over the muons: accumulators per fold, and the feature cache
# =============================================================================
u_0 = torch.zeros(D, device=DEVICE)
b_0 = model.branch_net(u_0.unsqueeze(0)).squeeze(0).detach()   # (p,)
P = b_0.numel()
T_acc = torch.zeros(K, P, dtype=torch.float64, device=DEVICE)
A_acc = torch.zeros(K, P, P, dtype=torch.float64, device=DEVICE)
S_acc = torch.zeros(K, P, P, dtype=torch.float64, device=DEVICE)
# per fold: n, sum w, H = sum wp, sum w^2 p(1-p), sum (wp)^2
stat = torch.zeros(K, 5, dtype=torch.float64, device=DEVICE)
FOLD = [slice(k, None, K) for k in range(K)]
cache_t, cache_w = [], []
n_cached, seen = 0, 0
stride = max(1, n // args.n_cache)                 # cache spread over the whole sample
t0 = time.time()

for lo in range(0, n, args.block):
    raw, w = read_muons(lo, min(lo + args.block, n))
    with torch.no_grad():
        for c in range(0, raw.shape[0], args.chunk):
            xb = raw[c:c + args.chunk, :7].to(DEVICE)
            wb = w[c:c + args.chunk].to(DEVICE)
            tb = model.trunk_net(model.normalize_muons(xb))              # (chunk, p)
            pr = torch.sigmoid(tb @ b_0 + model.bias)
            s = pr * (1 - pr)
            ws, wq = s * wb, s * s * wb
            for k in range(K):
                f = FOLD[k]
                tk, prk, wk = tb[f], pr[f], wb[f]
                wp = (prk * wk).double()
                stat[k, 0] += tk.shape[0]
                stat[k, 1] += wk.double().sum()
                stat[k, 2] += wp.sum()
                stat[k, 3] += (s[f] * wk ** 2).double().sum()
                stat[k, 4] += (wp ** 2).sum()
                # float32 matmul per chunk, summed in float64
                T_acc[k] += (ws[f] @ tk).double()
                A_acc[k] += ((tk * (ws[f] * (1 - 2 * prk)).unsqueeze(1)).T @ tk).double()
                S_acc[k] += ((tk * wq[f].unsqueeze(1)).T @ tk).double()
            if n_cached < args.n_cache:
                first = (-seen) % stride           # keep the global stride across chunks
                if first < tb.shape[0]:
                    j = min(args.n_cache - n_cached, tb[first::stride].shape[0])
                    cache_t.append(tb[first::stride][:j].clone())
                    cache_w.append(wb[first::stride][:j].clone())
                    n_cached += j
            seen += tb.shape[0]
    say(f'  derivatives: {min(lo + args.block, n):,}/{n:,} muons ({time.time() - t0:.0f}s)')
    del raw, w

Tc = torch.cat(cache_t)                                # (n_cached, p)
wc = torch.cat(cache_w)
fc = torch.arange(n_cached, device=DEVICE) % K         # the cache's own folds
del cache_t, cache_w
stat_np = stat.cpu().numpy()
n_mu, sumw, H0, var0, sq0 = stat_np.sum(0)

# The design side, by autograd on the branch net alone (no muons involved).
J = torch.autograd.functional.jacobian(
    lambda u: model.branch_net(u.unsqueeze(0)).squeeze(0), u_0).detach()           # (p, D)
Jd = J.double()


def derive(T, A, S, w_sum):
    """(grad, hess, act) in the normalized frame from one set of accumulators."""
    Hb = torch.autograd.functional.hessian(
        lambda u: model.branch_net(u.unsqueeze(0)).squeeze(0) @ T.float(), u_0).detach()
    g = (Jd.T @ T).cpu().numpy()
    h = (Jd.T @ A @ Jd + Hb.double()).cpu().numpy()
    a = ((Jd.T @ S @ Jd) / w_sum).cpu().numpy()
    return g, 0.5 * (h + h.T), 0.5 * (a + a.T)


Tt, At, St = T_acc.sum(0), A_acc.sum(0), S_acc.sum(0)
grad, hess, act = derive(Tt, At, St, sumw)
# leave-one-fold-out replicates
g_loo, h_loo, a_loo = (np.array(x) for x in zip(*[
    derive(Tt - T_acc[k], At - A_acc[k], St - S_acc[k], sumw - stat_np[k, 1])
    for k in range(K)]))
H0_loo = H0 - stat_np[:, 2]

n_eff = H0 ** 2 / sq0     # Kish: how many muons the hit sum really rests on
say('')
say(f'H_0 = {pm(H0, jack_se(H0_loo), ".4g")} {"" if GEN else "weighted "}hits over '
    f'{int(n_mu):,} muons (total weight {sumw:,.1f})')
say(f'  the spread of one simulator draw around it is sqrt(sum w^2 p(1-p)) = '
    f'{np.sqrt(var0):,.2f}, i.e. {np.sqrt(var0) / H0:.1%} of H_0')
say(f'  effective number of muons behind that sum: {n_eff:,.0f} of {int(n_mu):,} '
    f'(sum(wp)^2 / sum (wp)^2)')
if n_eff < 100:
    say(f'  WARNING: the hit sum rests on ~{n_eff:.0f} muons. Every number below, and '
        f'every error bar on it,\n  is dominated by that handful; raise --n_muons before '
        f'reading anything as established.')
say(f'  error bars are jackknife over {K} muon folds -- the MUON-SAMPLING error only. '
    f'They say nothing\n  about whether the surrogate itself is right, which is what '
    f'the simulator walks are for. The\n  errors are themselves uncertain to about '
    f'{1 / np.sqrt(2 * (K - 1)):.0%}.')
say(f'cached the trunk features of {n_cached:,} muons (stride {stride}, '
    f'{n_cached * P * 4 / 2 ** 20:.0f} MB)')
say(f'the design enters only through the {P} branch features, so that is the most '
    f'directions this surrogate can tell apart')

if args.check:   # the frozen-trunk identities against LCSONet.grad_phi / hess_phi
    xb, wb = read_muons(0, min(2 ** 15, n))
    xd, wd = xb[:, :7].to(DEVICE), wb.to(DEVICE)
    with torch.no_grad():
        tb = model.trunk_net(model.normalize_muons(xd))
        pr = torch.sigmoid(tb @ b_0 + model.bias)
        sw = pr * (1 - pr) * wd
        Tk, Ak = sw @ tb, (tb * (sw * (1 - 2 * pr)).unsqueeze(1)).T @ tb
    Hbk = torch.autograd.functional.hessian(
        lambda u: model.branch_net(u.unsqueeze(0)).squeeze(0) @ Tk, u_0).detach()
    jac = (2.0 / (model.upper_bound - model.lower_bound)).cpu()   # du/dphi
    g_auto = model.grad_phi(phi_0.to(DEVICE), xd, wd).cpu() / jac
    h_auto = model.hess_phi(phi_0.to(DEVICE), xd, wd).cpu() / (jac[:, None] * jac[None, :])
    g_frz, h_frz = (J.T @ Tk).cpu(), (J.T @ Ak @ J + Hbk).cpu()
    say(f'check on {xd.shape[0]:,} muons: grad rel. err '
        f'{float((g_frz - g_auto).norm() / g_auto.norm()):.2e}, hess rel. err '
        f'{float((h_frz - h_auto).norm() / h_auto.norm()):.2e}  (float32 noise is ~1e-5)')
    del xb, wb, xd, wd

# =============================================================================
# Cheap surrogate evaluation on the cached features
# =============================================================================
CACHE_BUDGET = 2 ** 24            # elements held at once in the design x muon product


@torch.no_grad()
def cached_folds(U):
    """Expected weighted hits of normalized designs U (m, D), per cache fold: (m, K)."""
    U = torch.as_tensor(U, dtype=torch.float32).to(DEVICE)
    out = torch.zeros(U.shape[0], K, dtype=torch.float64, device=DEVICE)
    m_batch = max(1, min(U.shape[0], CACHE_BUDGET // n_cached))
    for i in range(0, U.shape[0], m_batch):
        b = model.branch_net(U[i:i + m_batch])                      # (m, p)
        step = max(1, CACHE_BUDGET // b.shape[0])
        for c in range(0, n_cached, step):
            pr = torch.sigmoid(b @ Tc[c:c + step].T + model.bias) * wc[c:c + step]
            out[i:i + b.shape[0]].index_add_(1, fc[c:c + step], pr.double())
    return out.cpu().numpy()


H0c_folds = cached_folds(np.zeros((1, D)))[0]                        # (K,)
H0c = H0c_folds.sum()
rate_c, rate_f = H0c / float(wc.double().sum()), H0 / sumw
say(f'the cache reproduces the nominal hit rate to {rate_c / rate_f - 1:+.1%} '
    f'({rate_c:.4e} against {rate_f:.4e})')


def fold_ratio(f):
    """H(u)/H(0) and its jackknife error from per-fold hits f (m, K). Taken fold by fold,
    so the sampling error of H_0 cancels."""
    loo = (f.sum(1, keepdims=True) - f) / (H0c - H0c_folds)[None, :]
    return f.sum(1) / H0c, jack_se(loo.T)


def cached_ratio(U):
    return fold_ratio(cached_folds(U))


def hits_cached_grad(u, mask):
    """Differentiable cached hits of one design, restricted to the muons in `mask`."""
    b = model.branch_net(u.unsqueeze(0)).squeeze(0)
    acc = 0.0
    for c in range(0, n_cached, 2 ** 20):
        sl = slice(c, c + 2 ** 20)
        pr = torch.sigmoid(Tc[sl] @ b + model.bias) * wc[sl]
        acc = acc + (pr * mask[sl]).sum()
    return acc


# =============================================================================
# 1. Which parameters matter
# =============================================================================
# grad_i: first order for a full tolerance move of parameter i; oat: the exact answer.
E = np.concatenate([np.eye(D), -np.eye(D)])
oat_fold = cached_folds(E)
oat, oat_se = fold_ratio(oat_fold)
oat_p, oat_p_se = oat[:D] - 1.0, oat_se[:D]
oat_m, oat_m_se = oat[D:] - 1.0, oat_se[D:]
lin = grad / H0
lin_se = jack_se(g_loo / H0_loo[:, None])
quad = 0.5 * np.diag(hess) / H0
quad_se = jack_se(np.array([0.5 * np.diag(h_loo[k]) / H0_loo[k] for k in range(K)]))
worst_oat = np.where(oat_p >= oat_m, oat_p, oat_m)                  # the bad side
worst_se = np.where(oat_p >= oat_m, oat_p_se, oat_m_se)
order = np.argsort(-np.maximum(np.abs(oat_p), np.abs(oat_m)))

# The ranking per leave-one-out replicate: best and worst place of each parameter.
TOPN = min(5, D)
ranks = np.zeros((K, D), dtype=int)
for k in range(K):
    loo = (oat_fold.sum(1) - oat_fold[:, k]) / (H0c - H0c_folds[k])
    key = np.maximum(np.abs(loo[:D] - 1), np.abs(loo[D:] - 1))
    ranks[k] = np.argsort(np.argsort(-key))
rank_lo, rank_hi = ranks.min(0) + 1, ranks.max(0) + 1

say('')
say('=' * 79)
say(f'1. SENSITIVITY OF EACH PARAMETER   (a full +-{model.delta:.0%} move of that one '
    f'parameter)')
say('=' * 79)
say(f'{"parameter":22s} {"grad/H_0":>17s} {"curv/H_0":>17s} {"exact +1":>17s} '
    f'{"exact -1":>17s} {"rank":>8s}')
for k in order[:args.top]:
    flag = '' if resolved(worst_oat[k], worst_se[k]) else '?'
    rank = f'{rank_lo[k]}-{rank_hi[k]}' if rank_lo[k] != rank_hi[k] else f'{rank_lo[k]}'
    say(f'{labels[k]:22s} {pm(lin[k], lin_se[k]):>17s} {pm(quad[k], quad_se[k]):>17s} '
        f'{pm(oat_p[k], oat_p_se[k]):>17s} {pm(oat_m[k], oat_m_se[k]):>17s} '
        f'{rank + flag:>8s}')
n_res = int(sum(resolved(worst_oat[k], worst_se[k]) for k in range(D)))
stable_top = int(np.sum(rank_hi[order[:TOPN]] <= TOPN))
say(f'  {n_res}/{D} parameters have a one-at-a-time effect that is non-zero at {NS:g} '
    f'sigma ("?" marks the rest).\n  That the VALUES are pinned does not mean their order '
    f'is: rank is the best-to-worst place the\n  parameter takes over the {K} '
    f'leave-one-out replicates. {stable_top} of the top {TOPN} hold their place in all '
    f'of them.')

say('')
say('per magnet, share of the gradient norm:')
for m in sorted(set(magnet.tolist())):
    sel = magnet == m
    share = np.linalg.norm(grad[sel]) / np.linalg.norm(grad)
    share_loo = [np.linalg.norm(g_loo[k][sel]) / np.linalg.norm(g_loo[k]) for k in range(K)]
    say(f'  M{m}: {pm(share, jack_se(share_loo), ".3f")}  ({int(sel.sum())} parameters)')

# =============================================================================
# 2. Curvature and the active subspace
# =============================================================================
mu, Qh = np.linalg.eigh(hess)
mu, Qh = mu[::-1].copy(), Qh[:, ::-1].copy()        # most positive (worst) first
nu, Qa = np.linalg.eigh(act)
nu, Qa = np.clip(nu[::-1], 0, None).copy(), Qa[:, ::-1].copy()
mu_se = jack_se(np.array([np.linalg.eigvalsh(h_loo[k])[::-1] for k in range(K)]))
g_norm = np.linalg.norm(grad)
g_hat = grad / g_norm
curv_g = float(g_hat @ hess @ g_hat)
gn_loo = np.array([np.linalg.norm(g_loo[k]) / H0_loo[k] for k in range(K)])
pr_act = float(nu.sum() ** 2 / (nu ** 2).sum())     # participation ratio
cum = np.cumsum(nu) / nu.sum()
proj = np.cumsum((g_hat @ Qa) ** 2)                 # share of g in the top-k span

say('')
say('=' * 79)
say('2. CURVATURE AND ACTIVE DIRECTIONS')
say('=' * 79)
say(f'|grad|/H_0 = {pm(g_norm / H0, jack_se(gn_loo))} per normalized unit')
curv_g_se = jack_se(np.array([g_hat @ h_loo[k] @ g_hat / H0_loo[k] for k in range(K)]))
say(f'Hessian: max eigenvalue {pm(mu[0] / H0, mu_se[0] / H0)} H_0, min '
    f'{pm(mu[-1] / H0, mu_se[-1] / H0)} H_0, curvature along the gradient '
    f'{pm(curv_g / H0, curv_g_se)} H_0')
say(f'  a step of t = {args.t_max:g} along the gradient is worth '
    f'{args.t_max * g_norm / H0:+.3f} H_0 from the slope and '
    f'{0.5 * args.t_max ** 2 * curv_g / H0:+.3f} H_0 from the curvature, so second order '
    f'is\n  {abs(0.5 * args.t_max * curv_g / g_norm):.0%} of first order there')
n_mu_res = int(sum(resolved(mu[k], mu_se[k]) for k in range(D)))
say(f'  {n_mu_res}/{D} Hessian eigenvalues are resolved at {NS:g} sigma')
say(f'active subspace: participation ratio {pr_act:.1f} directions '
    f'(rank is at most p = {P}); {int(np.searchsorted(cum, 0.95) + 1)} of them carry 95% '
    f'of E[|dp/du|^2]')
say(f'  the total gradient lies {proj[2]:.1%} inside the top 3 active directions, '
    f'{proj[0]:.1%} in the first')


# =============================================================================
# 3. The worst directions
# =============================================================================
def trust_region_max(g, B, radius):
    """argmax of g.v + 0.5 v^T B v over ||v|| <= radius (More-Sorensen, by bisection on
    lam in (lam I - B) v = g). Returns (v, lam, relative KKT residual)."""
    ev, Q = np.linalg.eigh(B)
    gh = Q.T @ g
    top = ev[-1]

    def resid(v, lam):
        return float(np.linalg.norm(lam * v - B @ v - g) / np.linalg.norm(g))

    if top < 0:                                  # concave: the free maximum may be inside
        v = -np.linalg.solve(B, g)
        if np.linalg.norm(v) <= radius:
            return v, 0.0, resid(v, 0.0)

    def v_of(lam):
        return Q @ (gh / (lam - ev))

    lo = top + 1e-12 * max(1.0, abs(top))
    if np.linalg.norm(v_of(lo)) < radius:
        # hard case: g has no component on the top eigenvector; step out along it
        keep = ev < top - 1e-12 * max(1.0, abs(top))
        v = Q[:, keep] @ (gh[keep] / (top - ev[keep]))
        v = v + np.sqrt(max(radius ** 2 - v @ v, 0.0)) * Q[:, -1] * np.sign(gh[-1] or 1.0)
        return v, top, resid(v, top)
    hi = top + max(1e-12, np.linalg.norm(g) / radius)
    while np.linalg.norm(v_of(hi)) > radius:
        hi = top + 2 * (hi - top)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.linalg.norm(v_of(mid)) > radius else (lo, mid)
    lam = 0.5 * (lo + hi)
    return v_of(lam), lam, resid(v_of(lam), lam)


v_trs, lam_trs, kkt_trs = trust_region_max(grad, hess, args.t_max)
# check: no random point on the same sphere should beat it under the same model
probe = np.random.default_rng(args.seed + 7).standard_normal((4000, D))
probe *= args.t_max / np.linalg.norm(probe, axis=1, keepdims=True)
q_probe = probe @ grad + 0.5 * np.einsum('ij,jk,ik->i', probe, hess, probe)
q_trs = float(grad @ v_trs + 0.5 * v_trs @ hess @ v_trs)

# Worst design in the whole box: projected gradient ascent on the surrogate, fitted on
# half the cache folds and scored on the other half (else it fits the subsample's noise).
half = K // 2
train_mask = (fc < half).to(Tc.dtype)
torch.manual_seed(args.seed + 1)
starts = [torch.zeros(D), torch.from_numpy(np.sign(grad)).float() * 0.5,
          torch.from_numpy(unit(v_trs)).float()]
starts += [torch.rand(D) * 2 - 1 for _ in range(max(0, args.pga_restarts - len(starts)))]
starts = starts[:args.pga_restarts]
best_h, best_u = -np.inf, torch.zeros(D)
for u_init in starts:
    u = u_init.clone().to(DEVICE).clamp(-1, 1).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=args.pga_lr)
    for _ in range(args.pga_steps):
        opt.zero_grad()
        h = hits_cached_grad(u, train_mask)
        (-h).backward()
        here = (float(h), u.detach().clone())   # the value belongs to u BEFORE the step
        opt.step()
        with torch.no_grad():
            u.clamp_(-1, 1)                     # project back onto the box
        if here[0] > best_h:
            best_h, best_u = here
    with torch.no_grad():                       # and the point the last step landed on
        h_end = float(hits_cached_grad(u, train_mask))
    if h_end > best_h:
        best_h, best_u = h_end, u.detach().clone()
u_star = best_u.cpu()
star_folds = cached_folds(u_star.numpy()[None])[0]
star_in = star_folds[:half].sum() / H0c_folds[:half].sum()
star_out = star_folds[half:].sum() / H0c_folds[half:].sum()
q_star = float(1 + (grad @ u_star.numpy() + 0.5 * u_star.numpy() @ hess @ u_star.numpy())
               / H0)


def oriented(v, g):
    """Eigenvectors have an arbitrary sign; point them the way that raises the hits."""
    v = unit(v)
    return -v if g @ v < 0 else v


V_RANDOM = unit(np.random.default_rng(args.seed + 3).standard_normal(D))


def direction_of(name, g, h, a):
    """(unit vector, its own walk half-length or None, plot title) from one replicate."""
    if name == 'ascent':
        return unit(g), None, 'first-order worst: $+\\nabla H$'
    if name == 'descent':
        return unit(-g), None, 'descent: $-\\nabla H$'
    if name == 'worst':
        return (unit(trust_region_max(g, h, args.t_max)[0]), args.t_max,
                f'second-order worst at $t={args.t_max:g}$')
    if name == 'corner':
        return unit(np.sign(g)), None, 'worst box corner (first order)'
    if name == 'pga':
        return unit(u_star.numpy()), float(u_star.norm()), 'worst design in the box (ascent)'
    if name == 'random':
        return V_RANDOM, None, 'random direction (control)'
    if name.startswith('hess'):
        k = int(name[4:]) - 1
        return oriented(np.linalg.eigh(h)[1][:, ::-1][:, k], g), None, \
            f'Hessian eigenvector {k + 1}'
    if name.startswith('act'):
        k = int(name[3:]) - 1
        return oriented(np.linalg.eigh(a)[1][:, ::-1][:, k], g), None, \
            f'active direction {k + 1}'
    raise ValueError(f'unknown direction {name!r}')


DIRS = []
for name in args.dirs.split(','):
    v, t_star, title = direction_of(name, grad, hess, act)
    # drift: the same recipe on each leave-one-out replicate (pga is scored held-out instead)
    angles = [] if name == 'pga' else [
        angle_deg(v, direction_of(name, g_loo[k], h_loo[k], a_loo[k])[0]) for k in range(K)]
    DIRS.append({'name': name, 'v': v, 't_star': t_star, 'title': title, 'angles': angles})

n_pts = args.n_points + (args.n_points + 1) % 2              # odd -> t = 0 on the grid
for d in DIRS:
    d['ts'] = np.linspace(-1, 1, n_pts) * (d['t_star'] or args.t_max)
    d['slope'] = float(grad @ d['v'])
    d['curv'] = float(d['v'] @ hess @ d['v'])
    d['slope_se'] = jack_se([g_loo[k] @ d['v'] / H0_loo[k] for k in range(K)])
    d['curv_se'] = jack_se([d['v'] @ h_loo[k] @ d['v'] / H0_loo[k] for k in range(K)])
    # t_valid: how far the quadratic model stays within quad_tol of the surrogate
    tt = np.linspace(0, max(abs(d['ts'][0]), abs(d['ts'][-1])), 33)[1:]
    ex, _ = cached_ratio(np.clip(np.concatenate([np.outer(tt, d['v']),
                                                 np.outer(-tt, d['v'])]), -1, 1))
    t2 = np.concatenate([tt, -tt])
    qd = 1 + (t2 * d['slope'] + 0.5 * t2 ** 2 * d['curv']) / H0
    bad = np.abs(qd - ex) > args.quad_tol * np.abs(ex)
    bad = bad[:len(tt)] | bad[len(tt):]                      # fail on either side
    d['t_valid'] = float(tt[bad.argmax() - 1]) if bad.any() else float(tt[-1])
    d['t_valid_capped'] = not bad.any()
    # the far end of the walk, and t = t_max (comparable with the one-at-a-time table)
    r, se = cached_ratio(np.stack([np.clip(d['ts'][-1] * d['v'], -1, 1),
                                   np.clip(args.t_max * d['v'], -1, 1)]))
    d['end_cache'], d['end_cache_se'] = float(r[0]), float(se[0])
    d['at_tmax'], d['at_tmax_se'] = float(r[1]), float(se[1])

say('')
say('=' * 79)
say('3. THE WORST DIRECTIONS')
say('=' * 79)
say(f'trust region at ||u|| = {args.t_max:g}: lambda = {lam_trs:.4g}, KKT residual '
    f'{kkt_trs:.1e}, and it beats {(q_probe < q_trs).mean():.1%} of 4000 random '
    f'directions of the same length under the same model')
say(f'{"direction":10s} {"|v|inf":>7s} {"cos(g)":>7s} {"slope/H_0":>17s} '
    f'{"curv/H_0":>17s} {"H/H_0 at t_max":>17s} {"at walk end":>17s} {"t_valid":>8s} '
    f'{"drift":>7s}')
for d in DIRS:
    drift = f'{np.mean(d["angles"]):.1f}d' if d['angles'] else '-'
    tv = ('>' if d['t_valid_capped'] else ' ') + f'{d["t_valid"]:.2f}'
    say(f'{d["name"]:10s} {np.abs(d["v"]).max():7.3f} {g_hat @ d["v"]:+7.3f} '
        f'{pm(d["slope"] / H0, d["slope_se"]):>17s} '
        f'{pm(d["curv"] / H0, d["curv_se"]):>17s} '
        f'{pm(d["at_tmax"], d["at_tmax_se"], ".4g"):>17s} '
        f'{pm(d["end_cache"], d["end_cache_se"], ".4g"):>17s} {tv:>8s} {drift:>7s}')
    say(f'{"":10s} leading: ' + ', '.join(f'{labels[k]}{d["v"][k]:+.2f}'
                                          for k in np.argsort(-np.abs(d['v']))[:4]))
say(f'  slope and curvature are per unit t. "at t_max" is the surrogate at '
    f't = {args.t_max:g} for every direction,\n  which is the same size of step as one '
    f'parameter moved to its tolerance, so it is the column that\n  compares with the '
    f'table above; "at walk end" is each direction\'s own far end. t_valid is the\n  '
    f'largest |t| where the quadratic model is still within {args.quad_tol:.0%} of the '
    f'surrogate (">" = never\n  failed inside the range tested): past it the slope and '
    f'curvature columns stop describing the\n  walk, though the walk itself remains '
    f'valid. drift is the mean angle the direction moves when a\n  fold is left out.')

say('')
say(f'worst design in the box: H/H_0 = {star_in:.4g} on the half it was fitted to, '
    f'{star_out:.4g} on the held-out half')
say(f'  at ||u|| = {float(u_star.norm()):.2f} '
    f'({int((u_star.abs() > 0.99).sum())}/{D} parameters pinned to a face); the training '
    f'designs sit at ||u|| ~ {np.sqrt(D / 3):.1f}')
say(f'  the quadratic model would have said {q_star:.4g} there, which is why the number '
    f'to trust is the walk below, not the model')
say('  its largest components: ' + ', '.join(
    f'{labels[k]} {u_star[k]:+.2f}' for k in torch.argsort(u_star.abs(), descending=True)[:6]))
if star_out < 0.5 * star_in:
    say('  WARNING: the held-out value is far below the fitted one -- the ascent is '
        'exploiting the cached\n  subsample. Raise --n_cache.')

# reference: random designs uniform in the box
gen_mc = torch.Generator().manual_seed(args.seed + 2)
U_mc = (torch.rand(args.n_mc, D, generator=gen_mc) * 2 - 1).numpy()
mc, _ = cached_ratio(U_mc)
qs = np.percentile(mc, [5, 50, 95, 99])
say('')
say(f'{args.n_mc:,} designs drawn uniformly in the tolerance box: H/H_0 median '
    f'{qs[1]:.3f}, 5-95% [{qs[0]:.3f}, {qs[2]:.3f}], 99% {qs[3]:.3f}, max {mc.max():.3f}')
say(f'  so a random tolerance excursion is {np.mean(mc > 1):.0%} likely to make things '
    f'worse at all, while the\n  directions above are the ones chosen to')

# =============================================================================
# 4. Walk the directions: surrogate over the full sample, and the simulator
# =============================================================================
U_list = [np.zeros(D)]          # the nominal design is index 0, shared by all walks
for d in DIRS:
    d['idx'] = []
    for t in d['ts']:
        if abs(t) < 1e-12:
            d['idx'].append(0)
        else:
            d['idx'].append(len(U_list))
            U_list.append(t * d['v'])
U_raw = np.stack(U_list)
U_all = np.clip(U_raw, -1, 1)
n_clip = int((np.abs(U_raw) > 1 + 1e-6).sum())
if n_clip:
    say(f'note: {n_clip} components across {len(U_list)} designs fall outside the '
        f'+-delta box and are clamped to its faces, which bends those walks')
PHI = model.denormalize_phi(torch.from_numpy(U_all).float().to(DEVICE)).cpu()
n_des, R = len(U_list), args.n_repeats
walk_cache, walk_cache_se = cached_ratio(U_all)

pred = np.zeros((K, n_des))     # sum_x w p        expected hits, per fold
pvar = np.zeros(n_des)          # sum_x w^2 p(1-p) spread of ONE simulator draw
sim = np.zeros((R, n_des))      # sum_x w hit      simulated hits, per repeat
svar = np.zeros((R, n_des))     # sum_x w^2 hit    its counting error, per repeat
if args.walk:
    say('')
    say(f'walking {n_des} designs over {n_sim:,} muons'
        + (f' x {R} repeats = {n_des * n_sim * R / 1e9:.1f}e9 tracks to simulate'
           if args.sim else ' (surrogate only)'))
    PHI_D = PHI.to(DEVICE)
    t0 = time.time()
    for lo in range(0, n_sim, args.block):
        raw, w = read_muons(lo, min(lo + args.block, n_sim))
        with torch.no_grad():
            for c in range(0, raw.shape[0], args.chunk):
                xb = raw[c:c + args.chunk, :7].to(DEVICE)
                wb = w[c:c + args.chunk].to(DEVICE)
                pr = model.predict_proba(PHI_D, xb.unsqueeze(0))        # (n_des, chunk)
                pr, q = pr * wb, pr * (1 - pr) * wb ** 2
                for k in range(K):
                    pred[k] += pr[:, FOLD[k]].sum(1).double().cpu().numpy()
                pvar += q.sum(1).double().cpu().numpy()
        if args.sim:
            wd = w.double()
            for i in range(n_des):
                for r in range(R):
                    hits = shield(PHI[i], raw).double()  # a new transport seed every call
                    sim[r, i] += float((hits * wd).sum())
                    svar[r, i] += float((hits * wd ** 2).sum())
        say(f'  walk: {min(lo + args.block, n_sim):,}/{n_sim:,} muons '
            f'({time.time() - t0:.0f}s)')
        del raw, w

pred_tot = pred.sum(0)
pstd = np.sqrt(pvar)
smean, smc = sim.mean(0), (sim.std(0, ddof=1) if R > 1 else np.zeros(n_des))
scount = np.sqrt(svar).mean(0)
serr = smc if R > 1 else scount
if args.walk:
    ratio = pred_tot / pred_tot[0]
    ratio_se = jack_se([(pred_tot - pred[k]) / (pred_tot[0] - pred[k][0]) for k in range(K)])
else:
    ratio, ratio_se = walk_cache, walk_cache_se


def sim_ratio(i):
    """Simulator H(t)/H(0) and its error (only the counting term is independent: the two
    ends share the muons)."""
    if i == 0:
        return 1.0, 0.0
    sr = smean[i] / smean[0]
    return sr, sr * np.sqrt((serr[i] / smean[i]) ** 2 + (serr[0] / smean[0]) ** 2)


say('')
say('=' * 79)
say('4. THE WALKS' + ('   (surrogate and simulator on the same muons)' if args.sim
                      else '   (surrogate only)'))
say('=' * 79)
if args.walk:
    say(f'nominal design over these {n_sim:,} muons: surrogate {pred_tot[0]:,.2f} hits'
        + (f', simulator {smean[0]:,.1f} +- {serr[0]:,.1f}' if args.sim else '')
        + '. The columns below are ratios to it.')
head = f'{"direction":10s} {"t":>7s} {"H/H_0 surrogate":>19s}'
if args.sim:
    head += f' {"H/H_0 simulator":>19s} {"z":>6s}'
say(head + f'  {"cache":>9s}')
zs = []
for d in DIRS:
    for j, t in enumerate(d['ts']):
        i = d['idx'][j]
        line = (f'{d["name"] if j == 0 else "":10s} {t:+7.3f} '
                f'{pm(ratio[i], ratio_se[i], ".4g"):>19s}')
        if args.sim:
            sr, sr_se = sim_ratio(i)
            z = (ratio[i] - sr) / np.hypot(ratio_se[i], sr_se) if i else 0.0
            line += f' {pm(sr, sr_se, ".4g"):>19s} {z:+6.1f}'
            if j == len(d['ts']) - 1:
                zs.append(abs(z))
        say(line + f'  {walk_cache[i]:9.4g}')
if args.sim:
    say(f'  z is the surrogate-simulator gap in combined sigma. |z| < {NS:g} means the '
        f'two agree within what\n  the muon sample can resolve; larger means the '
        f'surrogate is wrong there, not unlucky. Both errors\n  are muon-sampling only, '
        f'so z does not know about the transport systematics.')
    say(f'  at the far end of the walks: max |z| = {max(zs):.1f} '
        f'({DIRS[int(np.argmax(zs))]["name"]})')
elif args.walk:
    say(f'  cache is the same ratio from the {n_cached:,} cached muons: it should track '
        f'the full-sample column,\n  and where it does not, the cached scans above are '
        f'the ones to distrust.')

# =============================================================================
# 5. Conclusions
# =============================================================================
say('')
say('=' * 79)
say('5. CONCLUSIONS')
say('=' * 79)
i_best = order[0]
say(f'1. Nominal: H_0 = {H0:,.1f} {"" if GEN else "weighted "}hits over {int(n_mu):,} '
    f'muons (hit rate {H0 / sumw:.3e}), resting on ~{n_eff:,.0f} muons effectively.')
say(f'2. Most sensitive single parameter: {labels[i_best]}. Moving it alone to its '
    f'{"+" if oat_p[i_best] >= oat_m[i_best] else "-"}{model.delta:.0%} tolerance changes '
    f'the hits by {worst_oat[i_best]:+.1%} +- {worst_se[i_best]:.1%}.')
verdict = 'resolved' if resolved(worst_oat[i_best], worst_se[i_best]) else 'NOT resolved'
say(f'   It is {verdict} at {NS:g} sigma, and over the {K} leave-one-out replicates it '
    f'never leaves rank {rank_lo[i_best]}-{rank_hi[i_best]}.')
listed = [labels[k] for k in order if resolved(worst_oat[k], worst_se[k])][:args.top]
say(f'   {n_res}/{D} parameters have a resolved one-at-a-time effect; the largest are '
    + ', '.join(listed) + (f' and {n_res - len(listed)} more.' if n_res > len(listed)
                           else '.'))
dw = DIRS[int(np.argmax([d['at_tmax'] for d in DIRS]))]
say(f'3. Worst direction at the same step size as that ({args.t_max:g} normalized unit): '
    f'{dw["name"]}, H/H_0 = {pm(dw["at_tmax"], dw["at_tmax_se"], ".4g")}, i.e. '
    f'{abs(dw["at_tmax"] - 1) / abs(worst_oat[i_best]):.1f}x the worst single '
    f'parameter.')
say(f'   Its drift over the folds is '
    + (f'{np.mean(dw["angles"]):.1f} degrees' if dw['angles'] else 'not estimated')
    + f', and it is led by '
    + ', '.join(labels[k] for k in np.argsort(-np.abs(dw['v']))[:3]) + '.')
tv = [d['t_valid'] for d in DIRS]
say(f'4. The quadratic model tracks the surrogate to |t| ~ {min(tv):.2f}-{max(tv):.2f} '
    f'(within {args.quad_tol:.0%}), so slope-and-curvature statements are local: the box '
    f'corners at ||u|| = {np.sqrt(D):.1f} are far outside that range and only the walks '
    f'speak for them.')
say(f'5. Worst design found anywhere in the +-{model.delta:.0%} box: H/H_0 = '
    f'{star_out:.4g} on held-out muons ({star_in:.4g} on the half it was fitted to), '
    f'against a median random excursion of {qs[1]:.2f} and a 99th percentile of '
    f'{qs[3]:.2f}.')
if args.sim:
    say(f'6. Simulator agreement at the walk ends: max |z| = {max(zs):.1f} sigma '
        f'({DIRS[int(np.argmax(zs))]["name"]}); '
        + ('all directions agree within the muon-sampling error.'
           if max(zs) < NS else 'the surrogate does not reproduce the simulator there, '
                                'so the directions above are the surrogate\'s opinion '
                                'and not yet a result.'))
else:
    say('6. Not checked against the simulator in this run (--no_sim), so everything above '
        'is a statement about the surrogate, not about the shield.')

# =============================================================================
# Figures
# =============================================================================
SRC = f'generative muons, T = {args.temperature:g}' if GEN else 'real muons, weighted'

# --- 1. per-parameter sensitivity
fig, ax = plt.subplots(1, 2, figsize=(13, 9), constrained_layout=True, sharey=True)
o = order[::-1]                                   # least sensitive at the bottom
y = np.arange(D)
res_mask = np.array([resolved(worst_oat[k], worst_se[k]) for k in o])
ax[0].barh(y, lin[o], xerr=lin_se[o], error_kw=dict(lw=0.8, ecolor='k'),
           color=[(BAD if v > 0 else OK) if r else GREY for v, r in zip(lin[o], res_mask)])
ax[0].set_yticks(y, [labels[k] for k in o], fontsize=7)
ax[0].axvline(0, color='k', lw=0.8)
ax[0].set_xlabel('$(\\partial H/\\partial u_i)\\,/\\,H_0$   (per full tolerance)')
ax[0].set_title('First-order sensitivity')
ax[1].hlines(y, oat_m[o], oat_p[o], color=GREY, lw=1)
ax[1].errorbar(oat_p[o], y, xerr=oat_p_se[o], fmt='o', ms=4, color=BAD, lw=0.8,
               label='$u_i = +1$')
ax[1].errorbar(oat_m[o], y, xerr=oat_m_se[o], fmt='o', ms=4, color=OK, lw=0.8,
               label='$u_i = -1$')
ax[1].axvline(0, color='k', lw=0.8)
ax[1].set_xlabel('$H/H_0 - 1$')
ax[1].set_title(f'Exact one-at-a-time, $\\pm${model.delta:.0%}')
ax[1].legend()
fig.suptitle(f'Parameter sensitivity of the hit count ({SRC}); grey = not resolved at '
             f'{NS:g}$\\sigma$')
fig.savefig(os.path.join(FIGS, 'robustness_sensitivity.png'), dpi=150)
plt.close(fig)

# --- 2. spectra
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
ax[0].axhline(0, color='k', lw=0.8)
ax[0].errorbar(np.arange(1, D + 1), mu / H0, yerr=mu_se / H0, fmt='o-', ms=4, color=OK,
               lw=1, capsize=2)
ax[0].set(xlabel='index (most positive first)', ylabel='eigenvalue $/H_0$',
          title='Hessian at the nominal design')
ax[0].set_yscale('symlog', linthresh=max(1e-3, np.abs(mu / H0).max() * 1e-3))
pos = nu > nu.max() * 1e-12
ax[1].set(xlabel='index', ylabel='eigenvalue',
          title=f'Active subspace ($\\partial p/\\partial u$), participation {pr_act:.1f}')
ax[1].semilogy(np.arange(1, pos.sum() + 1), nu[pos], 'o-', ms=4, color=OK)
tw = ax[1].twinx()
tw.plot(np.arange(1, pos.sum() + 1), cum[pos], 's--', ms=3, color=BAD)
tw.plot(np.arange(1, pos.sum() + 1), proj[pos], '^:', ms=3, color='0.3')
tw.set_ylabel('cumulative share (red) / of $\\nabla H$ (grey)', color=BAD)
tw.set_ylim(0, 1.02)
tw.axhline(0.95, color=BAD, lw=0.8, ls=':')
fig.suptitle('Curvature and active directions')
fig.savefig(os.path.join(FIGS, 'robustness_spectra.png'), dpi=150)
plt.close(fig)

# --- 3. what the directions are made of
M = np.stack([d['v'] for d in DIRS])
fig, ax = plt.subplots(figsize=(max(9, D * 0.28), 1.4 + 0.55 * len(DIRS)),
                       constrained_layout=True)
lim = np.abs(M).max()
im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
ax.set_xticks(np.arange(D), labels, rotation=90, fontsize=7)
ax.set_yticks(np.arange(len(DIRS)),
              [d['name'] + (f' ({np.mean(d["angles"]):.0f}$^\\circ$)' if d['angles'] else '')
               for d in DIRS])
fig.colorbar(im, ax=ax, label='component (unit $\\ell_2$)')
ax.set_title('Directions in parameter space (label: mean jackknife drift)')
fig.savefig(os.path.join(FIGS, 'robustness_directions.png'), dpi=150)
plt.close(fig)

# --- 4. the walks
nc = min(3, len(DIRS))
nr = int(np.ceil(len(DIRS) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.9 * nr), constrained_layout=True,
                         squeeze=False)
for a in axes.ravel()[len(DIRS):]:
    a.axis('off')
for a, d in zip(axes.ravel(), DIRS):
    ts, idx = d['ts'], np.array(d['idx'])
    tt = np.linspace(ts[0], ts[-1], 101)
    a.axvspan(-d['t_valid'], d['t_valid'], color='0.92', zorder=0,
              label='quadratic model valid')
    a.plot(tt, 1 + (tt * d['slope'] + 0.5 * tt ** 2 * d['curv']) / H0, ':', color='0.4',
           lw=1.5, label='quadratic model')
    a.plot(ts, walk_cache[idx], '-', color=OK, lw=1, alpha=0.6,
           label=f'surrogate ({n_cached:,} cached)')
    a.errorbar(ts, ratio[idx], yerr=ratio_se[idx], fmt='o-', color=OK, ms=4, lw=2,
               capsize=2, label='surrogate (full sample)')
    if args.sim:
        sr = smean[idx] / smean[0]
        a.errorbar(ts, sr, yerr=serr[idx] / smean[0], fmt='s', ms=5, color=BAD, capsize=3,
                   lw=1.5, label=f'simulator (mean of {R})' if R > 1 else 'simulator')
        if R > 1:
            a.plot(np.repeat(ts[None], R, 0), sim[:, idx] / smean[0], '.', ms=3, color=BAD,
                   alpha=0.35, zorder=0, label='_')
    a.axvline(0, color=GREY, lw=1, ls='--')
    a.axhline(1, color=GREY, lw=0.8, ls=':')
    if d['t_star'] is not None:
        a.axvline(d['t_star'], color=BAD, lw=1, ls=':')
    a.set_title(f'{d["name"]}: {d["title"]}', fontsize=10)
    a.set_xlabel('step $t$ along $v$ (normalized units)')
    a.set_ylabel('$H(t)\\,/\\,H_0$')
    curves = [walk_cache[idx], ratio[idx]] + ([smean[idx] / smean[0]] if args.sim else [])
    if all(np.all(c > 0) for c in curves):
        a.set_yscale('log')
        a.set_ylim(0.5 * min(c.min() for c in curves), 2 * max(c.max() for c in curves))
    if a is axes.ravel()[0]:
        a.legend(fontsize=8)
fig.suptitle(f'Surrogate vs. simulator along the robustness directions ({SRC})')
fig.savefig(os.path.join(FIGS, 'robustness_walks.png'), dpi=150)
plt.close(fig)

# --- 5. the box as a whole
fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
edges = np.logspace(np.log10(max(mc.min(), 1e-8) * 0.9),
                    np.log10(max(mc.max(), star_out) * 1.2), 60)
ax.hist(np.clip(mc, edges[0], edges[-1]), bins=edges, color=OK, alpha=0.8)
ax.set_xscale('log')
ax.axvline(1.0, color='k', lw=1.2, ls='--', label='nominal')
ax.axvline(qs[2], color=GREY, lw=1, ls=':', label='95th percentile')
ax.axvline(star_out, color=BAD, lw=1.5, label='worst found (held out)')
ax.set(xlabel='$H/H_0$', ylabel='designs',
       title=f'{args.n_mc:,} designs uniform in the $\\pm${model.delta:.0%} box')
ax.legend()
fig.savefig(os.path.join(FIGS, 'robustness_mc.png'), dpi=150)
plt.close(fig)

# =============================================================================
# Save
# =============================================================================
report_path = os.path.join(args.out_dir, 'robustness_report.txt')
np.savez(os.path.join(args.out_dir, 'robustness.npz'),
         labels=np.array(labels), phi_0=phi_0.numpy(), H_0=H0, var_0=var0, n_muons=n_mu,
         sum_weights=sumw, muons=args.muons, temperature=args.temperature,
         n_folds=K, fold_stats=stat_np,
         grad=grad, grad_loo=g_loo, hess=hess, hess_loo=h_loo, active=act,
         hess_eigvals=mu, hess_eigval_se=mu_se, hess_eigvecs=Qh,
         act_eigvals=nu, act_eigvecs=Qa, act_participation=pr_act,
         oat_plus=oat_p, oat_plus_se=oat_p_se, oat_minus=oat_m, oat_minus_se=oat_m_se,
         rank_lo=rank_lo, rank_hi=rank_hi,
         mc=mc, u_star=u_star.numpy(), h_star_in=star_in,
         h_star_out=star_out, v_trs=v_trs, trs_lambda=lam_trs, trs_kkt=kkt_trs,
         dir_names=np.array([d['name'] for d in DIRS]), dir_vecs=M,
         dir_t=np.stack([d['ts'] for d in DIRS]),
         dir_idx=np.array([d['idx'] for d in DIRS]),
         dir_t_valid=np.array([d['t_valid'] for d in DIRS]),
         dir_at_tmax=np.array([d['at_tmax'] for d in DIRS]),
         dir_at_tmax_se=np.array([d['at_tmax_se'] for d in DIRS]),
         n_eff=n_eff,
         dir_drift=np.array([np.mean(d['angles']) if d['angles'] else np.nan
                             for d in DIRS]),
         designs=U_all, phis=PHI.numpy(), pred_folds=pred, pred=pred_tot, pred_std=pstd,
         ratio=ratio, ratio_se=ratio_se, pred_cache=walk_cache,
         sim=sim, sim_mean=smean, sim_mc_std=smc, sim_count_std=scount)
say('')
say(f'saved {report_path}, {args.out_dir}/robustness.npz and '
    f'{FIGS}/robustness_{{sensitivity,spectra,directions,walks,mc}}.png')
with open(report_path, 'w') as f:
    f.write('\n'.join(REPORT) + '\n')
