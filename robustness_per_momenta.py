"""Robustness resolved by muon kinematics: which muons drive the sensitivity, and do they agree.

robustness.py answers the question for the muon sample as a whole. That total hides the
thing a shield designer actually wants to know: the leak is not one population. Soft muons
are bent by the first magnets and hard ones by the last, so the design directions that
matter for one are not the directions that matter for the other, and a perturbation that
is harmless on average can be the worst case for the part of the spectrum that dominates
the background. This script splits the muons into --n_bins regions of --bin_var and redoes
the analysis inside each one:

  per bin      the hits and the hit rate, the gradient dH_b/du, the Hessian, the most
               sensitive parameters (first order and the exact one-at-a-time response)
  across bins  how much each bin contributes to the total gradient, and the cosines
               between the bins' own worst directions -- if those are far from 1, no
               single direction is "the" worst one and the bins are in conflict
  and then     every selected direction is walked with the surrogate and the simulator,
               and both are broken down by bin, so the plots of grad_test.py appear once
               per momentum region rather than once in total

The split is free. H_b, its gradient and its Hessian are sums over the muons of bin b, so
the frozen-trunk accumulators of robustness.py just need a bin index:

    T_b = sum_{x in b} w s_x t_x,  A_b = sum_{x in b} w s_x (1 - 2 p_x) t_x t_x^T

and then grad_b = J^T T_b and hess_b = J^T A_b J + d2(b.T_b)/du2 exactly, with J and the
branch Hessian taken by autograd on the branch net alone. One pass over the muons gives
every bin at once, and sum_b grad_b is the total gradient (printed as a check, together
with the usual --check against LCSONet.grad_phi).

Everything is in the NORMALIZED frame: u = 0 is the nominal design, u_i = +-1 puts
parameter i on the face of the +-delta box. Per-bin quantities are reported relative to
that bin's own H_b, since the bins differ by orders of magnitude in how many hits they
carry and the absolute numbers would only show the population size.

    python robustness_per_momenta.py --no_sim --n_muons 20000000
    python robustness_per_momenta.py --bin_var pt --n_bins 5 --n_muons_sim 5000000
    python robustness_per_momenta.py --bin_var charge --dirs ascent,bin1,bin-1
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
parser.add_argument('--bin_var', default='pz', choices=['pz', 'pt', 'p', 'charge'],
                    help='What splits the muons: p_z, the transverse momentum, the total '
                         'momentum, or the sign of the pdg code (2 bins)')
parser.add_argument('--n_bins', type=int, default=6,
                    help='Number of bins (ignored for --bin_var charge)')
parser.add_argument('--edges', default=None,
                    help='Explicit interior bin edges, comma separated, in GeV. Without '
                         'this the edges are the quantiles of a sample of the muons, so '
                         'every bin holds the same number of them')
parser.add_argument('--n_edge', type=int, default=int(2e6),
                    help='Muons sampled across the whole set to find the quantile edges')
parser.add_argument('--muons', default='gen', choices=['gen', 'data'],
                    help="'gen' samples the generative flow (unweighted, what lcso.py "
                         "trains on); 'data' reads the real file (per-muon weights, which "
                         "every sum then carries)")
parser.add_argument('--temperature', type=float, default=1.0, help='gen: flow temperature')
parser.add_argument('--seed', type=int, default=0,
                    help='Base seed: the muon blocks (gen), the random direction and the '
                         'Monte Carlo')
parser.add_argument('--pz_min', type=float, default=None,
                    help='gen: reject sampled muons below this p_z (GeV)')
parser.add_argument('--pt_max', type=float, default=None,
                    help='gen: reject sampled muons above this p_t (default 14)')
parser.add_argument('--n_muons', type=int, default=int(2e7),
                    help='Muons for the derivatives (one pass; 0 = all of the file)')
parser.add_argument('--n_muons_sim', type=int, default=0,
                    help='Muons for the simulator walks (0 = same as --n_muons)')
parser.add_argument('--n_cache', type=int, default=int(1e6),
                    help='Muons whose trunk features and bin index are kept for the cheap '
                         'per-bin scans. Costs n_cache * p * 4 bytes on the device')
parser.add_argument('--dirs', default='ascent,bin1,bin-1,random',
                    help='Directions to walk, comma separated. ascent (+grad of the '
                         'total), descent, worst (trust-region maximizer of the total '
                         'quadratic model at radius --t_max), corner (sign(grad)), '
                         'bin<k> (the ascent direction of bin k alone, 1-based, negative '
                         'counts from the end), binworst<k> (its trust-region maximizer), '
                         'random (a control)')
parser.add_argument('--t_max', type=float, default=1.0,
                    help='Half-length of the walks, in normalized units (an L2 step)')
parser.add_argument('--n_points', type=int, default=5,
                    help='Points per walk (forced odd so the nominal design is one)')
parser.add_argument('--n_repeats', type=int, default=1,
                    help='Simulate each design this many times')
parser.add_argument('--no_sim', dest='sim', action='store_false',
                    help='Skip the simulator entirely: surrogate analysis only')
parser.add_argument('--no_check', dest='check', action='store_false',
                    help='Skip the autograd cross-check of the frozen-trunk derivatives')
parser.add_argument('--n_mc', type=int, default=2048,
                    help='Designs drawn uniformly in the tolerance box, for the per-bin '
                         'reference distribution of H_b/H_b0')
parser.add_argument('--top', type=int, default=12, help='Rows in the printed tables')
parser.add_argument('--block', type=int, default=int(2e7), help='Muons held in memory')
parser.add_argument('--chunk', type=int, default=2 ** 19, help='Muon chunk for the surrogate')
parser.add_argument('--out_dir', default='outputs')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
if args.n_cache < 1:
    parser.error('--n_cache must be at least 1: the per-bin scans run against the cache')

torch.backends.cuda.matmul.allow_tf32 = False       # see robustness.py

DEVICE = torch.device(args.device)
FIGS = os.path.join(args.out_dir, 'figs')
os.makedirs(FIGS, exist_ok=True)
COLS = ['px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight']
GEN = args.muons == 'gen'
OK, BAD, GREY = '#0072B2', '#D55E00', '0.6'

for attr, val in (('PZ_MIN', args.pz_min), ('PT_MAX', args.pt_max)):
    if val is not None:
        setattr(ShipMuonShieldCuda, attr, float(val))
        print(f'kinematic envelope: {attr} = {val}')

shield = ShipMuonShieldCuda(n_samples=1, uniform_fields=True)
phi_0 = shield.initial_phi                                   # PHYSICAL
model, cfg = load_surrogate(args.model, phi_0)
model.eval().to(DEVICE)
D = model.dim
labels = [f'M{m}:{shield.idx_mag[i].split("[")[0]}' for m, i in shield.params_idx.tolist()]

if GEN:
    shield._load_flow()          # before any seeding; see grad_test.py / robustness.py


def read_muons(lo, hi):
    """Muons [lo, hi) as (raw, weights); weights is None when there are none."""
    if GEN:
        torch.manual_seed(args.seed + lo)
        return shield.sample_gen(n_samples=hi - lo, temperature=args.temperature), None
    with h5py.File(shield.muons_file, 'r') as f:
        x = np.stack([np.asarray(f[k][lo:hi], dtype=np.float32) for k in COLS], axis=1)
    x[:, 3], x[:, 4], x[:, 5] = 0.0, 0.0, -1.0
    x = torch.from_numpy(x)
    return x, x[:, 7]


if GEN:
    n = args.n_muons if args.n_muons > 0 else int(5e7)
    print(f'muons: generative flow at temperature {args.temperature:g}, seed {args.seed} '
          f'(unweighted)')
else:
    with h5py.File(shield.muons_file, 'r') as f:
        n = f['px'].shape[0]
    n = n if args.n_muons <= 0 else min(args.n_muons, n)
    print(f'muons: {shield.muons_file} (weighted)')
n_sim = n if args.n_muons_sim <= 0 else min(args.n_muons_sim, n)
print(f'model: {os.path.basename(args.model)} {cfg["model_type"]}, p = {cfg["p"]}, '
      f'delta = {model.delta}, D = {D}')

# =============================================================================
# The bins
# =============================================================================
UNIT = {'pz': 'GeV', 'pt': 'GeV', 'p': 'GeV', 'charge': ''}[args.bin_var]


def bin_value(x):
    """The kinematic variable the bins are cut in, from RAW muons (..., >=7)."""
    if args.bin_var == 'pz':
        return x[:, 2]
    if args.bin_var == 'pt':
        return x[:, :2].norm(dim=1)
    if args.bin_var == 'p':
        return x[:, :3].norm(dim=1)
    return torch.sign(x[:, 6])                     # pdg: +13 is mu-, -13 is mu+


if args.bin_var == 'charge':
    EDGES = torch.tensor([0.0])                    # sign < 0 | sign > 0
    BIN_LABELS = ['mu+ (pdg<0)', 'mu- (pdg>0)']
else:
    if args.edges:
        inner = sorted(float(s) for s in args.edges.split(','))
    else:
        # Quantiles over a sample spread across the whole set: the real file is not in a
        # random order, so taking them from the first block would bias the edges.
        want = min(args.n_edge, n)
        n_slices = 20 if not GEN else 1
        per = max(1, want // n_slices)
        vals = []
        for k in range(n_slices):
            lo = min(n - per, (k * (n - per)) // max(1, n_slices - 1)) if n_slices > 1 else 0
            xb, _ = read_muons(lo, min(lo + per, n))
            vals.append(bin_value(xb))
        vals = torch.cat(vals)
        if vals.numel() > 2 ** 23:              # torch.quantile refuses very large inputs
            vals = vals[::vals.numel() // 2 ** 23]
        qs = torch.linspace(0, 1, args.n_bins + 1)[1:-1]
        inner = torch.quantile(vals.double(), qs.double()).tolist()
        print(f'quantile edges from {vals.numel():,} muons: '
              + ', '.join(f'{v:.1f}' for v in inner))
        del vals
    EDGES = torch.tensor(inner, dtype=torch.float32)
    ends = ['0'] + [f'{v:.1f}' for v in EDGES.tolist()] + ['inf']
    BIN_LABELS = [f'{args.bin_var} {ends[k]}-{ends[k + 1]}' for k in range(len(ends) - 1)]
B = len(BIN_LABELS)
SHORT = [lab.split()[-1] if args.bin_var != 'charge' else lab.split()[0]
         for lab in BIN_LABELS]
EDGES_D = EDGES.to(DEVICE)


def bin_index(x, edges):
    """Bin of each RAW muon: [0, B). Everything outside falls in the end bins."""
    return torch.bucketize(bin_value(x).contiguous(), edges)


print(f'{B} bins in {args.bin_var}: ' + ' | '.join(BIN_LABELS))

# =============================================================================
# One pass: the frozen-trunk accumulators PER BIN, and the feature cache
# =============================================================================
u_0 = torch.zeros(D, device=DEVICE)
b_0 = model.branch_net(u_0.unsqueeze(0)).squeeze(0).detach()   # (p,)
P = b_0.numel()
T_acc = torch.zeros(B, P, dtype=torch.float64, device=DEVICE)
A_acc = torch.zeros(B, P, P, dtype=torch.float64, device=DEVICE)
H_b = np.zeros(B)               # sum_x w p        expected hits of the bin
var_b = np.zeros(B)             # sum_x w^2 p(1-p) its Bernoulli spread
cnt_b = np.zeros(B)             # muons in the bin
wsum_b = np.zeros(B)            # sum_x w in the bin
cache_t, cache_w, cache_b = [], [], []
n_cached = 0
t0 = time.time()

for lo in range(0, n, args.block):
    raw, w = read_muons(lo, min(lo + args.block, n))
    with torch.no_grad():
        for c in range(0, raw.shape[0], args.chunk):
            xb = raw[c:c + args.chunk, :7].to(DEVICE)
            ib = bin_index(xb, EDGES_D)                                  # (chunk,)
            tb = model.trunk_net(model.normalize_muons(xb))               # (chunk, p)
            pr = torch.sigmoid(tb @ b_0 + model.bias)
            s = pr * (1 - pr)
            if w is None:
                wb = torch.ones_like(pr)
            else:
                wb = w[c:c + args.chunk].to(DEVICE)
            ws = s * wb
            cnt_b += np.bincount(ib.cpu().numpy(), minlength=B)
            wsum_b += torch.zeros(B, dtype=torch.float64, device=DEVICE).index_add_(
                0, ib, wb.double()).cpu().numpy()
            H_b += torch.zeros(B, dtype=torch.float64, device=DEVICE).index_add_(
                0, ib, (pr * wb).double()).cpu().numpy()
            var_b += torch.zeros(B, dtype=torch.float64, device=DEVICE).index_add_(
                0, ib, (s * wb ** 2).double()).cpu().numpy()
            T_acc.index_add_(0, ib, (ws.unsqueeze(1) * tb).double())
            coef = ws * (1 - 2 * pr)
            for k in range(B):                    # one small matmul per bin per chunk
                m = ib == k
                if bool(m.any()):
                    tk = tb[m]
                    A_acc[k] += ((tk * coef[m].unsqueeze(1)).T @ tk).double()
            if n_cached < args.n_cache:            # keep the features, not the muons
                j = min(args.n_cache - n_cached, tb.shape[0])
                cache_t.append(tb[:j].clone())
                cache_b.append(ib[:j].clone())
                if w is not None:
                    cache_w.append(wb[:j].clone())
                n_cached += j
    print(f'  derivatives: {min(lo + args.block, n):,}/{n:,} muons '
          f'({time.time() - t0:.0f}s)')
    del raw, w

Tc = torch.cat(cache_t)                                # (n_cached, p)
bc = torch.cat(cache_b)                                # (n_cached,)
wc = torch.cat(cache_w) if cache_w else None
del cache_t, cache_w, cache_b

# The design side, exactly, by autograd on the branch net alone: one Jacobian, and one
# branch Hessian per bin (43 inputs each, no muons involved).
J = torch.autograd.functional.jacobian(
    lambda u: model.branch_net(u.unsqueeze(0)).squeeze(0), u_0).detach()            # (p, D)
Jd = J.double()
grad_b = (T_acc @ Jd).cpu().numpy()                                    # (B, D) dH_b/du
hess_b = np.zeros((B, D, D))
for k in range(B):
    Hb = torch.autograd.functional.hessian(
        lambda u: model.branch_net(u.unsqueeze(0)).squeeze(0) @ T_acc[k].float(),
        u_0).detach()
    hk = (Jd.T @ A_acc[k] @ Jd + Hb.double()).cpu().numpy()
    hess_b[k] = 0.5 * (hk + hk.T)
grad = grad_b.sum(0)                                   # the total, as in robustness.py
hess = hess_b.sum(0)
H_tot, var_tot = H_b.sum(), var_b.sum()

print(f'\nH_0 = {H_tot:,.2f} {"weighted " if not GEN else ""}hits over '
      f'{int(cnt_b.sum()):,} muons (total weight {wsum_b.sum():,.1f}), Bernoulli spread '
      f'{np.sqrt(var_tot):,.2f}')
print(f'cached {n_cached:,} muons ({n_cached * P * 4 / 2 ** 20:.0f} MB); the design enters '
      f'through {P} branch features')

if args.check:
    xb, wb = read_muons(0, min(2 ** 15, n))
    xd = xb[:, :7].to(DEVICE)
    wd = None if wb is None else wb.to(DEVICE)
    with torch.no_grad():
        tb = model.trunk_net(model.normalize_muons(xd))
        pr = torch.sigmoid(tb @ b_0 + model.bias)
        sw = pr * (1 - pr) * (1.0 if wd is None else wd)
    jac = (2.0 / (model.upper_bound - model.lower_bound)).cpu()          # du/dphi
    g_auto = model.grad_phi(phi_0.to(DEVICE), xd, wd).cpu() / jac
    g_frz = (J.T @ (sw @ tb)).cpu()
    print(f'check on {xd.shape[0]:,} muons: grad rel. err '
          f'{float((g_frz - g_auto).norm() / g_auto.norm()):.2e} (float32 noise is ~1e-6); '
          f'the bins add up to the total by construction, to '
          f'{np.linalg.norm(grad_b.sum(0) - grad) / max(np.linalg.norm(grad), 1e-30):.1e}')
    del xb, wb, xd, wd

# =============================================================================
# Cheap per-bin surrogate evaluation against the cached features
# =============================================================================
CACHE_BUDGET = 2 ** 24


@torch.no_grad()
def hits_cached_abs(U):
    """Expected (weighted) hits PER BIN of NORMALIZED designs U (m, D): (m, B).

    The trunk features and the bin of every cached muon are already known, so this is a
    matmul and a scatter-add.
    """
    U = U.to(DEVICE)
    out = torch.zeros(U.shape[0], B, dtype=torch.float64, device=DEVICE)
    m_batch = max(1, min(U.shape[0], CACHE_BUDGET // max(1, n_cached)))
    for i in range(0, U.shape[0], m_batch):
        b = model.branch_net(U[i:i + m_batch])                       # (m, p)
        step = max(1, CACHE_BUDGET // b.shape[0])
        for c in range(0, n_cached, step):
            pr = torch.sigmoid(b @ Tc[c:c + step].T + model.bias)     # (m, chunk)
            if wc is not None:
                pr = pr * wc[c:c + step]
            out[i:i + b.shape[0]].index_add_(1, bc[c:c + step], pr.double())
    return out.cpu()


H_0c = hits_cached_abs(torch.zeros(1, D)).numpy()[0]                  # (B,) the cache's H_b
rate_c = H_0c.sum() / (float(wc.double().sum()) if wc is not None else float(n_cached))
print(f'the cached subsample carries a nominal hit rate of {rate_c:.4e} against '
      f'{H_tot / wsum_b.sum():.4e} over the whole sample '
      f'({rate_c / (H_tot / wsum_b.sum()) - 1:+.1%})')


def hits_cached(U):
    """H_b(u)/H_b(0) per bin -- as a ratio the subsampling cancels."""
    return hits_cached_abs(U).numpy() / np.maximum(H_0c, 1e-30)

# =============================================================================
# Analysis
# =============================================================================
Hn = np.maximum(H_b, 1e-30)                    # guard the empty-bin division
gn = np.linalg.norm(grad_b, axis=1)
ghat_b = grad_b / np.maximum(gn, 1e-30)[:, None]
ghat = grad / max(np.linalg.norm(grad), 1e-30)
mu_b = np.array([np.linalg.eigvalsh(hess_b[k])[-1] for k in range(B)])
curv_b = np.array([ghat_b[k] @ hess_b[k] @ ghat_b[k] for k in range(B)])

print(f'\n{"bin":18s} {"muons":>12s} {"weight":>9s} {"hits":>12s} {"hit share":>10s} '
      f'{"rate":>10s} {"|g|/H":>8s} {"curv/H":>8s} {"maxeig/H":>9s} {"cos(g_tot)":>11s}')
for k in range(B):
    print(f'{BIN_LABELS[k]:18s} {int(cnt_b[k]):12,d} {wsum_b[k] / wsum_b.sum():8.1%} '
          f'{H_b[k]:12,.1f} {H_b[k] / max(H_tot, 1e-30):9.1%} '
          f'{H_b[k] / max(wsum_b[k], 1e-30):10.2e} {gn[k] / Hn[k]:8.2f} '
          f'{curv_b[k] / Hn[k]:8.2f} {mu_b[k] / Hn[k]:9.2f} {ghat_b[k] @ ghat:11.3f}')
print(f'{"total":18s} {int(cnt_b.sum()):12,d} {1.0:8.1%} {H_tot:12,.1f} {1.0:9.1%} '
      f'{H_tot / wsum_b.sum():10.2e} {np.linalg.norm(grad) / max(H_tot, 1e-30):8.2f} '
      f'{ghat @ hess @ ghat / max(H_tot, 1e-30):8.2f} '
      f'{np.linalg.eigvalsh(hess)[-1] / max(H_tot, 1e-30):9.2f} {1.0:11.3f}')
print('  |g|/H is the relative sensitivity of that bin: the fractional change in ITS hits '
      'for a full\n  tolerance step along its own worst direction. cos(g_tot) says whether '
      'that direction is the\n  one the total gradient points along.')

# --- exact one-at-a-time, per bin -------------------------------------------
E = torch.cat([torch.eye(D), -torch.eye(D)])
oat = hits_cached(E)                                    # (2D, B)
oat_p, oat_m = oat[:D] - 1.0, oat[D:] - 1.0             # relative change at u_i = +-1
oat_worst = np.maximum(oat_p, oat_m)                    # the bad side of each parameter
rel = grad_b / Hn[:, None]                              # (B, D) first-order, per bin
order = np.argsort(-np.abs(rel).max(0))                 # worst over any bin, first

print(f'\nmost sensitive parameters, by bin (exact one-at-a-time, the worse of u_i = +-1, '
      f'as a fraction of that bin\'s hits)')
print(f'{"parameter":22s} ' + ' '.join(f'{lab:>10s}' for lab in SHORT) + f' {"total":>10s}')
# The bins partition the muons, so the total ratio is the per-bin ratios re-weighted by
# the nominal hits of each bin.
oat_tot = np.maximum(oat[:D] @ H_0c, oat[D:] @ H_0c) / max(H_0c.sum(), 1e-30) - 1
for k in order[:args.top]:
    print(f'{labels[k]:22s} ' + ' '.join(f'{oat_worst[k, j]:+10.3f}' for j in range(B))
          + f' {oat_tot[k]:+10.3f}')

print('\nthe bin each parameter matters most to (first order):')
for k in order[:args.top]:
    j = int(np.argmax(np.abs(rel[:, k])))
    print(f'  {labels[k]:22s} {BIN_LABELS[j]:18s} {rel[j, k]:+7.3f} '
          f'(total {grad[k] / max(H_tot, 1e-30):+.3f})')

# --- do the bins want the same thing? ---------------------------------------
COS = ghat_b @ ghat_b.T
print('\ncosine between the bins\' own worst directions:')
print(' ' * 20 + ' '.join(f'{j + 1:>7d}' for j in range(B)))
for k in range(B):
    print(f'{BIN_LABELS[k]:18s} ' + ' '.join(f'{COS[k, j]:+7.2f}' for j in range(B)))
off = COS[~np.eye(B, dtype=bool)]
print(f'  off-diagonal cosine: min {off.min():+.2f}, median {np.median(off):+.2f}, '
      f'max {off.max():+.2f}')
print('  values near 1 mean one direction is worst for every region; small or negative '
      'ones mean\n  the regions are in conflict and no single walk can represent them.')


def trust_region_max(g, Bm, radius):
    """argmax of g.v + 0.5 v^T Bm v over ||v|| <= radius (Moré-Sorensen; see robustness.py)."""
    ev, Q = np.linalg.eigh(Bm)
    gh = Q.T @ g
    top = ev[-1]
    if top < 0:
        v = -np.linalg.solve(Bm, g)
        if np.linalg.norm(v) <= radius:
            return v

    def v_of(lam):
        return Q @ (gh / (lam - ev))

    lo = top + 1e-12 * max(1.0, abs(top))
    if np.linalg.norm(v_of(lo)) < radius:
        keep = ev < top - 1e-12 * max(1.0, abs(top))
        v = Q[:, keep] @ (gh[keep] / (top - ev[keep]))
        alpha = np.sqrt(max(radius ** 2 - v @ v, 0.0))
        return v + alpha * Q[:, -1] * (1.0 if gh[-1] >= 0 else -1.0)
    hi = top + max(1e-12, np.linalg.norm(g) / radius)
    while np.linalg.norm(v_of(hi)) > radius:
        hi = top + 2 * (hi - top)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.linalg.norm(v_of(mid)) > radius else (lo, mid)
    return v_of(0.5 * (lo + hi))


# --- reference distribution over the box, per bin ---------------------------
gen_mc = torch.Generator().manual_seed(args.seed + 2)
U_mc = torch.rand(args.n_mc, D, generator=gen_mc) * 2 - 1
mc = hits_cached(U_mc)                                   # (n_mc, B)
print(f'\n{args.n_mc:,} designs uniform in the tolerance box, H_b/H_b0 per bin:')
print(f'{"bin":18s} {"median":>9s} {"95%":>9s} {"max":>9s}')
for k in range(B):
    print(f'{BIN_LABELS[k]:18s} {np.median(mc[:, k]):9.3f} '
          f'{np.percentile(mc[:, k], 95):9.3f} {mc[:, k].max():9.3f}')

# =============================================================================
# The directions to walk
# =============================================================================
def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-30)


rng = np.random.default_rng(args.seed + 3)
CATALOG = {
    'ascent': lambda: (unit(grad), 'first-order worst for the total'),
    'descent': lambda: (unit(-grad), 'descent for the total'),
    'worst': lambda: (unit(trust_region_max(grad, hess, args.t_max)),
                      f'second-order worst for the total at $t={args.t_max:g}$'),
    'corner': lambda: (unit(np.sign(grad)), 'worst box corner (first order)'),
    'random': lambda: (unit(rng.standard_normal(D)), 'random direction (control)'),
}
def bin_arg(s, name):
    """'1' is the first bin and '-1' the last, as in Python indexing."""
    k = int(s)
    k = k - 1 if k > 0 else k
    if not -B <= k < B:
        raise ValueError(f'{name!r} asks for a bin outside 1..{B} (or -1..-{B})')
    return k % B


DIRS = []
for name in [s.strip() for s in args.dirs.split(',') if s.strip()]:
    if name in CATALOG:
        v, title = CATALOG[name]()
    elif name.startswith('binworst') and name[8:].lstrip('-').isdigit():
        k = bin_arg(name[8:], name)
        v = unit(trust_region_max(grad_b[k], hess_b[k], args.t_max))
        title = f'second-order worst for {BIN_LABELS[k]}'
    elif name.startswith('bin') and name[3:].lstrip('-').isdigit():
        k = bin_arg(name[3:], name)
        v, title = unit(grad_b[k]), f'worst for {BIN_LABELS[k]} alone'
    else:
        raise ValueError(f'unknown direction {name!r}; pick from {list(CATALOG)} or '
                         f'bin<k> / binworst<k> with k in 1..{B} (or negative)')
    DIRS.append({'name': name, 'v': v, 'title': title})

n_pts = args.n_points + (args.n_points + 1) % 2              # odd -> t = 0 on the grid
ts = np.linspace(-args.t_max, args.t_max, n_pts)
print(f'\n{"direction":12s} {"cos(g_tot)":>10s}   per-bin slope (dH_b/dt)/H_b')
for d in DIRS:
    d['ts'] = ts
    d['slope_b'] = grad_b @ d['v']
    d['curv_b'] = np.array([d['v'] @ hess_b[k] @ d['v'] for k in range(B)])
    print(f'{d["name"]:12s} {ghat @ d["v"]:+10.3f}   '
          + ' '.join(f'{d["slope_b"][k] / Hn[k]:+6.2f}' for k in range(B)))
print('  a direction with mixed signs makes some muons worse and others better: that is '
      'the trade-off\n  the total gradient averages away.')

# Every walk passes through the nominal design, so it is index 0 and is simulated once.
U_list = [torch.zeros(D)]
for d in DIRS:
    d['idx'] = []
    for t in d['ts']:
        if abs(t) < 1e-12:
            d['idx'].append(0)
        else:
            d['idx'].append(len(U_list))
            U_list.append(torch.from_numpy(t * d['v']).float())
U_raw = torch.stack(U_list)
U_all = U_raw.clamp(-1, 1)
if int((U_raw.abs() > 1 + 1e-6).sum()):
    print(f'note: {int((U_raw.abs() > 1 + 1e-6).sum())} components fall outside the box '
          f'and are clamped to its faces')
PHI = model.denormalize_phi(U_all.to(DEVICE)).cpu()
n_des, R = len(U_list), max(1, args.n_repeats)
walk_cache = hits_cached(U_all)                              # (n_des, B), free

# =============================================================================
# Walk the directions with the simulator and the surrogate, binning both
# =============================================================================
pred = np.zeros((n_des, B))        # sum_x w p        expected hits, per bin
pvar = np.zeros((n_des, B))        # sum_x w^2 p(1-p) its Bernoulli spread
sim = np.zeros((R, n_des, B))      # sum_x w hit      simulated hits, per bin per repeat
svar = np.zeros((R, n_des, B))     # sum_x w^2 hit    its counting error
if args.sim:
    print(f'\n{n_des} designs x {n_sim:,} muons x {R} repeats = '
          f'{n_des * n_sim * R / 1e9:.1f}e9 tracks to simulate')
    PHI_D = PHI.to(DEVICE)
    t0 = time.time()
    for lo in range(0, n_sim, args.block):
        raw, w = read_muons(lo, min(lo + args.block, n_sim))
        ib_cpu = bin_index(raw, EDGES)                       # the block's bins, once
        with torch.no_grad():
            for c in range(0, raw.shape[0], args.chunk):
                xb = raw[c:c + args.chunk, :7].to(DEVICE)
                ib = bin_index(xb, EDGES_D)
                pr = model.predict_proba(PHI_D, xb.unsqueeze(0))     # (n_des, chunk)
                q = pr * (1 - pr)
                if w is not None:
                    wb = w[c:c + args.chunk].to(DEVICE)
                    pr, q = pr * wb, q * wb ** 2
                acc = torch.zeros(n_des, B, dtype=torch.float64, device=DEVICE)
                pred += acc.index_add_(1, ib, pr.double()).cpu().numpy()
                pvar += acc.zero_().index_add_(1, ib, q.double()).cpu().numpy()
        wd = torch.ones(raw.shape[0], dtype=torch.float64) if w is None else w.double()
        for i in range(n_des):
            for r in range(R):
                hits = shield(PHI[i], raw).double()   # a new transport seed every call
                acc = torch.zeros(B, dtype=torch.float64)
                sim[r, i] += acc.index_add_(0, ib_cpu, hits * wd).numpy()
                svar[r, i] += acc.zero_().index_add_(0, ib_cpu, hits * wd ** 2).numpy()
        print(f'  walk: {min(lo + args.block, n_sim):,}/{n_sim:,} muons '
              f'({time.time() - t0:.0f}s)')
        del raw, w, wd, ib_cpu

pstd = np.sqrt(pvar)
smean = sim.mean(0)
smc = sim.std(0, ddof=1) if R > 1 else np.zeros((n_des, B))
scount = np.sqrt(svar).mean(0)

if args.sim:
    print('\nH_b at the far end of each walk, relative to the nominal design '
          '(surrogate | simulator):')
    print(f'{"direction":12s} ' + ' '.join(f'{lab:>15s}' for lab in SHORT)
          + f' {"total":>15s}')
    for d in DIRS:
        i = d['idx'][-1]
        cells = []
        for k in list(range(B)) + [None]:
            if k is None:
                p_r = pred[i].sum() / max(pred[0].sum(), 1e-30)
                s_r = smean[i].sum() / max(smean[0].sum(), 1e-30)
            else:
                p_r = pred[i, k] / max(pred[0, k], 1e-30)
                s_r = smean[i, k] / max(smean[0, k], 1e-30)
            cells.append(f'{p_r:6.2f}|{s_r:<6.2f}'.rjust(15))
        print(f'{d["name"]:12s} ' + ' '.join(cells))

# =============================================================================
# Figures
# =============================================================================
HITS = 'hits' if GEN else 'weighted hits'
SRC = (f'generative muons, T = {args.temperature:g}' if GEN else 'real muons, weighted')
CMAP = plt.get_cmap('viridis')
BCOL = [CMAP(k / max(B - 1, 1) * 0.9) for k in range(B)]
x_b = np.arange(B)

# --- 1. what each region is and how sensitive it is -------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
ax[0, 0].bar(x_b, wsum_b / wsum_b.sum(), color=BCOL)
ax[0, 0].set(ylabel='share of the muon weight', title='Population')
ax[0, 1].bar(x_b, H_b / max(H_tot, 1e-30), color=BCOL)
ax[0, 1].set(ylabel='share of the hits', title='Where the hits come from')
ax[1, 0].bar(x_b, gn / Hn, color=BCOL)
ax[1, 0].set(ylabel='$|\\nabla H_b| / H_b$', title='Relative sensitivity')
ax[1, 1].bar(x_b, curv_b / Hn, color=[BAD if v > 0 else OK for v in curv_b])
ax[1, 1].axhline(0, color='k', lw=0.8)
ax[1, 1].set(ylabel='$v^T \\nabla^2 H_b v / H_b$ along its own $v$', title='Curvature')
for a in ax.ravel():
    a.set_xticks(x_b, SHORT, rotation=30, ha='right', fontsize=8)
fig.suptitle(f'Muon regions in {args.bin_var} ({SRC})')
fig.savefig(os.path.join(FIGS, 'robustness_bins_overview.png'), dpi=150)
plt.close(fig)

# --- 2. which parameters matter to which region -----------------------------
o = order[::-1]
fig, ax = plt.subplots(1, 2, figsize=(4 + 0.9 * B, 10), constrained_layout=True,
                       sharey=True)
for a, M, ttl in ((ax[0], rel[:, o].T, 'First order: $(\\partial H_b/\\partial u_i)/H_b$'),
                  (ax[1], oat_worst[o], 'Exact one-at-a-time (worse side)')):
    lim = np.abs(M).max()
    im = a.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    a.set_xticks(x_b, SHORT, rotation=45, ha='right', fontsize=8)
    a.set_title(ttl, fontsize=10)
    fig.colorbar(im, ax=a)
ax[0].set_yticks(np.arange(D), [labels[k] for k in o], fontsize=7)
fig.suptitle(f'Sensitivity per parameter and muon region ({SRC})')
fig.savefig(os.path.join(FIGS, 'robustness_bins_sensitivity.png'), dpi=150)
plt.close(fig)

# --- 3. do the regions agree on the worst direction? ------------------------
fig, ax = plt.subplots(figsize=(1.6 + 0.8 * B, 1.2 + 0.8 * B), constrained_layout=True)
im = ax.imshow(COS, cmap='RdBu_r', vmin=-1, vmax=1)
for k in range(B):
    for j in range(B):
        ax.text(j, k, f'{COS[k, j]:+.2f}', ha='center', va='center', fontsize=7,
                color='k' if abs(COS[k, j]) < 0.6 else 'w')
ax.set_xticks(x_b, SHORT, rotation=45, ha='right', fontsize=8)
ax.set_yticks(x_b, SHORT, fontsize=8)
fig.colorbar(im, ax=ax, label='cosine')
ax.set_title('Alignment of the regions\' own worst directions')
fig.savefig(os.path.join(FIGS, 'robustness_bins_alignment.png'), dpi=150)
plt.close(fig)

# --- 4. the walks, one panel per direction, one line per region -------------
nc = min(3, len(DIRS))
nr = int(np.ceil(len(DIRS) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(4.8 * nc, 4.0 * nr), constrained_layout=True,
                         squeeze=False)
for a in axes.ravel()[len(DIRS):]:
    a.axis('off')
for a, d in zip(axes.ravel(), DIRS):
    idx = np.array(d['idx'])
    for k in range(B):
        base = pred[0, k] if (args.sim and pred[0, k] > 0) else H_0c[k]
        curve = (pred[idx, k] / base if args.sim else walk_cache[idx, k])
        a.plot(d['ts'], curve, '-o', ms=3, color=BCOL[k], lw=1.5, label=SHORT[k])
        if args.sim:
            e = (smc if R > 1 else scount)[idx, k] / max(smean[0, k], 1e-30)
            a.errorbar(d['ts'], smean[idx, k] / max(smean[0, k], 1e-30), yerr=e, fmt='s',
                       ms=4, color=BCOL[k], capsize=2, lw=1, alpha=0.85, label='_')
    tot = ((pred[idx].sum(1) / max(pred[0].sum(), 1e-30)) if args.sim
           else (walk_cache[idx] * H_0c).sum(1) / H_0c.sum())
    a.plot(d['ts'], tot, '--', color='k', lw=2, label='total')
    a.axvline(0, color=GREY, lw=1, ls='--')
    a.axhline(1, color=GREY, lw=0.8, ls=':')
    a.set_yscale('log')
    a.set_title(f'{d["name"]}: {d["title"]}', fontsize=9)
    a.set_xlabel('step $t$ along $v$ (normalized units)')
    a.set_ylabel('$H_b(t)\\,/\\,H_b(0)$')
    if a is axes.ravel()[0]:
        a.legend(fontsize=7, ncol=2)
fig.suptitle(f'Response by muon region along each direction -- lines are the surrogate, '
             f'squares the simulator ({SRC})' if args.sim else
             f'Response by muon region along each direction (surrogate, {SRC})')
fig.savefig(os.path.join(FIGS, 'robustness_bins_walks.png'), dpi=150)
plt.close(fig)

# --- 5. the box as a whole, per region --------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
edges_mc = np.logspace(np.log10(max(mc.min(), 1e-8) * 0.9), np.log10(mc.max() * 1.2), 50)
for k in range(B):
    ax.hist(np.clip(mc[:, k], edges_mc[0], edges_mc[-1]), bins=edges_mc, histtype='step',
            lw=1.6, color=BCOL[k], label=SHORT[k])
ax.axvline(1.0, color='k', lw=1.2, ls='--', label='nominal')
ax.set_xscale('log')
ax.set(xlabel='$H_b/H_b(0)$', ylabel='designs',
       title=f'{args.n_mc:,} designs uniform in the $\\pm${model.delta:.0%} box, by region')
ax.legend(fontsize=8, ncol=2)
fig.savefig(os.path.join(FIGS, 'robustness_bins_mc.png'), dpi=150)
plt.close(fig)

np.savez(os.path.join(args.out_dir, 'robustness_per_momenta.npz'),
         labels=np.array(labels), bin_labels=np.array(BIN_LABELS), bin_var=args.bin_var,
         edges=EDGES.numpy(), phi_0=phi_0.numpy(), muons=args.muons,
         temperature=args.temperature, H_b=H_b, var_b=var_b, counts=cnt_b, weights=wsum_b,
         grad_b=grad_b, hess_b=hess_b, grad=grad, hess=hess, cos=COS,
         oat_plus=oat_p, oat_minus=oat_m, mc=mc, H_0_cache=H_0c,
         dir_names=np.array([d['name'] for d in DIRS]),
         dir_vecs=np.stack([d['v'] for d in DIRS]), t=ts,
         dir_idx=np.array([d['idx'] for d in DIRS]), designs=U_all.numpy(),
         phis=PHI.numpy(), pred=pred, pred_std=pstd, pred_cache=walk_cache, sim=sim,
         sim_mean=smean, sim_mc_std=smc, sim_count_std=scount)
print(f'\nsaved {FIGS}/robustness_bins_{{overview,sensitivity,alignment,walks,mc}}.png and '
      f'{args.out_dir}/robustness_per_momenta.npz')
