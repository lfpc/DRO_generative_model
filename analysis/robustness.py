"""Robustness of the nominal muon-shield design, read off the LCSO surrogate.

Per-parameter sensitivity (gradient, curvature, exact one-at-a-time), Hessian spectrum, active
subspace, worst directions, random designs in the box, and walks along the directions checked
with the simulator. Writes <out_dir>/robustness.npz and robustness_report.txt; plots.py draws
the figures.

The surrogate is integrated over the muons (for the flow on quadrature grids, not sampled):
the derivatives on --grid, everything else on the cached --cache_grid. Hits are quoted per
--n_ref muons. The errors are the surrogate's own: the standard error of the ensemble's
members (lcso/train.py --stack), sd / sqrt(K). The simulator walks use --n_muons_sim
sampled muons, with Poisson errors.

Directions (--dirs): ascent (+grad), descent, worst (trust-region maximizer of the quadratic
model at --t_max), corner (sign of grad), pga (worst design in the box, by projected-gradient
ascent on the surrogate), random, hess<k> and act<k> (k-th Hessian / active eigenvector).

    python -m analysis.robustness --no_sim                                   # surrogate only, minutes
    python -m analysis.robustness --n_muons_sim 1000000000                   # with the simulated walks
"""
import os
from types import SimpleNamespace

import numpy as np
import torch

from analysis.core import (Report, branch_jacobian, cached_hits, derivatives, feature_cache, muon_blocks, nominal_pass,
                           one_group, seed_se, setup, simulate, trust_region_max, unit)
import cli
from lcso.surrogate import load_members
from paths import RUN

parser = cli.add_common(cli.parser(__doc__))
parser.add_argument('--temperature', type=float, default=1.0, help='momentum temperature of the muons; 1 = the physical flux')
parser.add_argument('--grid', default='801x64', help='muon points of the derivatives (muons.py: for the flow, latent points per side x azimuths)')
parser.add_argument('--cache_grid', default='201x64', help='muon points whose trunk features are cached')
parser.add_argument('--n_ref', type=float, default=1e9, help='hits are quoted per this many muons')
parser.add_argument('--n_muons_sim', type=int, default=int(1e9), help='sampled muons for the simulated walks')
parser.add_argument('--seed', type=int, default=0, help='seed of the sampled muons')
parser.add_argument('--n_sigma', type=float, default=3.0, help='significance for "resolved"')
parser.add_argument('--dirs', default='ascent,worst,act1,act2,hess1,pga,corner,random')
parser.add_argument('--t_max', type=float, default=1.0, help='walk half-length (L2, normalized)')
parser.add_argument('--n_points', type=int, default=5, help='points per walk (made odd)')
parser.add_argument('--no_sim', dest='sim', action='store_false', help='walks on the surrogate only')
parser.add_argument('--n_mc', type=int, default=4096, help='random designs in the box')
parser.add_argument('--pga_restarts', type=int, default=4,
                    help='starts of the box search: nominal, sign(grad)/2, worst direction, then random')
parser.add_argument('--pga_steps', type=int, default=200)
parser.add_argument('--pga_lr', type=float, default=0.05)
parser.add_argument('--n_scan', type=int, default=8, help='parameters scanned one at a time')
parser.add_argument('--plane', default='worst,act1', help='two directions spanning the scanned plane')
parser.add_argument('--plane_n', type=int, default=31, help='points per side of the plane')
parser.add_argument('--top', type=int, default=15, help='rows in the printed tables')
parser.add_argument('--block', type=int, default=int(2e7), help='sampled muons held in memory (simulator)')
parser.add_argument('--out_dir', default=RUN)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
say = Report()
NS = args.n_sigma
shield, model = setup(args)
MEMBERS = load_members(args.model, shield.phi_0, args.device)
D, labels = model.dim, shield.labels


def integrate(m, cache=True):
    """Everything taken from the grids for one network: H0, grad, hess, active matrix, cache."""
    acc = nominal_pass(m, shield.muons.points(args.grid, args.n_ref, args.temperature), one_group, 1)
    J = branch_jacobian(m)
    grad, hess = derivatives(m, J, acc.T[0], acc.A[0])
    out = SimpleNamespace(H0=float(acc.H[0]), grad=grad, hess=hess,
                          act=((J.T @ acc.S[0] @ J) / acc.N[0]).cpu().numpy())
    if cache:
        out.cache = feature_cache(m, shield.muons.points(args.cache_grid, args.n_ref, args.temperature), one_group)
        out.H0c = float(cached_hits(m, out.cache, 1, np.zeros((1, D)))[0, 0])
    return out


def ratio_of(m, I, U):
    """H(u)/H(0) of designs U (m, D) for network m, on its cache."""
    return cached_hits(m, I.cache, 1, U)[:, 0] / I.H0c


def pm(value, se, fmt='.3f'):
    return f'{value:{fmt}} +- {se:{fmt}}'


FLUX = f'muons {args.muons}' + (f' at momentum temperature {args.temperature:g}' if args.temperature != 1 else '')
say(f'model {args.model} ({len(MEMBERS)} members); {FLUX}, integrated on the points '
    f'{args.grid} (derivatives) and {args.cache_grid} (cache); '
    f'hits per {args.n_ref:.0e} muons; errors: the members\' standard error; "resolved" = {NS:g} sigma')
S = integrate(model)
grad, hess, act, H0 = S.grad, S.hess, S.act, S.H0
say(f'H_0 = {H0:.2f} hits per {args.n_ref:.0e} muons; the cache points reproduce it to {S.H0c / H0 - 1:+.1e}')


def ratio(U):
    return ratio_of(model, S, U)


# =============================================================================
# 1. Which parameters matter
# =============================================================================
OAT = np.concatenate([np.eye(D), -np.eye(D)])
oat = ratio(OAT)
oat_p, oat_m = oat[:D] - 1, oat[D:] - 1
order = np.argsort(-np.maximum(np.abs(oat_p), np.abs(oat_m)))
lin, quad = grad / H0, 0.5 * np.diag(hess) / H0
magnet = np.array([int(s[1]) for s in labels])

# =============================================================================
# 2. Curvature and the active subspace
# =============================================================================
mu, Qh = np.linalg.eigh(hess)
mu, Qh = mu[::-1].copy(), Qh[:, ::-1].copy()           # most positive (worst) first
nu, Qa = np.linalg.eigh(act)
nu, Qa = np.clip(nu[::-1], 0, None).copy(), Qa[:, ::-1].copy()
participation = float(nu.sum() ** 2 / (nu ** 2).sum())
g_hat = unit(grad)

# =============================================================================
# 3. The worst directions
# =============================================================================
t, w = S.cache.t, S.cache.w


def cached_h(u):
    """Differentiable hits of one design over the cache."""
    b = model.branch_net(u[None])[0]
    return sum((torch.sigmoid(t[c:c + 2 ** 20].float() @ b + model.bias).double() * w[c:c + 2 ** 20]).sum()
               for c in range(0, len(t), 2 ** 20))


v_trs = trust_region_max(grad, hess, args.t_max)
torch.manual_seed(args.seed + 1)
starts = [torch.zeros(D), torch.from_numpy(np.sign(grad)).float() * 0.5, torch.from_numpy(unit(v_trs)).float()]
starts = (starts + [torch.rand(D) * 2 - 1 for _ in range(args.pga_restarts)])[:args.pga_restarts]
best_h, u_star = -np.inf, torch.zeros(D)
for u0 in starts:
    u = u0.to(args.device).clamp(-1, 1).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=args.pga_lr)
    for _ in range(args.pga_steps + 1):
        opt.zero_grad()
        h = cached_h(u)
        if float(h) > best_h:
            best_h, u_star = float(h), u.detach().cpu().clone()
        (-h).backward()
        opt.step()
        with torch.no_grad():
            u.clamp_(-1, 1)                     # project back onto the box


def oriented(v, g):
    """Eigenvectors have an arbitrary sign: point them the way that raises the hits."""
    return unit(v) * (1 if g @ v >= 0 else -1)


def direction(name):
    """(unit vector, walk half-length)."""
    if name == 'ascent':
        return unit(grad), args.t_max
    if name == 'descent':
        return -unit(grad), args.t_max
    if name == 'worst':
        return unit(v_trs), args.t_max
    if name == 'corner':
        return unit(np.sign(grad)), args.t_max
    if name == 'pga':
        return unit(u_star.numpy()), float(u_star.norm())
    if name == 'random':
        return unit(np.random.default_rng(args.seed + 3).standard_normal(D)), args.t_max
    if name.startswith('hess'):
        return oriented(Qh[:, int(name[4:]) - 1], grad), args.t_max
    if name.startswith('act'):
        return oriented(Qa[:, int(name[3:]) - 1], grad), args.t_max
    raise ValueError(f'unknown direction {name!r}')


NAMES = args.dirs.split(',')
V = np.stack([direction(nm)[0] for nm in NAMES])
n_pts = args.n_points | 1                                # odd, so t = 0 is on the grid
TS = np.stack([np.linspace(-1, 1, n_pts) * direction(nm)[1] for nm in NAMES])
AT1 = np.clip(args.t_max * V, -1, 1)
designs = [np.zeros(D)]                                 # the walks; index 0: the nominal design
IDX = np.zeros(TS.shape, dtype=int)
for i, (v, ts) in enumerate(zip(V, TS)):
    for j, tt in enumerate(ts):
        if tt != 0:
            IDX[i, j] = len(designs)
            designs.append(np.clip(tt * v, -1, 1))      # walks past the box bend onto its faces
designs = np.stack(designs)
U_mc = (torch.rand(args.n_mc, D, generator=torch.Generator().manual_seed(args.seed + 2)) * 2 - 1).numpy()

# every design that needs a value: one-at-a-time, directions at t_max, walks, box worst
ALL = np.concatenate([OAT, AT1, designs, u_star.numpy()[None]])
cut = np.cumsum([0, len(OAT), len(AT1), len(designs), 1])


def values(m, I):
    """Per network: the quantities the errors are taken over."""
    r = ratio_of(m, I, ALL)
    return dict(H0=I.H0, lin=I.grad / I.H0, mu=np.linalg.eigvalsh(I.hess)[::-1] / I.H0,
                oat=r[cut[0]:cut[1]], at1=r[cut[1]:cut[2]], walk=r[cut[2]:cut[3]], star=r[cut[3]])


main = values(model, S)
mem = []
for k, m in enumerate(MEMBERS):
    Im = integrate(m)
    mem.append(values(m, Im))
    del Im
    print(f'  member {k + 1}/{len(MEMBERS)} integrated')
se = {key: seed_se([v[key] for v in mem]) for key in main}
oat_p_se, oat_m_se = se['oat'][:D], se['oat'][D:]
larger = np.abs(oat_p) > np.abs(oat_m)                     # the side with the larger effect, as plots.py
resolved = np.abs(np.where(larger, oat_p, oat_m)) > NS * np.where(larger, oat_p_se, oat_m_se)
if mem:   # the best and worst place of each parameter over the members
    key = np.array([np.maximum(np.abs(v['oat'][:D] - 1), np.abs(v['oat'][D:] - 1)) for v in mem])
    ranks = np.argsort(np.argsort(-key, axis=1), axis=1) + 1
    rank_lo, rank_hi = ranks.min(0), ranks.max(0)
else:
    rank_lo = rank_hi = np.argsort(order) + 1

say('\n' + '=' * 79 + '\n1. SENSITIVITY OF EACH PARAMETER  (one parameter moved to +-10%)\n' + '=' * 79)
say(f'{"parameter":20s} {"grad/H_0":>17s} {"curv/H_0":>9s} {"exact +1":>17s} {"exact -1":>17s} {"rank":>6s}')
for k in order[:args.top]:
    rank = f'{rank_lo[k]}' + (f'-{rank_hi[k]}' if rank_hi[k] != rank_lo[k] else '')
    say(f'{labels[k]:20s} {pm(lin[k], se["lin"][k]):>17s} {quad[k]:+9.3f} {pm(oat_p[k], oat_p_se[k]):>17s} '
        f'{pm(oat_m[k], oat_m_se[k]):>17s} {rank + ("" if resolved[k] else "?"):>6s}')
say(f'  {resolved.sum()}/{D} parameters have an effect resolved at {NS:g} sigma ("?" marks the rest); '
    f'rank is the best-worst place over the {len(mem)} members')
say('share of the gradient norm per magnet: ' + ', '.join(
    f'M{m} {np.linalg.norm(grad[magnet == m]) / np.linalg.norm(grad):.2f}' for m in np.unique(magnet)))

say('\n' + '=' * 79 + '\n2. CURVATURE AND ACTIVE DIRECTIONS\n' + '=' * 79)
say(f'|grad|/H_0 = {np.linalg.norm(grad) / H0:.3f} per normalized unit')
mu_se = se['mu'] * H0
say(f'Hessian eigenvalues: max {pm(mu[0] / H0, mu_se[0] / H0)} H_0, min {pm(mu[-1] / H0, mu_se[-1] / H0)} H_0; '
    f'{int((np.abs(mu) > NS * mu_se).sum())}/{D} resolved; top five ' + ', '.join(f'{x / H0:+.3f}' for x in mu[:5]))
say(f'active subspace: participation ratio {participation:.2f}; the first direction carries '
    f'{nu[0] / nu.sum():.1%}, and holds {(g_hat @ Qa[:, 0]) ** 2:.1%} of the gradient')

at_1, at_1_se = main['at1'], se['at1']
say('\n' + '=' * 79 + '\n3. THE WORST DIRECTIONS\n' + '=' * 79)
say(f'{"direction":10s} {"cos(grad)":>9s} {"slope/H_0":>9s} {"curv/H_0":>9s} {f"H/H_0 at t={args.t_max:g}":>17s}  leading parameters')
for nm, v, r, s in zip(NAMES, V, at_1, at_1_se):
    say(f'{nm:10s} {g_hat @ v:+9.3f} {grad @ v / H0:+9.3f} {v @ hess @ v / H0:+9.3f} {pm(r, s, ".4g"):>17s}  '
        + ', '.join(f'{labels[k]} {v[k]:+.2f}' for k in np.argsort(-np.abs(v))[:4]))
star = main['star']
say(f'worst design in the box (pga): H/H_0 = {pm(star, se["star"], ".4g")}; '
    f'{int((u_star.abs() > 0.99).sum())}/{D} parameters on a face')
say('  largest components: ' + ', '.join(f'{labels[k]} {u_star[k]:+.2f}' for k in u_star.abs().argsort(descending=True)[:6]))
mc = ratio(U_mc)
q = np.percentile(mc, [5, 50, 95, 99])
say(f'{args.n_mc:,} random designs uniform in the box: H/H_0 median {q[1]:.3f}, 5-95% [{q[0]:.3f}, {q[2]:.3f}], '
    f'99% {q[3]:.3f}, max {mc.max():.3f}; {np.mean(mc > 1):.0%} are worse than nominal')

# =============================================================================
# 4. Walk the directions: surrogate (integrated) and simulator (sampled muons)
# =============================================================================
ratio_w, ratio_se = main['walk'], se['walk']
pred = ratio_w * H0
sim = np.zeros(len(designs))
if args.sim:
    say(f'\nwalking {len(designs)} designs with the simulator over {args.n_muons_sim:,} sampled muons')
    blocks = muon_blocks(shield, args.n_muons_sim, args.block, args.seed, args.temperature)
    sim = simulate(model, shield, blocks, designs, one_group, 1)[:, 0]

say('\n' + '=' * 79 + '\n4. THE WALKS\n' + '=' * 79)
if args.sim:
    say(f'nominal design: surrogate {H0:,.1f} hits per {args.n_ref:.0e} muons, simulator {sim[0]:,.0f} '
        f'over {args.n_muons_sim:,}')
say(f'{"direction":10s} {"t":>7s} {"H/H_0 surrogate":>19s}' + (f' {"H/H_0 simulator":>19s} {"z":>6s}' if args.sim else ''))
for nm, ts, idx in zip(NAMES, TS, IDX):
    for j, (tt, i) in enumerate(zip(ts, idx)):
        line = f'{nm if j == 0 else "":10s} {tt:+7.3f} {pm(ratio_w[i], ratio_se[i], ".4g"):>19s}'
        if args.sim and i:
            sr = sim[i] / sim[0]
            sr_se = sr * np.sqrt(1 / max(sim[i], 1) + 1 / sim[0])   # Poisson, both counts
            line += f' {pm(sr, sr_se, ".4g"):>19s} {(ratio_w[i] - sr) / np.hypot(ratio_se[i], sr_se):+6.1f}'
        say(line)
if args.sim:
    say('  z: surrogate - simulator in combined sigma (the members\' scatter and the simulator\'s Poisson noise)')

# =============================================================================
# 5. Scans: the most sensitive parameters one at a time, and the plane of two directions
# =============================================================================
scan_t = np.linspace(-1, 1, 21)
scan_k = order[:args.n_scan]
scan = np.stack([ratio(np.outer(scan_t, np.eye(D)[k])) for k in scan_k])
a_name, b_name = args.plane.split(',')
va = V[NAMES.index(a_name)]
vb = unit(V[NAMES.index(b_name)] - (V[NAMES.index(b_name)] @ va) * va)     # orthogonalized
plane_s = np.linspace(-1.5, 1.5, args.plane_n)
Sa, Sb = np.meshgrid(plane_s, plane_s)
plane = ratio(Sa.reshape(-1, 1) * va + Sb.reshape(-1, 1) * vb).reshape(args.plane_n, args.plane_n)

np.savez(os.path.join(args.out_dir, 'robustness.npz'),
         labels=np.array(labels), phi_0=shield.phi_0.numpy(), H_0=H0, n_ref=args.n_ref, temperature=args.temperature,
         grid=args.grid, cache_grid=args.cache_grid, n_members=len(mem),
         grad=grad, grad_members=np.array([v['lin'] * v['H0'] for v in mem]), hess=hess, active=act,
         hess_eigvals=mu, hess_eigval_se=mu_se, hess_eigvecs=Qh,
         act_eigvals=nu, act_eigvecs=Qa, act_participation=participation,
         oat_plus=oat_p, oat_plus_se=oat_p_se, oat_minus=oat_m, oat_minus_se=oat_m_se,
         rank_lo=rank_lo, rank_hi=rank_hi, mc=mc, u_star=u_star.numpy(), h_star_out=star, h_star_se=se['star'],
         dir_names=np.array(NAMES), dir_vecs=V, dir_t=TS, dir_idx=IDX, dir_at_tmax=at_1, dir_at_tmax_se=at_1_se,
         designs=designs, phis=model.denormalize_phi(torch.as_tensor(designs, dtype=torch.float32,
                                                                     device=args.device)).cpu().numpy(),
         pred=pred, ratio=ratio_w, ratio_se=ratio_se, sim=sim, n_muons_sim=args.n_muons_sim,
         scan_t=scan_t, scan_params=scan_k, scan=scan, plane_dirs=np.array([a_name, b_name]),
         plane_axes=np.stack([va, vb]), plane_s=plane_s, plane=plane)
say.save(os.path.join(args.out_dir, 'robustness_report.txt'))
print(f'saved {args.out_dir}/robustness.npz and robustness_report.txt')
