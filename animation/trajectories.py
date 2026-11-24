"""Trajectories of muons through the shield while the design changes, and where in (p_z, p_t) the
hits come from, for animation/render.py.

The design is walked along one parameter (--param, from -range% to +range%) or along one of
robustness.py's directions (--dir, t from -t_max to +t_max), in --frames designs. Every muon is
transported by the simulator's CUDA transport plane after plane (every --dz m), so its state at
each plane is its trajectory. Every muon draws its random numbers from its own stream (the seed of
the plane and its index), the same in every frame, so a path changes only where the design does,
not with the random numbers. For each design:
  * the magnets' iron blocks (the simulator's ARB8s, all four quadrants) for drawing;
  * context: the trajectories of --n_ctx muons sampled per p_z bin (--bins), as many per bin as the
    bin's share of the nominal design's hits (from the surrogate), at least --min_per_bin; the same
    muons in every frame;
  * the hits: --n_hits muons of the physical flux (the same muons in every frame), their hits on
    the last sensitive plane (x, y, p_z, p_t), Y per 1e9 muons, and the trajectories of a fixed
    fraction --show of the hits. Which hits are shown is decided per muon once (a uniform number
    per muon, the same in every frame), so a muon that keeps hitting stays drawn, and the number of
    drawn hits follows Y. The hits and their trajectories are the same simulation: the hit flags
    come from a first pass of each block, the shown paths from a second pass with the same seeds;
  * the probability of a hit per (p_z, p_t) cell (--pz_edges, --pt_edges) and the expected hits
    per cell, from the surrogate (its p averaged over the flux in the cell, on the quadrature grid
    --cache_grid); the simulation's hits per cell too, though at ~1e-6 hits per muon 1e8 muons fill
    few cells (render.py draws the simulated hits as points instead). The surrogate was trained
    with the old hit cut, which kept ~0.75 of the crossings (report, Setup).
Writes <out>/<name>.npz. Needs a GPU (MuonsAndMatter's transport, called, not changed).

    python -m animation.trajectories --param M7:dXOut --range 20
    python -m animation.trajectories --dir act1
"""
import os
from types import SimpleNamespace

import numpy as np
import torch

import cli
import cuda_muons_ship as cms
from analysis.core import Restricted, cached_hits, restrict_shield
from lcso.surrogate import load_surrogate
from paths import PROJECTS_DIR, RESULTS, RUN
from shield import FREE, NOMINAL, USE_DILUTED, MuonShield

parser = cli.add_common(cli.parser(__doc__), model=True, fixed=True)
parser.set_defaults(fixed='M0:')
parser.add_argument('--param', default='', help='walk this parameter (label M<i>:<column>)')
parser.add_argument('--range', type=float, default=20, help='--param from -range%% to +range%%')
parser.add_argument('--dir', default='', help="walk this direction of <run>/robustness.npz (dir_names, e.g. 'act1', 'worst')")
parser.add_argument('--t_max', type=float, default=1.0, help='--dir from -t_max to +t_max (L2 norm of u; 1 = one tolerance unit)')
parser.add_argument('--run', default=RUN)
parser.add_argument('--frames', type=int, default=21, help='designs along the walk')
parser.add_argument('--n_ctx', type=int, default=300, help='context muons whose trajectories are drawn in every frame')
parser.add_argument('--bins', type=float, nargs='+', default=[10, 20, 30, 45, 75, 125, 200, 300, np.inf],
                    help='p_z bin edges (GeV) of the context muons (and of the colours)')
parser.add_argument('--min_per_bin', type=int, default=15, help='context muons per bin at least')
parser.add_argument('--n_hits', type=int, default=int(1e8), help='muons of the flux for the hits, per frame')
parser.add_argument('--block', type=int, default=int(1e7), help='muons held in memory')
parser.add_argument('--show', type=float, default=0.2, help='fraction of the hits whose trajectories are drawn')
parser.add_argument('--pz_edges', type=float, nargs=3, default=[15, 370, 40], help='map: log p_z from, to, cells')
parser.add_argument('--pt_edges', type=float, nargs=3, default=[0, 5, 30], help='map: p_t from, to, cells')
parser.add_argument('--cache_grid', default='401x32', help="the surrogate's map: muon quadrature grid (muons.py)")
parser.add_argument('--dz', type=float, default=0.25, help='trajectory sampling step along z (m)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out', default=os.path.join(RESULTS, 'animations'))
args = parser.parse_args()
dev = args.device
STEP = 0.02                                             # the transport's step (alias histograms)
HIST = os.path.join(PROJECTS_DIR, 'MuonsAndMatter/cuda_muons/data')

shield = MuonShield(args.muons, fields=args.fields)
fixed = [p for p in args.fixed.split(',') if p]
if fixed:
    restrict_shield(shield, fixed)
shield.muons.load()
D = shield.dim

# --- the walk
if bool(args.param) == bool(args.dir):
    raise SystemExit('give one of --param, --dir')
if args.param:
    s = np.linspace(-args.range, args.range, args.frames)              # % of the nominal value
    U = np.zeros((args.frames, D))
    U[:, shield.labels.index(args.param)] = s / (100 * shield.DELTA)
    name, xlabel = f'param_{args.param.replace(":", "_")}', f'change of {args.param} [%]'
else:
    r = np.load(os.path.join(args.run, 'robustness.npz'))
    v = r['dir_vecs'][[str(n) for n in r['dir_names']].index(args.dir)]
    s = np.linspace(-args.t_max, args.t_max, args.frames)
    U = s[:, None] * v[None] / np.linalg.norm(v)
    name, xlabel = f'dir_{args.dir}', f'step along {args.dir} (tolerance units)'
phi_0 = shield.phi_0.double().numpy()
phis = torch.tensor(phi_0 * (1 + shield.DELTA * U), dtype=torch.float32)
nom = int(np.argmin(np.abs(s)))
print(f'{name}: {args.frames} designs; field maps first')
shield.prefetch(phis)

# --- the (p_z, p_t) cells and the surrogate's probability of a hit in each, per design
PZ = np.geomspace(args.pz_edges[0], args.pz_edges[1], int(args.pz_edges[2]) + 1)
PT = np.linspace(args.pt_edges[0], args.pt_edges[1], int(args.pt_edges[2]) + 1)
NZ, NT = len(PZ) - 1, len(PT) - 1
EDGES = np.asarray(args.bins, dtype=float)


def cell(pz, pt):
    """Cell index iz * NT + it of (p_z, p_t), NZ * NT outside the map (torch)."""
    iz = torch.bucketize(pz.contiguous(), torch.as_tensor(PZ, dtype=pz.dtype, device=pz.device)) - 1
    it = torch.bucketize(pt.contiguous(), torch.as_tensor(PT, dtype=pt.dtype, device=pt.device)) - 1
    ok = (iz >= 0) & (iz < NZ) & (it >= 0) & (it < NT)
    return torch.where(ok, iz * NT + it, torch.full_like(iz, NZ * NT))


idx = torch.tensor(FREE)
model = load_surrogate(args.model, NOMINAL[idx[:, 0], idx[:, 1]].clone(), dev)
if len(shield.keep) < len(FREE):
    model = Restricted(model, shield.keep)
t, w, g, gb = [], [], [], []
with torch.no_grad():
    for x, wx in shield.muons.points(args.cache_grid, 1e9):
        x = x.to(dev)
        t.append(model.features(x).half())
        w.append(wx.to(dev))
        g.append(cell(x[:, 2], x[:, :2].norm(dim=1)))
        gb.append(torch.bucketize(x[:, 2].contiguous(), torch.as_tensor(EDGES[1:-1], dtype=x.dtype, device=dev)))
cache = SimpleNamespace(t=torch.cat(t), w=torch.cat(w), g=torch.cat(g))
G = NZ * NT + 1
sur_hits = cached_hits(model, cache, G, U)[:, :-1].reshape(-1, NZ, NT)            # expected hits per 1e9
sur_n = torch.zeros(G, dtype=torch.float64, device=dev).index_add_(0, cache.g, cache.w).cpu().numpy()[:-1].reshape(NZ, NT)
sur_p = sur_hits / np.where(sur_n > 0, sur_n, np.nan)
cache.g = torch.cat(gb)                                  # the context bins' shares of the nominal hits
share = cached_hits(model, cache, len(EDGES) - 1, U[nom:nom + 1])[0]
share /= share.sum()
print(f'surrogate: Y = {sur_hits[nom].sum() + 0:.0f} per 1e9 at nominal inside the map ({NZ}x{NT} cells)')
del cache, t, w, g, gb
torch.cuda.empty_cache()

# --- the transport, plane after plane
hist = {k: cms.load_histogram(os.path.join(HIST, f'alias_histograms_G4_{m}.pkl'))[0]
        for k, m in (('iron', 'Fe'), ('concrete', 'CONCRETE'))}


def environment(phi):
    """(transport environment, iron blocks (n, 8, 3) of all quadrants) of design phi."""
    det = cms.get_design_from_params(params=shield.full_design(phi).numpy(), fSC_mag=False, field_mode='read_file',
                                     sensitive_film_params=None, field_map_file=shield.field_map(phi),
                                     add_cavern=True, add_target=False, sensitive_decay_vessel=False,
                                     NI_from_B=True, use_diluted=USE_DILUTED, SND=False, cores_field=1)
    env = cms.set_environment(cms.get_corners_from_detector(det, True), cms.get_cavern_from_detector(det), hist,
                              cms.get_magnetic_field_from_detector(det, True), device=dev)
    return env, cms.get_corners_from_detector(det, False).numpy()


def propagate(env, x, z_to, seed):
    """Muons x (n, 7: px, py, pz, x, y, z, charge as pdg) to the plane z_to: (positions, momenta, alive)."""
    q = -x[:, 6] / 13
    pos, mom = cms.propagate_muons_with_cuda(x[:, 3:6], x[:, :3], q, env, z_to, 100000, STEP, True, seed, dev,
                                             to_cpu=False)
    return pos, mom, pos[:, 2] >= z_to


P1, P2 = MuonShield.PLANES
ZP = [p['position'] - p['dz'] / 2 for p in (P1, P2)]
zs = np.union1d(np.arange(0, ZP[1], args.dz), ZP)      # the transport stops at a plane only at z >= 0
iz = [1 + int(np.searchsorted(zs, z)) for z in ZP]      # their index in the trajectories


def trajectories(env, x, seed, keep=()):
    """Transport the muons x plane after plane (zs): (hit flags (n,), (x, y) on the last plane (n, 2),
    positions of the muons keep (len(keep), 1 + len(zs), 3)): the production point, then at each
    plane; a muon that stops (absorbed, turned back) before a plane gets the point where it stopped
    there and NaN after. Muon i draws its random numbers from curand(seed + plane, i) whatever the
    design and the other muons, so a muon's path changes between designs only where the iron does
    (common random numbers); what keep records does not change them."""
    keep = torch.as_tensor(np.asarray(keep, dtype=np.int64), device=dev)
    out = torch.full((len(keep), len(zs) + 1, 3), float('nan'), device=dev)
    pos, mom, q = x[:, 3:6].to(dev).clone(), x[:, :3].to(dev).clone(), x[:, 6].to(dev)
    out[:, 0] = pos[keep]
    alive = torch.ones(len(x), dtype=torch.bool, device=dev)
    inside = torch.ones(len(x), dtype=torch.bool, device=dev)
    xy = torch.full((len(x), 2), float('nan'), device=dev)
    for k, z in enumerate(zs):
        gm = alive & (pos[:, 2] < z)
        go = gm.nonzero().squeeze(1)
        if len(go):
            # every muon keeps its index in the kernel, whose random numbers are curand(seed, index):
            # the same stream for a muon in every design (the others get p = 0 and stop at once)
            p, m, ok = propagate(env, torch.cat([torch.where(gm[:, None], mom, 0.0), pos, q[:, None]], 1), z, seed + k)
            p, m, ok = p[go], m[go], ok[go]
            pos[go], mom[go] = p, m
            alive[go[~ok]] = False
            moved = torch.zeros(len(x), dtype=torch.bool, device=dev)
            moved[go] = True
            mk = moved[keep]
            out[mk, k + 1] = pos[keep[mk]]
        for kp, pl in zip(iz, (P1, P2)):
            if k + 1 == kp:
                inside &= alive & (pos[:, 2] >= z) & (pos[:, 0].abs() < pl['dx'] / 2) & (pos[:, 1].abs() < pl['dy'] / 2)
                xy = pos[:, :2].clone()
    return inside.cpu().numpy(), xy[inside].cpu().numpy(), out.cpu().numpy()


def magnet_z(phi):
    """(8, 2): start and end of each magnet along z (m)."""
    z, out = 0.0, []
    for row in shield.full_design(phi).numpy():
        z += row[0] / 100
        out.append((z, z + 2 * row[1] / 100))
        z += 2 * row[1] / 100
    return out


def block(lo):
    """The muons lo:lo + block of the hits' sample (the same in every frame) and their draw for --show."""
    torch.manual_seed(args.seed + 1 + lo)
    x = shield.sample(min(args.block, args.n_hits - lo))
    r = torch.rand(len(x), generator=torch.Generator().manual_seed(args.seed + 2 + lo)).numpy()
    return x, r


# --- the context muons: per p_z bin, in proportion to the bin's share of the nominal hits
n_bin = np.maximum(args.min_per_bin, np.round(args.n_ctx * share)).astype(int)
torch.manual_seed(args.seed)
ctx = torch.cat([shield.sample(int(n), pz_band=(lo, hi)) for n, lo, hi in zip(n_bin, EDGES[:-1], EDGES[1:])])
for k in range(len(n_bin)):
    print(f'  p_z {EDGES[k]:g}-{EDGES[k + 1]:g} GeV: {100 * share[k]:.1f}% of the nominal hits, {n_bin[k]} context muons')

# the simulation's muons per cell (the same muons in every frame)
sim_n = np.zeros(NZ * NT + 1)
for lo in range(0, args.n_hits, args.block):
    x, _ = block(lo)
    sim_n += np.bincount(cell(x[:, 2], x[:, :2].norm(dim=1)).numpy(), minlength=NZ * NT + 1)
sim_n = sim_n[:-1].reshape(NZ, NT)

SEED = args.seed * 1000 + 7
CT, CH, HT, HP, HX, SH, BLOCKS, NH = [], [], [], [], [], [], [], []
for f in range(args.frames):
    env, blocks = environment(phis[f])
    hf, _, t = trajectories(env, ctx, SEED, np.arange(len(ctx)))
    CT.append(t)
    CH.append(hf)
    ht, hp, hx, sh = [], [], [], np.zeros(NZ * NT + 1)
    for lo in range(0, args.n_hits, args.block):
        x, r = block(lo)
        sd = args.seed * 1000 + 99 + lo
        hit, xy, _ = trajectories(env, x, sd)                       # pass 1: which hit
        mom = x[hit, :3]
        hx.append(np.c_[xy, mom[:, 2].numpy(), mom[:, :2].norm(dim=1).numpy()])
        sh += np.bincount(cell(mom[:, 2], mom[:, :2].norm(dim=1)).numpy(), minlength=NZ * NT + 1)
        sel = np.where(hit & (r < args.show))[0]
        if len(sel):                                                # pass 2: the paths of the shown ones
            hit2, _, t = trajectories(env, x, sd, sel)
            if not hit2[sel].all():
                print(f'  warning: {(~hit2[sel]).sum()} shown hits do not repeat (transport not reproducible)')
            ht.append(t)
            hp.append(x[sel, 2].numpy())
    hx = np.concatenate(hx)
    HT.append(np.concatenate(ht) if ht else np.zeros((0, len(zs) + 1, 3), np.float32))
    HP.append(np.concatenate(hp) if hp else np.zeros(0, np.float32))
    HX.append(hx)
    SH.append(sh[:-1].reshape(NZ, NT))
    NH.append(len(hx) * 1e9 / args.n_hits)
    BLOCKS.append(blocks)
    print(f'frame {f + 1}/{args.frames} ({s[f]:+.2f}): Y = {NH[-1]:.0f} per 1e9 ({len(hx)} hits, {len(HP[-1])} drawn); '
          f'surrogate {sur_hits[f].sum():.0f} in the map', flush=True)
    del env
    torch.cuda.empty_cache()

os.makedirs(args.out, exist_ok=True)
out = os.path.join(args.out, f'{name}.npz')
def obj(a):
    """Ragged per-frame arrays as one object array."""
    o = np.empty(len(a), dtype=object)
    o[:] = [None] * len(a)
    for i, v in enumerate(a):
        o[i] = v
    return o


np.savez_compressed(out, s=s, xlabel=xlabel, U=U, zs=zs,
                    ctx_traj=np.array(CT, dtype=np.float32), ctx_pz=ctx[:, 2].numpy(), ctx_hit=np.array(CH),
                    hit_traj=obj(HT), hit_pz=obj(HP), show=args.show, edges=EDGES, share=share,
                    blocks=np.array(BLOCKS, dtype=np.float32), mag_z=np.array([magnet_z(p) for p in phis]),
                    hits_xy=obj(HX), Y=np.array(NH), n_hits=args.n_hits,
                    planes=np.array([[p['position'], p['dx'], p['dy']] for p in (P1, P2)]),
                    map_pz=PZ, map_pt=PT, sur_p=sur_p, sur_n=sur_n, sur_hits=sur_hits, sim_hits=np.array(SH), sim_n=sim_n)
print(f'saved {out}')
