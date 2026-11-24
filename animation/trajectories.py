"""Trajectories of muons through the shield while the design changes, and where in (p_z, p_t) the
hits come from, for animation/render.py.

The design is walked along one parameter (--param, from -range% to +range%) or along one of
robustness.py's directions (--dir, t from -t_max to +t_max), in --frames designs. Every muon is
transported by the simulator's CUDA transport plane after plane (every --dz m through the magnets,
every --dz_far m after them), so its state at each plane is its trajectory. Every muon draws its
random numbers from its own stream (the seed of the plane and its index), the same in every frame,
so a path changes only where the design does, not with the random numbers. The part of the shield
that the walk does not change (up to the end of the magnet before the first one that changes:
M0 for the directions, M6 for an M7 parameter) is transported once and shared by every frame.
For each design:
  * the magnets' iron blocks (the simulator's ARB8s, all four quadrants) for drawing;
  * context: the trajectories of --n_ctx muons sampled per p_z bin (--bins), as many per bin as the
    bin's share of the nominal design's hits (from the surrogate), at least --min_per_bin; the same
    muons in every frame;
  * the hits: --n_hits muons of the physical flux (the same muons in every frame), their hits on
    the last sensitive plane (x, y, p_z, p_t), Y per 1e9 muons;
  * tracked muons: about --n_track of the muons that hit in at least one design of the walk (a
    fixed draw per muon), whose trajectories are recorded in every frame, hitting or not, so a
    beam can be seen moving onto the plane. The hits and the paths are the same simulation: a
    first pass over all frames gives the hit flags, a second the tracked paths, with the same seeds;
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
parser.add_argument('--n_ctx', type=int, default=350, help='context muons whose trajectories are drawn in every frame')
parser.add_argument('--bins', type=float, nargs='+', default=[10, 20, 30, 45, 75, 125, 200, 300, np.inf],
                    help='p_z bin edges (GeV) of the context muons (and of the colours)')
parser.add_argument('--min_per_bin', type=int, default=15, help='context muons per bin at least')
parser.add_argument('--n_hits', type=int, default=int(1e8), help='muons of the flux for the hits, per frame')
parser.add_argument('--block', type=int, default=int(1e7), help='muons held in memory')
parser.add_argument('--n_track', type=int, default=150, help='about this many muons that hit somewhere along the walk, drawn in every frame')
parser.add_argument('--pz_edges', type=float, nargs=3, default=[15, 370, 40], help='map: log p_z from, to, cells')
parser.add_argument('--pt_edges', type=float, nargs=3, default=[0, 5, 30], help='map: p_t from, to, cells')
parser.add_argument('--cache_grid', default='401x32', help="the surrogate's map: muon quadrature grid (muons.py)")
parser.add_argument('--dz', type=float, default=0.25, help='trajectory sampling step along z through the magnets (m)')
parser.add_argument('--dz_far', type=float, default=2.0, help='the step after the last magnet (straight paths) (m)')
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


def magnet_z(phi):
    """(8, 2): start and end of each magnet along z (m)."""
    z, out = 0.0, []
    for row in shield.full_design(phi).numpy():
        z += row[0] / 100
        out.append((z, z + 2 * row[1] / 100))
        z += 2 * row[1] / 100
    return out


P1, P2 = MuonShield.PLANES
ZP = [p['position'] - p['dz'] / 2 for p in (P1, P2)]
MAG_Z = np.array([magnet_z(p) for p in phis])
z_fine = MAG_Z[:, -1, 1].max() + 1                      # past the last magnet: straight lines, coarser steps
zs = np.union1d(np.r_[np.arange(0, z_fine, args.dz), np.arange(z_fine, ZP[1], args.dz_far)], ZP)
iz = [1 + int(np.searchsorted(zs, z)) for z in ZP]      # their index in the trajectories
# the part of the shield that no design of the walk changes: up to the end of the magnet before the
# first one that changes (its gap included in the change); there every frame has the same paths
D = np.stack([shield.full_design(p).numpy() for p in phis])
changed = np.abs(D - D[nom]).max(axis=(0, 2)) > 1e-6
m0 = int(np.argmax(changed)) if changed.any() else 8
z_shared = MAG_Z[nom, m0 - 1, 1] if m0 > 0 else 0.0
K0 = int(np.searchsorted(zs, z_shared, side='right'))   # steps k < K0 end at zs[k] <= z_shared
print(f'{len(zs)} steps; M{m0} is the first magnet that changes: steps up to z = {zs[K0 - 1] if K0 else 0:.2f} m '
      f'transported once, in the nominal design')


def start(x):
    """Transport state of the muons x at production."""
    n = len(x)
    return dict(pos=x[:, 3:6].to(dev).clone(), mom=x[:, :3].to(dev).clone(), q=x[:, 6].to(dev),
                alive=torch.ones(n, dtype=torch.bool, device=dev), inside=torch.ones(n, dtype=torch.bool, device=dev),
                xy=torch.full((n, 2), float('nan'), device=dev))


def advance(env, st, seed, k0, k1, move=None, keep=None, out=None):
    """Transport the state st through the steps k0..k1-1 (to zs[k]), in place. move: the muons
    moved (bool, default all); keep, out: record the positions of the muons keep in out[:, k + 1].
    Every muon keeps its index in the kernel, whose random numbers are curand(seed + k, index) (the
    others get p = 0 and stop at once): a muon's path does not depend on the design elsewhere or
    on which other muons are moved (common random numbers). A muon that stops (absorbed, turned
    back) keeps the point where it stopped and NaN after; inside: it crossed both planes inside."""
    pos, mom, alive = st['pos'], st['mom'], st['alive']
    for k in range(k0, k1):
        z = zs[k]
        gm = alive & (pos[:, 2] < z)
        if move is not None:
            gm &= move
        go = gm.nonzero().squeeze(1)
        if len(go):
            p, m, ok = propagate(env, torch.cat([torch.where(gm[:, None], mom, 0.0), pos, st['q'][:, None]], 1), z, seed + k)
            pos[go], mom[go] = p[go], m[go]
            alive[go[~ok[go]]] = False
            if keep is not None:
                mk = gm[keep]
                out[mk, k + 1] = pos[keep[mk]]
        for kp, pl in zip(iz, (P1, P2)):
            if k + 1 == kp:
                st['inside'] &= alive & (pos[:, 2] >= z) & (pos[:, 0].abs() < pl['dx'] / 2) & (pos[:, 1].abs() < pl['dy'] / 2)
                st['xy'] = pos[:, :2].clone()


def clone(st, device=dev):
    return {k: v.to(device).clone() for k, v in st.items()}


def block(lo):
    """The muons lo:lo + block of the hits' sample (the same in every frame) and their draw for --n_track."""
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

# --- the shared part, once: the context muons' paths and every block's state at z_shared
SEED = args.seed * 1000 + 7
env0, _ = environment(phis[nom])
ALL = torch.arange(len(ctx), device=dev)
ctx_out = torch.full((len(ctx), len(zs) + 1, 3), float('nan'), device=dev)
ctx_out[:, 0] = ctx[:, 3:6].to(dev)
ctx_st = start(ctx)
advance(env0, ctx_st, SEED, 0, K0, keep=ALL, out=ctx_out)
sim_n = np.zeros(NZ * NT + 1)                            # the simulation's muons per cell
BLK = []
for lo in range(0, args.n_hits, args.block):
    x, r = block(lo)
    sim_n += np.bincount(cell(x[:, 2], x[:, :2].norm(dim=1)).numpy(), minlength=NZ * NT + 1)
    st = start(x)
    advance(env0, st, args.seed * 1000 + 99 + lo, 0, K0)
    BLK.append((x, r, clone(st, 'cpu')))
    del st
    print(f'  block {lo // args.block + 1}: {BLK[-1][2]["alive"].float().mean():.3f} of the muons alive at z = {z_shared:.2f} m', flush=True)
sim_n = sim_n[:-1].reshape(NZ, NT)
torch.cuda.empty_cache()

# --- pass 1, every frame: the context paths and which muons hit
CT, CH, HX, SH, BLOCKS, NH, HIT = [], [], [], [], [], [], []
for f in range(args.frames):
    env, blocks = environment(phis[f])
    st, t = clone(ctx_st), ctx_out.clone()
    advance(env, st, SEED, K0, len(zs), keep=ALL, out=t)
    CT.append(t.cpu().numpy())
    CH.append(st['inside'].cpu().numpy())
    hx, sh, hits = [], np.zeros(NZ * NT + 1), []
    for b, (x, r, st0) in enumerate(BLK):
        st = clone(st0)
        advance(env, st, args.seed * 1000 + 99 + b * args.block, K0, len(zs))
        hit = st['inside'].cpu().numpy()
        xy = st['xy'][st['inside']].cpu().numpy()
        del st
        mom = x[hit, :3]
        hx.append(np.c_[xy, mom[:, 2].numpy(), mom[:, :2].norm(dim=1).numpy()])
        sh += np.bincount(cell(mom[:, 2], mom[:, :2].norm(dim=1)).numpy(), minlength=NZ * NT + 1)
        hits.append(np.where(hit)[0])
    hx = np.concatenate(hx)
    HX.append(hx)
    HIT.append(hits)
    SH.append(sh[:-1].reshape(NZ, NT))
    NH.append(len(hx) * 1e9 / args.n_hits)
    BLOCKS.append(blocks)
    print(f'frame {f + 1}/{args.frames} ({s[f]:+.2f}): Y = {NH[-1]:.0f} per 1e9 ({len(hx)} hits); '
          f'surrogate {sur_hits[f].sum():.0f} in the map', flush=True)
    del env
    torch.cuda.empty_cache()

# --- the tracked muons: a fixed draw (r) among those that hit in some frame
UNION = [np.unique(np.concatenate([HIT[f][b] for f in range(args.frames)])) for b in range(len(BLK))]
n_union = sum(len(u) for u in UNION)
q = min(1.0, args.n_track / max(n_union, 1))
TRACK = [u[BLK[b][1][u] < q] for b, u in enumerate(UNION)]
print(f'{n_union} muons hit in some frame; {sum(len(t) for t in TRACK)} tracked', flush=True)
track_pz = np.concatenate([BLK[b][0][t, 2].numpy() for b, t in enumerate(TRACK)])

# --- pass 2, every frame: the tracked muons' paths (the same seeds, so the same hits)
HT, TH = [], []
for f in range(args.frames):
    env, _ = environment(phis[f])
    ht, th = [], []
    for b, (x, r, st0) in enumerate(BLK):
        sel = TRACK[b]
        if not len(sel):
            continue
        sd = args.seed * 1000 + 99 + b * args.block
        keep = torch.as_tensor(sel, device=dev)
        move = torch.zeros(len(x), dtype=torch.bool, device=dev)
        move[keep] = True
        out = torch.full((len(sel), len(zs) + 1, 3), float('nan'), device=dev)
        out[:, 0] = x[sel, 3:6].to(dev)
        st = start(x)
        advance(env0, st, sd, 0, K0, move, keep, out)
        advance(env, st, sd, K0, len(zs), move, keep, out)
        h2 = st['inside'][keep].cpu().numpy()
        h1 = np.isin(sel, HIT[f][b])
        if (h1 != h2).any():
            print(f'  warning: frame {f + 1}: {(h1 != h2).sum()} tracked muons hit differently in pass 2')
        ht.append(out.cpu().numpy())
        th.append(h2)
        del st, out
    HT.append(np.concatenate(ht) if ht else np.zeros((0, len(zs) + 1, 3), np.float32))
    TH.append(np.concatenate(th) if th else np.zeros(0, bool))
    print(f'tracked, frame {f + 1}/{args.frames}: {TH[-1].sum()} of {len(TH[-1])} hit', flush=True)
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
                    track_traj=np.array(HT, dtype=np.float32), track_pz=track_pz, track_hit=np.array(TH), n_union=n_union, edges=EDGES, share=share,
                    blocks=np.array(BLOCKS, dtype=np.float32), mag_z=MAG_Z,
                    hits_xy=obj(HX), Y=np.array(NH), n_hits=args.n_hits,
                    planes=np.array([[p['position'], p['dx'], p['dy']] for p in (P1, P2)]),
                    map_pz=PZ, map_pt=PT, sur_p=sur_p, sur_n=sur_n, sur_hits=sur_hits, sim_hits=np.array(SH), sim_n=sim_n)
print(f'saved {out}')
