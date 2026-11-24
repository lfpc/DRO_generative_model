"""Trajectories of muons through the shield while the design changes, for animation/render.py.

The design is walked along one parameter (--param, from -range% to +range%) or along one of
robustness.py's directions (--dir, t from -t_max to +t_max), in --frames designs. For each design:
  * the magnets' iron blocks (the simulator's ARB8s, all four quadrants) for drawing;
  * the trajectories of muons of one batch (--n_batch sampled above --traj_pz GeV): the simulator's
    CUDA transport run plane after plane (every --dz m), so its state at each plane is the
    trajectory; the same seeds in every frame, so the paths change with the design, not with the
    random numbers. Drawn: the first --n_traj of the batch, --n_hitters that hit the nominal design,
    and --n_changers that miss it and hit other designs of the walk (those hitting in most frames),
    all picked from these same realizations (a first pass over the frames);
  * the hits on the sensitive plane of --n_hits muons of the physical flux (the same muons in
    every frame): their (x, y) at the last plane and their count, per 1e9 muons.
Writes <out>/<name>.npz. Needs a GPU (MuonsAndMatter's transport, called, not changed).

    python -m animation.trajectories --param M7:dXOut --range 20
    python -m animation.trajectories --dir act1
"""
import os

import numpy as np
import torch

import cli
import cuda_muons_ship as cms
from analysis.core import restrict_shield
from paths import PROJECTS_DIR, RESULTS, RUN
from shield import USE_DILUTED, MuonShield

parser = cli.add_common(cli.parser(__doc__), model=False, fixed=True)
parser.set_defaults(fixed='M0:')
parser.add_argument('--param', default='', help='walk this parameter (label M<i>:<column>)')
parser.add_argument('--range', type=float, default=20, help='--param from -range%% to +range%%')
parser.add_argument('--dir', default='', help="walk this direction of <run>/robustness.npz (dir_names, e.g. 'act1', 'worst')")
parser.add_argument('--t_max', type=float, default=1.0, help='--dir from -t_max to +t_max (L2 norm of u; 1 = one tolerance unit)')
parser.add_argument('--run', default=RUN)
parser.add_argument('--frames', type=int, default=21, help='designs along the walk')
parser.add_argument('--n_batch', type=int, default=2 * 10 ** 7, help='muons transported plane after plane, from which the drawn ones are picked')
parser.add_argument('--n_traj', type=int, default=300, help='drawn muons taken as they come (the first of the batch)')
parser.add_argument('--traj_pz', type=float, default=20, help='... sampled with p_z above this (GeV)')
parser.add_argument('--n_hitters', type=int, default=40, help='drawn muons that hit the nominal design')
parser.add_argument('--n_changers', type=int, default=40, help='drawn muons that miss the nominal design and hit others of the walk')
parser.add_argument('--n_hits', type=int, default=int(1e8), help='muons of the flux for the sensitive plane, per frame')
parser.add_argument('--block', type=int, default=int(1e7), help='muons held in memory')
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
print(f'{name}: {args.frames} designs; field maps first')
shield.prefetch(phis)

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


def trajectories(env, x, zs, seed, keep=()):
    """Transport the muons x plane after plane (zs): (hit flags of all of x (n,), positions of the muons
    keep (len(keep), 1 + len(zs), 3)): the production point, then at each plane; a muon that stops
    (absorbed, turned back) before a plane gets the point where it stopped there and NaN after; NaN
    before it is produced. The random numbers depend on the seed and on the muons transported, so the
    same x and seed give the same paths: what keep records does not change them."""
    keep = torch.as_tensor(np.asarray(keep, dtype=np.int64), device=dev)
    out = torch.full((len(keep), len(zs) + 1, 3), float('nan'), device=dev)
    pos, mom, q = x[:, 3:6].to(dev).clone(), x[:, :3].to(dev).clone(), x[:, 6].to(dev)
    out[:, 0] = pos[keep]                               # the production point (z from -2.14 to 0)
    alive = torch.ones(len(x), dtype=torch.bool, device=dev)
    inside = torch.ones(len(x), dtype=torch.bool, device=dev)
    for k, z in enumerate(zs):
        go = (alive & (pos[:, 2] < z)).nonzero().squeeze(1)
        if len(go):
            p, m, ok = propagate(env, torch.cat([mom[go], pos[go], q[go, None]], 1), z, seed + k)
            pos[go], mom[go] = p, m
            alive[go[~ok]] = False
            moved = torch.zeros(len(x), dtype=torch.bool, device=dev)
            moved[go] = True
            mk = moved[keep]
            out[mk, k + 1] = pos[keep[mk]]
        for kp, pl in zip(iz, (P1, P2)):
            if k + 1 == kp:
                inside &= alive & (pos[:, 2] >= z) & (pos[:, 0].abs() < pl['dx'] / 2) & (pos[:, 1].abs() < pl['dy'] / 2)
    return inside.cpu().numpy(), out.cpu().numpy()


P1, P2 = MuonShield.PLANES


def plane_hits(env, x, seed):
    """(x, y, |p| at production) on the last sensitive plane of the muons x that cross both planes inside them."""
    p0 = x[:, :3].norm(dim=1).to(dev)
    for p, sd in ((P1, seed), (P2, seed + 1)):
        pos, mom, ok = propagate(env, x, p['position'] - p['dz'] / 2, sd)
        ok &= (pos[:, 0].abs() < p['dx'] / 2) & (pos[:, 1].abs() < p['dy'] / 2)
        x, p0 = torch.cat([mom, pos, x[:, 6:7].to(dev)], 1)[ok], p0[ok]
    return torch.cat([x[:, 3:5], p0[:, None]], 1).cpu().numpy()


# --- the muons drawn: one batch of --n_batch muons above --traj_pz, transported plane after plane in
# every frame with the same seeds. Whether a muon hits depends mostly on its random scattering, so the
# muons are picked from these realizations (a first pass, hit flags only), not by a separate search,
# and a second pass records the paths of the picked ones
torch.manual_seed(args.seed)
batch = shield.sample(args.n_batch, pz_band=(args.traj_pz, np.inf))
ZP = [p['position'] - p['dz'] / 2 for p in (P1, P2)]
zs = np.union1d(np.arange(0, ZP[1], args.dz), ZP)      # the transport stops at a plane only at z >= 0
iz = [1 + int(np.searchsorted(zs, z)) for z in ZP]      # their index in the trajectories


def magnet_z(phi):
    """(8, 2): start and end of each magnet along z (m)."""
    z, out = 0.0, []
    for row in shield.full_design(phi).numpy():
        z += row[0] / 100
        out.append((z, z + 2 * row[1] / 100))
        z += 2 * row[1] / 100
    return out


SEED = args.seed * 1000 + 7
nom = int(np.argmin(np.abs(s)))
HF = []
for f in range(args.frames):                            # pass 1: which muons of the batch hit, per frame
    env, _ = environment(phis[f])
    HF.append(trajectories(env, batch, zs, SEED)[0])
    print(f'pass 1, frame {f + 1}/{args.frames} ({s[f]:+.2f}): {HF[-1].sum()} of the batch hit', flush=True)
    del env
    torch.cuda.empty_cache()
HF = np.array(HF)
gain = np.where(~HF[nom] & HF.any(0))[0]
gain = gain[np.argsort(-HF[:, gain].sum(0), kind='stable')][:args.n_changers]   # those hitting in most frames
hold = np.where(HF[nom])[0][:args.n_hitters]
rand = np.setdiff1d(np.arange(args.n_traj), np.r_[hold, gain])
sel = np.r_[rand, hold, gain]
kind = np.r_[np.zeros(len(rand), int), np.ones(len(hold), int), np.full(len(gain), 2)]   # 0 random, 1 hit nominal, 2 start hitting
print(f'drawn: {len(rand)} random, {len(hold)} that hit the nominal design (of {HF[nom].sum()}), '
      f'{len(gain)} that miss it and hit elsewhere on the walk (of {(~HF[nom] & HF.any(0)).sum()})')

T, BLOCKS, HX, NH = [], [], [], []
for f in range(args.frames):                            # pass 2: their paths, and the plane's hits
    env, blocks = environment(phis[f])
    hf, t = trajectories(env, batch, zs, SEED, sel)
    if (hf != HF[f]).any():
        print(f'  warning: {(hf != HF[f]).sum()} hit flags differ from pass 1 (transport not reproducible)')
    T.append(t)
    hx = []
    for lo in range(0, args.n_hits, args.block):
        torch.manual_seed(args.seed + 1 + lo)
        hx.append(plane_hits(env, shield.sample(min(args.block, args.n_hits - lo)).to(dev), args.seed * 1000 + 99 + lo))
    hx = np.concatenate(hx)
    HX.append(hx)
    NH.append(len(hx) * 1e9 / args.n_hits)
    BLOCKS.append(blocks)
    print(f'frame {f + 1}/{args.frames} ({s[f]:+.2f}): {HF[f, sel].sum()} of the drawn muons hit; '
          f'Y = {NH[-1]:.0f} per 1e9 ({len(hx)} hits)', flush=True)
    del env
    torch.cuda.empty_cache()
muons, HF, candidate = batch[torch.as_tensor(sel)], HF[:, sel], kind > 0

os.makedirs(args.out, exist_ok=True)
out = os.path.join(args.out, f'{name}.npz')
np.savez_compressed(out, s=s, xlabel=xlabel, U=U, zs=zs, traj=np.array(T, dtype=np.float32),
                    p=muons[:, :3].norm(dim=1).numpy(), charge=-muons[:, 6].numpy() / 13, candidate=candidate, kind=kind,
                    hit=HF,
                    blocks=np.array(BLOCKS, dtype=np.float32), mag_z=np.array([magnet_z(p) for p in phis]), hits_xy=np.array(HX, dtype=object), Y=np.array(NH),
                    n_hits=args.n_hits, planes=np.array([[p['position'], p['dx'], p['dy']] for p in (P1, P2)]))
print(f'saved {out}')
