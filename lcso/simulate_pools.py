"""Simulate a pool of designs once; the surrogate is trained on pools (lcso/data.py).

Designs: u = delta * (scrambled Sobol in [-1, 1]^D) * U(0, 1) (the radial pull towards the
nominal design; --no_radial: uniform in the region), the first --n_nominal of them at u = 0;
u = +-1 is a parameter at +-10%. --designs takes them from a .npy file (n, D) instead.
Each design gets its own --n_muons fresh muons from --muons, the momentum at --temperature;
with --pz_band, --n_muons muons with p_z in a range (the model is of p(hit | u, x) for each
muon, so choosing the muons by x biases nothing: it puts the simulation where the training data
is thin; the flow draws the production point only for those).
--sparse K moves only 1 to K parameters per design (data.pool_designs).
Every hit is kept and a miss with probability --keep_neg, so the stored rows are a case-control
sample: a model fitted to them with the logit offset -log(keep_neg) is a model of p(hit | u, x).

Workers split the designs (i % n_workers == worker) and write shard_<worker>.pt, re-saved
every --save_every seconds and resumed from it. The simulator can crash on extreme designs (a
CUDA illegal memory access, which kills the process): every design is logged in
shard_<worker>.attempts before it is simulated, and on resuming, a design that has already been
attempted --max_attempts times is skipped and listed in the shard's 'failed'. Rerun the worker
until it prints 'done' (run_all.sh does).

    python -m lcso.simulate_pools --name T1.5_d1 --n_designs 1024 --worker 0 --n_workers 2
"""
import json
import os
import time

import torch

import cli
from paths import FIELD_MAPS, POOLS
from shield import MuonShield

from .data import pool_designs

if __name__ == '__main__':
    parser = cli.add_common(cli.parser(__doc__), model=False)
    parser.add_argument('--name', required=True, help='pool name (sub-directory of paths.POOLS)')
    parser.add_argument('--delta', type=float, default=1.0, help='design region half-width, in u')
    parser.add_argument('--temperature', type=float, default=1.5, help='momentum temperature of the muons')
    parser.add_argument('--n_designs', type=int, default=2048)
    parser.add_argument('--n_nominal', type=int, default=16, help='designs at u = 0')
    parser.add_argument('--no_radial', dest='radial', action='store_false', help='designs uniform in the region')
    parser.add_argument('--sparse', type=int, default=0, help='move only 1 to K parameters per design (0: all)')
    parser.add_argument('--pz_band', default=None, help="'lo,hi' (GeV): simulate only the muons with p_z in it")
    parser.add_argument('--designs', default=None, help='.npy of designs (n, D); overrides the draw')
    parser.add_argument('--n_muons', type=int, default=int(1e7), help='per design (in the --pz_band, if given)')
    parser.add_argument('--keep_neg', type=float, default=4e-3, help='fraction of misses kept')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--block', type=int, default=int(1e7), help='muons simulated at once (GPU memory)')
    parser.add_argument('--save_every', type=float, default=600, help='seconds between saves of the shard')
    parser.add_argument('--max_attempts', type=int, default=2, help='tries per design before it is skipped')
    parser.add_argument('--map_workers', type=int, default=4, help='field maps computed at once (8 CPU cores each)')
    args = parser.parse_args()

    shield = MuonShield(args.muons, fields=args.fields, map_workers=args.map_workers,
                        map_dir=os.path.join(FIELD_MAPS, f'pool_{args.name}_{args.worker}'))
    shield.muons.load()
    out_dir = os.path.join(POOLS, args.name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'shard_{args.worker}.pt')
    U = pool_designs(args.n_designs, args.delta, args.n_nominal, args.seed, shield.dim, args.radial, args.designs,
                     args.sparse)
    band = [float(v) for v in args.pz_band.split(',')] if args.pz_band else None
    args.n_designs, args.dim = len(U), U.shape[1]
    mine = [i for i in range(args.n_designs) if i % args.n_workers == args.worker]
    shard = dict(args=vars(args), ids=[], n_hits=[], n_muons=[], d=[], x=[], y=[], U=U)
    if os.path.exists(path):
        shard = torch.load(path, weights_only=False)
        shard['args'] = vars(args)
        print(f'resuming {path}: {len(shard["ids"])} designs done')
    attempts_path = path.replace('.pt', '.attempts')
    attempts = json.load(open(attempts_path)) if os.path.exists(attempts_path) else {}
    done = set(shard['ids'])
    shard['failed'] = sorted(int(i) for i, n in attempts.items() if n >= args.max_attempts and int(i) not in done)
    if shard['failed']:
        print(f'skipping {len(shard["failed"])} designs that crashed the simulator {args.max_attempts} times: {shard["failed"]}')
    todo = [i for i in mine if i not in done and i not in set(shard['failed'])]

    def save():
        torch.save(shard, path + '.tmp')
        os.replace(path + '.tmp', path)

    def save_attempts():
        with open(attempts_path, 'w') as f:
            json.dump(attempts, f)

    t0 = t_save = time.time()
    for k, i in enumerate(todo):
        shield.prefetch([shield.phi(U[j]) for j in todo[k:k + 2 * args.map_workers]])   # the next field maps
        attempts[str(i)] = attempts.get(str(i), 0) + 1
        save_attempts()
        torch.manual_seed(1_000_003 * args.seed + i)
        mus, hits = [], []
        for lo in range(0, args.n_muons, args.block):
            mu = shield.sample(min(args.block, args.n_muons - lo), args.temperature, device=args.device, pz_band=band)
            mus.append(mu)
            hits.append(shield(shield.phi(U[i]), mu).to(mu.device))
        mu, hit = torch.cat(mus), torch.cat(hits)
        del mus, hits
        shield.drop_map(shield.phi(U[i]))
        keep = hit | (torch.rand(len(mu), device=mu.device) < args.keep_neg)
        del attempts[str(i)]                          # it ran: only designs that crash are counted
        shard['ids'].append(i)
        shard['n_hits'].append(int(hit.sum()))
        shard['n_muons'].append(len(mu))
        shard['d'].append(torch.full((int(keep.sum()),), i, dtype=torch.int32))
        shard['x'].append(mu[keep].float().cpu())
        shard['y'].append(hit[keep].to(torch.int8).cpu())
        print(f'design {i} ({k + 1}/{len(todo)}): {int(hit.sum())} hits, {int(keep.sum())} rows, '
              f'{(time.time() - t0) / (k + 1):.2f} s/design', flush=True)
        if time.time() - t_save > args.save_every or k + 1 == len(todo):
            for key in ('d', 'x', 'y'):                    # keep the shard compact on disk
                shard[key] = [torch.cat(shard[key])] if shard[key] else []
            save()
            save_attempts()
            t_save = time.time()
    shard['U'] = U
    save()
    print(f'done: {path}')
