"""Simulate a pool of muon-shield designs once; the surrogate is trained on pools (lcso/data.py).

Designs: u = delta * (scrambled Sobol in [-1, 1]^43) * U(0, 1) (the radial pull towards the
nominal design; --no_radial: uniform in the region), the first --n_nominal of them at u = 0;
u = +-1 is a parameter at +-10%. --designs takes them from a .npy file (n, 43) instead.
Each design gets its own --n_muons fresh flow muons at --temperature. Every hit is kept and
a miss with probability --keep_neg, so the stored rows are a case-control sample: a model
fitted to them with the logit offset -log(keep_neg) is a model of p(hit | phi, x).

Workers split the designs (i % n_workers == worker) and write shard_<worker>.pt, re-saved
every --save_every designs and resumed from it. The simulator can crash on extreme designs (a
CUDA illegal memory access, which kills the process): every design is logged in
shard_<worker>.attempts before it is simulated, and on resuming, a design that has already been
attempted --max_attempts times is skipped and listed in the shard's 'failed'. Rerun the worker
until it prints 'done'.

    python -m lcso.simulate_pools --name T1.5_d1 --n_designs 2048 --worker 0 --n_workers 3
"""
import argparse
import json
import os
import time

import torch

from shield import MuonShield

from .data import POOL_DIR, pool_designs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=type(
        'F', (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter), {}))
    parser.add_argument('--name', required=True, help='pool name (sub-directory)')
    parser.add_argument('--delta', type=float, default=1.0, help='design region half-width, in u')
    parser.add_argument('--temperature', type=float, default=1.5)
    parser.add_argument('--n_designs', type=int, default=2048)
    parser.add_argument('--n_nominal', type=int, default=16, help='designs at u = 0')
    parser.add_argument('--no_radial', dest='radial', action='store_false', help='designs uniform in the region')
    parser.add_argument('--designs', default=None, help='.npy of designs (n, 43), relative to the repo; overrides the draw')
    parser.add_argument('--n_muons', type=int, default=int(1e7), help='per design')
    parser.add_argument('--keep_neg', type=float, default=4e-3, help='fraction of misses kept')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=16)
    parser.add_argument('--max_attempts', type=int, default=2, help='tries per design before it is skipped')
    args = parser.parse_args()

    out_dir = os.path.join(POOL_DIR, args.name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'shard_{args.worker}.pt')
    U = pool_designs(args.n_designs, args.delta, args.n_nominal, args.seed, radial=args.radial, designs=args.designs)
    args.n_designs = len(U)
    mine = [i for i in range(args.n_designs) if i % args.n_workers == args.worker]
    shard = dict(args=vars(args), ids=[], n_hits=[], n_muons=[], d=[], x=[], y=[])
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

    shield = MuonShield()
    shield._load_flow()
    phi_0 = shield.phi_0.numpy()
    t0 = time.time()
    for k, i in enumerate(todo):
        attempts[str(i)] = attempts.get(str(i), 0) + 1
        with open(attempts_path, 'w') as fa:
            json.dump(attempts, fa)
        torch.manual_seed(1_000_003 * args.seed + i)
        mu = shield.sample(args.n_muons, args.temperature)
        hit = shield(phi_0 * (1 + 0.1 * U[i]), mu).cpu()
        keep = hit | (torch.rand(len(mu)) < args.keep_neg)
        attempts[str(i)] = 0                          # it ran: only designs that crash count up
        with open(attempts_path, 'w') as fa:
            json.dump(attempts, fa)
        shard['ids'].append(i)
        shard['n_hits'].append(int(hit.sum()))
        shard['n_muons'].append(len(mu))
        shard['d'].append(torch.full((int(keep.sum()),), i, dtype=torch.int32))
        shard['x'].append(mu[keep].float())
        shard['y'].append(hit[keep].to(torch.int8))
        print(f'design {i} ({k + 1}/{len(todo)}): {int(hit.sum())} hits, {int(keep.sum())} rows, '
              f'{(time.time() - t0) / (k + 1):.1f} s/design', flush=True)
        if (k + 1) % args.save_every == 0 or k + 1 == len(todo):
            for key in ('d', 'x', 'y'):                    # keep the shard compact on disk
                shard[key] = [torch.cat(shard[key])] if shard[key] else []
            save()
    shard['U'] = U
    save()
    print(f'done: {path}')
