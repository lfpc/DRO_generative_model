"""Reference designs: a fixed set of designs simulated on one stored muon sample.

The surrogate is scored on these same muons (validation/validate.py), so its errors are not mixed
with the sampling noise of the muons; only the simulator's own randomness remains.

Designs (in u, u_i = +-1 at +-10%): u = 0 repeated --n_nominal times (separate transport
seeds: the simulator noise), --n_random uniform in the half box and as many in the box, and,
with --oat, every parameter alone at +-1. Two references are used: 'T1', the nominal muons, and
'T1.5', a hot check sample drawn like the T = 1.5 pools (many more hits per design, so the
one-at-a-time secants and curvatures are measured on it).

When the shield gains parameters (shield.FREE), --stage extend keeps the stored muons and the
designs simulated so far (the new parameters at nominal) and appends --n_random new designs per
region in all the parameters and, if the reference has them, the new parameters one at a time;
--stage simulate then simulates only those (with the same --n_workers as before).

    python -m validation.references --name T1 --stage sample --temperature 1 --n_muons 600000000
    python -m validation.references --name T1 --stage simulate --worker 0 --n_workers 2
    python -m validation.references --name T1 --stage extend --n_random 8
    python -m validation.references --name T1 --stage simulate --designs extra.npz --tag worst   # other designs
"""
import glob
import os
import time

import numpy as np
import torch

import cli
from paths import REFERENCES
from shield import MuonShield


def ref_designs(n_nominal, n_random, oat, D, seed=5):
    rng = np.random.default_rng(seed)
    names = ['nominal'] * n_nominal + [f'half{i}' for i in range(n_random)] + [f'box{i}' for i in range(n_random)]
    U = [np.zeros((n_nominal, D)), rng.uniform(-0.5, 0.5, (n_random, D)), rng.uniform(-1, 1, (n_random, D))]
    if oat:
        names += [f'oat{s}{i}' for i in range(D) for s in '+-']
        U.append(np.concatenate([[s * np.eye(D)[i] for s in (1, -1)] for i in range(D)]))
    return names, np.concatenate(U).astype(np.float32)


def extend_designs(names, U, D, n_random, seed=6):
    """names and U (m, D) of a reference's designs U (m, d < D) with the parameters d..D-1 added:
    the old designs with them at nominal, then n_random new designs per region in all D, then, if
    the old designs include the one-at-a-time ones, the new parameters alone at +-1."""
    rng, d = np.random.default_rng(seed), U.shape[1]
    nh, nb = (sum(n.startswith(g) for n in names) for g in ('half', 'box'))
    names = list(names) + [f'half{nh + i}' for i in range(n_random)] + [f'box{nb + i}' for i in range(n_random)]
    new = [np.pad(U, ((0, 0), (0, D - d))), rng.uniform(-0.5, 0.5, (n_random, D)), rng.uniform(-1, 1, (n_random, D))]
    if any(n.startswith('oat') for n in names):
        names += [f'oat{s}{i}' for i in range(d, D) for s in '+-']
        new.append(np.concatenate([[s * np.eye(D)[i] for s in (1, -1)] for i in range(d, D)]))
    return names, np.concatenate(new).astype(np.float32)


def read_muons(path, lo, n):
    """Rows lo .. lo + n of a stored (N, 7) muons.npy, by plain reads (a memmap on NFS crawls)."""
    with open(path, 'rb') as f:
        np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f)
        f.seek(f.tell() + 28 * lo)
        return np.fromfile(f, dtype='<f4', count=7 * n).reshape(-1, 7)


def load(name):
    """(names, U (m, D), hits (m,), n_muons, path of muons.npy) of a finished reference."""
    d = os.path.join(REFERENCES, name)
    shards = [torch.load(f, weights_only=False) for f in sorted(glob.glob(os.path.join(d, 'shard_[0-9]*.pt')))]
    meta = torch.load(os.path.join(d, 'meta.pt'), weights_only=False)
    hits = np.full(len(meta['names']), np.nan)
    for s in shards:
        for i, h in s['hits'].items():
            hits[i] = h.sum()
    return meta['names'], meta['U'], hits, meta['n_muons'], os.path.join(d, 'muons.npy')


def hit_index(name):
    """{design index: indices of the muons that hit} of a finished reference."""
    out = {}
    for f in sorted(glob.glob(os.path.join(REFERENCES, name, 'shard_[0-9]*.pt'))):
        out.update(torch.load(f, weights_only=False)['hit_idx'])
    return out


if __name__ == '__main__':
    parser = cli.add_common(cli.parser(__doc__), model=False)
    parser.add_argument('--name', required=True, help='the reference: paths.REFERENCES/<name>/')
    parser.add_argument('--stage', choices=('sample', 'extend', 'simulate'), required=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--n_muons', type=int, default=int(1e9))
    parser.add_argument('--n_nominal', type=int, default=4)
    parser.add_argument('--n_random', type=int, default=16, help='per region (half box, box); with extend, the new ones')
    parser.add_argument('--oat', action='store_true', help='also every parameter alone at +-10%%')
    parser.add_argument('--block', type=int, default=25_000_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--designs', default=None, help='simulate: .npz with names and U, instead of the reference designs')
    parser.add_argument('--tag', default=None, help='with --designs: written to shard_<tag>_<worker>.pt')
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    args = parser.parse_args()
    d = os.path.join(REFERENCES, args.name)
    os.makedirs(d, exist_ok=True)
    shield = MuonShield(args.muons)
    shield.muons.load()

    if args.stage == 'sample':
        names, U = ref_designs(args.n_nominal, args.n_random, args.oat, D=shield.dim)
        t0 = time.time()
        with open(os.path.join(d, 'muons.npy.tmp'), 'wb') as f:     # sequential writes: a memmap on NFS crawls
            np.lib.format.write_array_header_1_0(f, dict(descr='<f4', fortran_order=False, shape=(args.n_muons, 7)))
            for lo in range(0, args.n_muons, args.block):
                torch.manual_seed(10 ** 9 * args.seed + lo)
                f.write(shield.sample(min(args.block, args.n_muons - lo), args.temperature).numpy().astype('<f4').tobytes())
                print(f'{min(lo + args.block, args.n_muons):,} muons ({time.time() - t0:.0f} s)', flush=True)
        os.replace(os.path.join(d, 'muons.npy.tmp'), os.path.join(d, 'muons.npy'))
        torch.save(dict(names=names, U=U, n_muons=args.n_muons, temperature=args.temperature,
                        block=args.block), os.path.join(d, 'meta.pt'))
        print(f'saved {d}/muons.npy and meta.pt ({len(names)} designs)')
        raise SystemExit

    meta = torch.load(os.path.join(d, 'meta.pt'), weights_only=False)
    if args.stage == 'extend':
        d0 = meta['U'].shape[1]
        if d0 >= shield.dim:
            print(f'{d}: the designs already have {d0} parameters')
            raise SystemExit
        meta['names'], meta['U'] = extend_designs(meta['names'], meta['U'], shield.dim, args.n_random)
        meta.setdefault('extended', []).append(dict(from_dim=d0, to_dim=shield.dim, n_random=args.n_random))
        torch.save(meta, os.path.join(d, 'meta.pt.tmp'))
        os.replace(os.path.join(d, 'meta.pt.tmp'), os.path.join(d, 'meta.pt'))
        print(f'{d}: {d0} -> {shield.dim} parameters, {len(meta["names"])} designs')
        raise SystemExit
    if meta['U'].shape[1] != shield.dim:
        raise SystemExit(f'{d}: designs of {meta["U"].shape[1]} parameters, the shield has {shield.dim} (--stage extend)')
    path = os.path.join(d, f'shard_{args.worker}.pt')
    if args.designs:
        extra = np.load(args.designs)
        meta.update(names=[str(n) for n in extra['names']], U=extra['U'].astype(np.float32))
        path = os.path.join(d, f'shard_{args.tag}_{args.worker}.pt')
    shard = torch.load(path, weights_only=False) if os.path.exists(path) else dict(hits={}, hit_idx={})
    B = meta['block']
    muons = read_muons(os.path.join(d, 'muons.npy'), 0, meta['n_muons'])      # into RAM, once
    mine = [i for i in range(len(meta['names'])) if i % args.n_workers == args.worker and i not in shard['hits']]
    print(f'{len(mine)} designs to simulate on {meta["n_muons"]:,} muons')
    for k, i in enumerate(mine):
        t0 = time.time()
        phi = shield.phi(meta['U'][i])
        per_block, idx = [], []
        for lo in range(0, meta['n_muons'], B):
            h = shield(phi, torch.from_numpy(muons[lo:lo + B])).cpu()
            per_block.append(int(h.sum()))
            idx.append(lo + torch.nonzero(h)[:, 0].numpy())
        shard['hits'][i] = np.array(per_block)
        shard['hit_idx'][i] = np.concatenate(idx)
        torch.save(shard, path + '.tmp')
        os.replace(path + '.tmp', path)
        print(f'{meta["names"][i]} ({k + 1}/{len(mine)}): {sum(per_block)} hits ({time.time() - t0:.0f} s)', flush=True)
    print('done')
