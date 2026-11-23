"""The training data of the LCSO surrogate: pools of simulated designs (lcso/simulate_pools.py).

A pool is a set of designs, each simulated once with its own fresh muons. Every hit is kept and a
fraction keep_neg of the misses, so the stored rows are a case-control sample: a model fitted to
them with the logit offset -log(keep_neg) is a model of p(hit | u, x); pools that kept different
fractions train together, each row with its own pool's offset. A pool lives in
paths.POOLS/<name>/shard_<worker>.pt, with its designs (redrawn from its arguments, pool_designs,
if a shard lacks them) and its raw muons.

    load_pool('A+B')             every row of one or several pools (design ids offset)
    training_rows(pool)          the rows a training run uses
    train_val(pool, seed, dev)   those rows split by design into training and validation
"""
import glob
import os

import numpy as np
import torch
from scipy.stats import qmc

from paths import POOLS

from .surrogate import muon_inputs
from .training import Rows, split

_cache = {}


def pool_designs(n, delta, n_nominal, seed, dim, radial=True, designs=None, sparse=0, faces=0.0, fixed=()):
    """The designs of a pool, from its arguments: u = delta * (scrambled Sobol in [-1, 1]^dim),
    pulled towards u = 0 by a U(0, 1) factor if radial, the first n_nominal at u = 0; or the
    designs of a .npy file. sparse = K > 0: each design moves only 1 to K parameters (uniform in
    [-1, 1], the others at 0), the one-at-a-time effects and their pairs; faces = f: a fraction f
    of the moved parameters at their limit (+-1), where the analyses look. fixed: indices of
    parameters held at nominal (the draw is unchanged otherwise)."""
    if designs:
        return np.load(designs).astype(np.float32)
    u = qmc.Sobol(d=dim, scramble=True, seed=seed).random(n) * 2 - 1
    rng = np.random.default_rng(seed)
    if sparse:
        k = rng.integers(1, sparse + 1, n)
        rank = np.argsort(rng.random((n, dim)), axis=1)
        u *= rank < k[:, None]
        if faces:
            u = np.where(rng.random(u.shape) < faces, np.sign(u), u)
    elif radial:
        u *= rng.random((n, 1))
    u[:n_nominal] = 0
    u[:, list(fixed)] = 0
    return (delta * u).astype(np.float32)


def to_x(mu):
    """The trunk's input, (px, py, pz, x, y, z, charge), of raw muons (n, 7), float32."""
    return muon_inputs(torch.as_tensor(mu)).float()


def load_pool(name):
    """All designs simulated so far in a pool, or in several ('A+B+C', design ids offset):
    dict(U, ids, d, x, y, keep_neg) with keep_neg the fraction of misses kept, per design (M,).
    Pools are cached; clear_cache() frees them."""
    if '+' in name:
        parts = [load_pool(p) for p in name.split('+')]
        off = np.cumsum([0] + [len(P['U']) for P in parts])
        return dict(U=torch.cat([P['U'] for P in parts]), ids=np.concatenate([P['ids'] + o for P, o in zip(parts, off)]),
                    d=torch.cat([P['d'] + int(o) for P, o in zip(parts, off)]), x=torch.cat([P['x'] for P in parts]),
                    y=torch.cat([P['y'] for P in parts]), keep_neg=torch.cat([P['keep_neg'] for P in parts]))
    if name not in _cache:
        shards = [torch.load(f, weights_only=False) for f in sorted(glob.glob(os.path.join(POOLS, name, 'shard_*.pt')))]
        if not shards:
            raise FileNotFoundError(f'no shards of pool {name!r} in {POOLS}')
        a = shards[0]['args']
        U = next((s['U'] for s in shards if 'U' in s), None)
        if U is None:
            U = pool_designs(a['n_designs'], a['delta'], a['n_nominal'], a['seed'], a.get('dim', 65), a.get('radial', True),
                             a.get('designs'), a.get('sparse', 0), a.get('faces', 0.0), a.get('fixed_idx', ()))
        U = torch.as_tensor(np.asarray(U), dtype=torch.float32)
        _cache[name] = dict(U=U, ids=np.array(sum((s['ids'] for s in shards), [])),
                            d=torch.cat([t for s in shards for t in s['d']]).long(),
                            x=to_x(torch.cat([t for s in shards for t in s['x']])),
                            y=torch.cat([t for s in shards for t in s['y']]).float(),
                            keep_neg=torch.full((len(U),), float(a['keep_neg'])))
    return _cache[name]


def clear_cache():
    """Forget the loaded pools (they hold ~20-30 GB of host memory for all of them)."""
    _cache.clear()


def training_rows(pool):
    """Every row of the pool(s), the simulated designs renumbered 0..M-1; the logit offset per row
    if the pools kept different fractions of the misses."""
    P = load_pool(pool)
    sel = torch.as_tensor(np.sort(P['ids']))
    new = torch.full((len(P['U']),), -1, dtype=torch.long)
    new[sel] = torch.arange(len(sel))
    keep = torch.zeros(len(P['U']), dtype=torch.bool)
    keep[sel] = True
    keep = keep[P['d']]
    q = P['keep_neg'][sel]
    offset = -float(np.log(q[0])) if bool((q == q[0]).all()) else -torch.log(P['keep_neg'][P['d'][keep]])
    return Rows(P['U'][sel], new[P['d'][keep]], P['x'][keep], P['y'][keep], offset)


def train_val(pool, seed, dev, val_frac=0.1, max_val_rows=2_000_000):
    """(train, val) Rows on dev: split by design in host memory, so only the halves go to the GPU;
    the validation designs' rows thinned to at most max_val_rows (hits and misses alike)."""
    train, val = split(training_rows(pool), val_frac, seed)
    if len(val.y) > max_val_rows:
        keep = torch.randperm(len(val.y), generator=torch.Generator().manual_seed(seed))[:max_val_rows]
        val = Rows(val.U, val.d[keep], val.x[keep], val.y[keep], val.off(keep))
    return train.to(dev), val.to(dev)
