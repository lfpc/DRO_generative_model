"""The training data of the LCSO surrogate: pools of simulated designs (lcso/simulate_pools.py).

A pool is a set of designs, each simulated once with its own fresh muons. Every hit is kept and a
fraction keep_neg of the misses, so the stored rows are a case-control sample: a model fitted to
them with the logit offset -log(keep_neg) is a model of p(hit | u, x). A pool lives in
paths.POOLS/<name>/shard_<worker>.pt, with its designs (redrawn from its arguments, pool_designs,
if a shard lacks them) and its raw muons.

Pools of fewer parameters load too, the missing ones at nominal (u = 0), shield.FREE being ordered
so: the first analysis' pools (r1_*) had 43 parameters, without the magnet heights (and every muon
at x = y = 0, z = -1 m), the second's (r2_*) 50, without the hadron absorber and the yoke heights.
Both were simulated with M4's exit (dXOut, gapOut, dXYokeOut) at nominal, which shield.full_design
no longer does (M4 is rectangular): they are not the current shield and are not trained on.

    load_pool('A+B', dim)        every row of one or several pools (design ids offset)
    training_rows(config, dim)   the rows a training run uses (a budget of designs and muons, if given)
    train_val(config, dim, dev)  those rows split by design into training and validation
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


def pool_designs(n, delta, n_nominal, seed, dim, radial=True, designs=None):
    """The designs of a pool, from its arguments: u = delta * (scrambled Sobol in [-1, 1]^dim),
    pulled towards u = 0 by a U(0, 1) factor if radial, the first n_nominal at u = 0; or the
    designs of a .npy file."""
    if designs:
        return np.load(designs).astype(np.float32)
    u = qmc.Sobol(d=dim, scramble=True, seed=seed).random(n) * 2 - 1
    if radial:
        u *= np.random.default_rng(seed).random((n, 1))
    u[:n_nominal] = 0
    return (delta * u).astype(np.float32)


def to_x(mu):
    """The trunk's input, (px, py, pz, x, y, z, charge), of raw muons (n, 7), float32."""
    return muon_inputs(torch.as_tensor(mu)).float()


def load_pool(name, dim):
    """All designs simulated so far in a pool, or in several ('A+B+C', design ids offset), with
    dim parameters: dict(U, ids, d, x, y, keep_neg, n_muons) with n_muons the muons simulated per
    design (M,). Pools are cached; clear_cache() frees them."""
    if '+' in name:
        parts = [load_pool(p, dim) for p in name.split('+')]
        if len({P['keep_neg'] for P in parts}) > 1:
            raise ValueError(f'{name}: the pools kept different fractions of the misses')
        if len({P['fields'] for P in parts}) > 1:
            raise ValueError(f'{name}: the pools were simulated in different fields (field map, uniform)')
        off = np.cumsum([0] + [len(P['U']) for P in parts])
        return dict(U=torch.cat([P['U'] for P in parts]), ids=np.concatenate([P['ids'] + o for P, o in zip(parts, off)]),
                    d=torch.cat([P['d'] + int(o) for P, o in zip(parts, off)]), x=torch.cat([P['x'] for P in parts]),
                    y=torch.cat([P['y'] for P in parts]), keep_neg=parts[0]['keep_neg'],
                    n_muons=torch.cat([P['n_muons'] for P in parts]), fields=parts[0]['fields'])
    if name not in _cache:
        shards = [torch.load(f, weights_only=False) for f in sorted(glob.glob(os.path.join(POOLS, name, 'shard_*.pt')))]
        if not shards:
            raise FileNotFoundError(f'no shards of pool {name!r} in {POOLS}')
        a = shards[0]['args']
        U = next((s['U'] for s in shards if 'U' in s), None)
        if U is None:
            U = pool_designs(a['n_designs'], a['delta'], a['n_nominal'], a['seed'], dim=a.get('dim', 43),
                             radial=a.get('radial', True), designs=a.get('designs'))
        U = torch.as_tensor(np.asarray(U), dtype=torch.float32)
        U = torch.cat([U, torch.zeros(len(U), dim - U.shape[1])], 1)      # missing parameters at nominal
        _cache[name] = dict(U=U, ids=np.array(sum((s['ids'] for s in shards), [])),
                            d=torch.cat([t for s in shards for t in s['d']]).long(),
                            x=to_x(torch.cat([t for s in shards for t in s['x']])),
                            y=torch.cat([t for s in shards for t in s['y']]).float(),
                            keep_neg=a['keep_neg'], n_muons=torch.full((len(U),), float(a['n_muons'])),
                            fields=a.get('fields', 'uniform'))
    return _cache[name]


def clear_cache():
    """Forget the loaded pools (they hold ~20-30 GB of host memory for all of them)."""
    _cache.clear()


def training_rows(c, dim):
    """Rows of budget / n designs of the pool, n muons each (n = None: every design and muon),
    the misses thinned further to keep_neg."""
    P = load_pool(c['pool'], dim)
    g = torch.Generator().manual_seed(c['seed'])
    M = len(P['ids']) if c.get('n') is None else int(c['budget'] // c['n'])
    avail = P['ids'][torch.randperm(len(P['ids']), generator=g).numpy()]
    if M > len(avail):
        raise ValueError(f'{c["pool"]} has {len(avail)} designs, {M} needed')
    sel = torch.as_tensor(np.sort(avail[:M]))
    member = torch.zeros(len(P['U']), dtype=torch.bool)
    member[sel] = True
    keep = member[P['d']]
    if c.get('n') is not None:                                                    # n muons per design
        keep &= torch.rand(len(P['y']), generator=g) < c['n'] / P['n_muons'][P['d']]
    keep &= (P['y'] > 0) | (torch.rand(len(P['y']), generator=g) < c['keep_neg'] / P['keep_neg'])   # thin the misses
    new = torch.full((len(P['U']),), -1, dtype=torch.long)
    new[sel] = torch.arange(M)
    return Rows(P['U'][sel], new[P['d'][keep]], P['x'][keep], P['y'][keep], -float(np.log(c['keep_neg'])))


def train_val(c, dim, dev, val_frac=0.1, max_val_rows=2_000_000):
    """(train, val) Rows on dev: split by design in host memory, so only the halves go to the GPU;
    the validation designs' rows thinned to at most max_val_rows (hits and misses alike)."""
    train, val = split(training_rows(c, dim), val_frac, c['seed'])
    if len(val.y) > max_val_rows:
        keep = torch.randperm(len(val.y), generator=torch.Generator().manual_seed(c['seed']))[:max_val_rows]
        val = Rows(val.U, val.d[keep], val.x[keep], val.y[keep], val.offset)
    return train.to(dev), val.to(dev)
