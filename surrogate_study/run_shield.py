"""The surrogate study on the muon shield: fit on the simulated pool, score on the reference.

Training rows come from a pool (shield_pool.py): budget / n designs, each with n of its 1e7
muons (hits and kept misses subsampled alike), the misses optionally thinned further to
keep_neg (case-control: the logit offset -log(keep_neg) makes the model one of p(hit | u, x)).
The trunk sees (px, py, pz, charge).

Scores (no exact truth here): on each reference (shield_ref.py; the same muons the simulator
saw), the predicted vs. simulated hits at u = 0 (h0_err), in the half box and in the box
(pred_half, pred_box: median |log pred / sim|); on the T = 1.5 reference also the one-at-a-time
secants (H(+e_i) - H(-e_i)) / 2 and curvatures H(+e_i) + H(-e_i) - 2 H(0) of the 43 parameters
(oat_slope_err, oat_curv_err: relative vector errors, and their cosines). The gradient and
Hessian at u = 0 on the T = 1 muons are saved for the convergence plots.

Experiments, one setting at a time away from DEFAULTS: budget (every architecture), n (muons
per design), delta and temperature (pools), shape (DeepONet), rank (Taylor), scheme (stream),
keep_neg, and hot (momentum temperature 2 and 3 with 1e6 muons x up to 9000 designs). Results: <out>/results_<worker>.jsonl and <out>/arrays/<key>.npz, resumable.

    python run_shield.py --list
    python run_shield.py --worker 0          # as many workers as wanted; each claims the jobs left
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

import run_toys
import shield_ref
from surrogates import derivatives, make, n_params
from training import Rows, fit, split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # after this folder: both have a plots.py
from shield import MuonShield  # noqa: E402
from surrogate import load_surrogate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL, OUT = (os.path.join(ROOT, 'outputs/surrogate_study', d) for d in ('shield_pool', 'shield'))
DEFAULTS = dict(kind='deeponet', budget=4_000_000_000, n=10_000_000, pool='T1.5_d1', scheme='fixed', buffer=64,
                keep_neg=4e-3, p=16, hidden=64, depth=2, act='silu', lr=1e-3, batch=16384, max_steps=40000, seed=0)
KINDS = ['deeponet', 'taylor', 'taylor:8', 'taylor-sub:4', 'taylor-res', 'quad', 'rsm']
BUDGETS = [300_000_000, 1_000_000_000, 3_000_000_000, 10_000_000_000, 18_000_000_000]
REFS = ('T1', 'T1.5')
# production surrogates scored alongside: the earlier one, and the final ensemble (train_surrogate.py)
PRODUCTION = {'existing': 'outputs/lcso_deeponet.pt', 'final': 'outputs/lcso.pt'}


def experiments():
    return {
        'budget': [dict(kind=k, budget=b) for k in KINDS for b in BUDGETS],
        'scheme': [dict(kind=k, budget=b, scheme='stream') for k in ('deeponet', 'taylor') for b in BUDGETS],
        'n': [dict(kind=k, budget=2_000_000_000, n=n) for k in ('deeponet', 'taylor')
              for n in (1_000_000, 3_000_000, 10_000_000)],
        'delta': [dict(kind=k, pool=p, budget=3_000_000_000) for k in ('deeponet', 'taylor', 'taylor:8', 'quad')
                  for p in ('T1.5_d0.5', 'T1.5_d1', 'T1.5_d2')],
        'temperature': [dict(kind=k, pool=p, budget=3_000_000_000) for k in ('deeponet', 'taylor') for p in ('T1_d1', 'T1.5_d1', 'T2_d1')],
        'shape': [dict(hidden=h, depth=d) for h in (32, 64, 128, 256) for d in (2, 3)]
        + [dict(p=p) for p in (4, 8, 16, 32, 64)]
        + [dict(act=a) for a in ('relu', 'silu', 'gelu', 'tanh', 'softplus')],
        'rank': [dict(kind=f'{k}:{r}') for k in ('taylor', 'taylor-sub') for r in (2, 4, 8, 16)],
        'keep_neg': [dict(kind=k, keep_neg=q) for k in ('deeponet', 'taylor') for q in (2.5e-4, 1e-3, 4e-3)],
        # what the above points to: hotter muons, many designs with 1e6 muons each
        'hot': [dict(kind=k, pool=p, n=1_000_000, budget=b) for k in ('deeponet', 'taylor-sub:4', 'taylor:8')
                for p in ('T2_m1e6', 'T3_m1e6') for b in (3_000_000_000, 9_000_000_000)],
    }


def jobs(seeds):
    out = {}
    for over in sum(experiments().values(), []):
        for s in range(seeds):
            c = dict(DEFAULTS, **over, seed=s)
            out[key(c)] = c
    return list(out.values())


def key(c):
    return hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def to_x(mu):
    """(px, py, pz, charge) of raw muons (..., 7)."""
    mu = torch.as_tensor(mu)
    return torch.cat([mu[:, :3], -torch.sign(mu[:, 6:7])], 1).float()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_pools = {}


def load_pool(name):
    """All designs simulated so far in a pool, or in several ('A+B+C', design ids offset):
    dict(U, ids, d, x, y, keep_neg, n_muons) with n_muons the muons simulated per design (M,)."""
    if '+' in name:
        parts = [load_pool(p) for p in name.split('+')]
        if len({P['keep_neg'] for P in parts}) > 1:
            raise ValueError(f'{name}: the pools kept different fractions of the misses')
        off = np.cumsum([0] + [len(P['U']) for P in parts])
        return dict(U=torch.cat([P['U'] for P in parts]), ids=np.concatenate([P['ids'] + o for P, o in zip(parts, off)]),
                    d=torch.cat([P['d'] + int(o) for P, o in zip(parts, off)]), x=torch.cat([P['x'] for P in parts]),
                    y=torch.cat([P['y'] for P in parts]), keep_neg=parts[0]['keep_neg'],
                    n_muons=torch.cat([P['n_muons'] for P in parts]))
    if name not in _pools:
        shards = [torch.load(f, weights_only=False) for f in sorted(glob.glob(os.path.join(POOL, name, 'shard_*.pt')))]
        a = shards[0]['args']
        from shield_pool import pool_designs
        U = pool_designs(a['n_designs'], a['delta'], a['n_nominal'], a['seed'])
        _pools[name] = dict(U=torch.as_tensor(U), ids=np.array(sum((s['ids'] for s in shards), [])),
                            d=torch.cat([t for s in shards for t in s['d']]).long(),
                            x=to_x(torch.cat([t for s in shards for t in s['x']])),
                            y=torch.cat([t for s in shards for t in s['y']]).float(),
                            keep_neg=a['keep_neg'], n_muons=torch.full((len(U),), float(a['n_muons'])))
    return _pools[name]


def training_rows(c):
    """Rows of budget / n designs of the pool, n muons each (n = None: all budget / max designs,
    every muon), misses thinned to keep_neg."""
    P = load_pool(c['pool'])
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


def train_val(c, dev, val_frac=0.1, max_val_rows=2_000_000):
    """(train, val) Rows on dev: split in host memory, so only the halves go to the GPU; the
    validation designs' rows thinned to at most max_val_rows (hits and misses alike)."""
    train, val = split(training_rows(c), val_frac, c['seed'])
    if len(val.y) > max_val_rows:
        keep = torch.randperm(len(val.y), generator=torch.Generator().manual_seed(c['seed']))[:max_val_rows]
        val = Rows(val.U, val.d[keep], val.x[keep], val.y[keep], val.offset)
    return train.to(dev), val.to(dev)


class Reference:
    """A finished shield_ref.py run, its muons as (px, py, pz, charge) in pinned host memory."""

    def __init__(self, name):
        self.names, self.U, self.hits, self.n, path = shield_ref.load(name)
        self.X = torch.cat([to_x(torch.from_numpy(shield_ref.read_muons(path, lo, 50_000_000)))
                            for lo in range(0, self.n, 50_000_000)]).pin_memory()
        self.U = torch.as_tensor(self.U)
        self.group = lambda p: np.array([n.startswith(p) for n in self.names])

    @torch.no_grad()
    def predict(self, model):
        """Predicted hits (sum over the muons) of every reference design."""
        dev = model.bias.device
        chunk = 2 ** 27 // (model.P + 3 * len(self.U))     # ~0.5 GB of features and logits at a time
        b = model.branch(self.U.to(dev))
        out = torch.zeros(len(self.U), dtype=torch.float64, device=dev)
        for c in range(0, self.n, chunk):
            t = model.trunk(self.X[c:c + chunk].to(dev, non_blocking=True))
            out += torch.sigmoid(b @ t.T + model.bias).double().sum(1)
        return out.cpu().numpy()


def ref_scores(R, pred, tag):
    nom = R.group('nominal')
    s0 = R.hits[nom].mean()
    out = {f'h0_err_{tag}': float(pred[nom].mean() / s0 - 1)}
    for grp in ('half', 'box'):
        m = R.group(grp) & (R.hits > 0)
        out[f'pred_{grp}_{tag}'] = float(np.median(np.abs(np.log(pred[m] / R.hits[m]))))
    if R.group('oat').any():
        idx = {n: i for i, n in enumerate(R.names)}
        D = R.U.shape[1]
        for name, f in (('slope', lambda p, m, z: (p - m) / 2), ('curv', lambda p, m, z: p + m - 2 * z)):
            sim = np.array([f(R.hits[idx[f'oat+{i}']], R.hits[idx[f'oat-{i}']], s0) for i in range(D)])
            hat = np.array([f(pred[idx[f'oat+{i}']], pred[idx[f'oat-{i}']], pred[nom].mean()) for i in range(D)])
            out[f'oat_{name}_err'] = float(np.linalg.norm(hat - sim) / np.linalg.norm(sim))
            out[f'oat_{name}_cos'] = float(hat @ sim / (np.linalg.norm(hat) * np.linalg.norm(sim)))
    return out


def score(model, refs, arrays):
    out = {}
    for tag, R in refs.items():
        pred = R.predict(model)
        arrays[f'pred_{tag}'] = pred
        out.update(ref_scores(R, pred, tag))
    X = (refs['T1'] if 'T1' in refs else next(iter(refs.values()))).X          # derivatives on the T = 1 muons
    H0, g, H = derivatives(model, X, chunk=min(2 ** 20, 2 ** 27 // model.P))
    arrays.update(H0=H0 * len(X), g=g.numpy() * len(X), H=H.numpy() * len(X))   # in hits, not rates
    return out


# ---------------------------------------------------------------------------
# A production surrogate (surrogate.LCSONet, e.g. outputs/lcso_deeponet.pt) in the same interface
# ---------------------------------------------------------------------------
class Production(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m, self.bias, self.D, self.P = model, model.bias, model.dim, model.hp['p']

    def branch(self, u):                       # its u is ours: +-1 = +-10% of phi_0
        return self.m.branch_net(u)

    def trunk(self, x):                        # and so are its inputs (px, py, pz, charge)
        return self.m.trunk_net((x - self.m.x_loc) / self.m.x_scale)


def run(c, refs, dev):
    t0 = time.time()
    arrays = {}
    if c['kind'] in PRODUCTION:
        model = Production(load_surrogate(os.path.join(ROOT, PRODUCTION[c['kind']]), MuonShield().phi_0, dev))
        out = score(model, refs, arrays)
        return dict(out, seconds=time.time() - t0), arrays
    train, val = train_val(c, dev)
    torch.manual_seed(c['seed'])
    model = make(c['kind'], 43, 4, p=c['p'], hidden=c['hidden'], depth=c['depth'], act=c['act']).to(dev)
    info = fit(model, train, val, c['scheme'], c['buffer'], c['max_steps'], c['batch'], c['lr'],
               evals_per_epoch=4, seed=c['seed'])
    out = score(model, refs, arrays)
    torch.save(dict(config=c, state_dict=model.state_dict()), os.path.join(OUT, 'arrays', f'{key(c)}.pt'))
    out.update(steps=info['steps'], val_bce=info['val_bce'], designs=train.U.shape[0] + val.U.shape[0],
               rows=len(train.y), hits=int(train.y.sum()), params=n_params(model), seconds=time.time() - t0)
    arrays['curve'] = np.array(info['curve'])
    return out, arrays


def load_results(out=OUT):
    rows = []
    for f in glob.glob(os.path.join(out, 'results_*.jsonl')):
        with open(f) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=type(
        'F', (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter), {}))
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--worker', default='0', help='name of this worker (its results file); workers claim jobs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', default=OUT, help='results; the trained models go to <OUT>/arrays/<key>.pt')
    parser.add_argument('--refs', default=','.join(REFS))
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--only', default=None, help='JSON overrides: run just this config (with --out)')
    parser.add_argument('--skip_pools', default='', help='comma-separated pools not to use yet (still simulating)')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT = args.out                      # where run() saves the models

    todo = jobs(args.seeds) + [dict(kind=k, seed=0) for k in PRODUCTION]
    if args.only:
        todo = [dict(DEFAULTS, **json.loads(args.only))]
    done = {r['key'] for r in load_results(args.out)}
    todo = [c for c in todo if key(c) not in done and c.get('pool') not in args.skip_pools.split(',')]
    todo.sort(key=lambda c: (-c.get('budget', 0), key(c)))
    print(f'{len(todo)} jobs to do', flush=True)
    if args.list or not todo:
        raise SystemExit
    os.makedirs(os.path.join(args.out, 'arrays'), exist_ok=True)
    refs = {t: Reference(t) for t in args.refs.split(',')}
    print('references loaded: ' + ', '.join(f'{t} ({R.n:,} muons, {len(R.U)} designs)' for t, R in refs.items()), flush=True)
    for i, c in enumerate(run_toys.claim(todo, args.out)):
        res, arrays = run(c, refs, args.device)
        k = key(c)
        np.savez(os.path.join(args.out, 'arrays', f'{k}.npz'), **arrays)
        with open(os.path.join(args.out, f'results_{args.worker}.jsonl'), 'a') as f:
            f.write(json.dumps(dict(key=k, config=c, **res)) + '\n')
        print(f'[{i + 1}] ' + ' '.join(f'{a}={c[a]}' for a in ('kind', 'budget', 'n', 'pool', 'scheme', 'hidden', 'depth', 'p', 'act', 'keep_neg', 'seed') if a in c)
              + ': ' + ' '.join(f'{a} {v:.3g}' for a, v in res.items() if isinstance(v, float)), flush=True)
