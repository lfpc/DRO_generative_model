"""Which architecture and training choices matter: fit on a simulated pool, score on the references.

Training rows come from a pool (lcso/simulate_pools.py, lcso/data.py): budget / n designs, each with n of its muons
(hits and kept misses subsampled alike), the misses optionally thinned further to keep_neg
(case-control: the logit offset -log(keep_neg) makes the model one of p(hit | u, x)). The trunk
sees (px, py, pz, x, y, z, charge), positions in m.

Scores: validation/validate.py's, on the reference designs simulated on stored muons
(validation/references.py: 'T1', the nominal muons, and 'T1.5', the hot check sample). The
gradient and Hessian at u = 0 on the nominal muons are saved for the convergence plots.

Experiments, one setting at a time away from DEFAULTS: budget (every architecture), n (muons
per design), delta and temperature (pools), shape (DeepONet), rank (Taylor), scheme (stream),
keep_neg, and hot (momentum temperature 2 and 3, 1e6 muons x up to 9000 designs). Results:
<out>/results_<worker>.jsonl and <out>/arrays/<key>.npz, resumable.

    python experiments/run_shield.py --list
    python experiments/run_shield.py --worker 0     # as many workers as wanted; each claims the jobs left

Run in the first analysis (43 parameters, muons on axis): the report's App. C.
"""
import hashlib
import json
import os
import time

import numpy as np
import torch

import run_toys

import cli
from lcso.data import train_val
from lcso.models import make, n_params
from lcso.surrogate import load_surrogate
from lcso.training import fit
from paths import EXPERIMENTS, checkpoint
from shield import MuonShield
from validation.validate import REFS, Production, Reference, score

OUT = os.path.join(EXPERIMENTS, 'shield')
DEFAULTS = dict(kind='deeponet', budget=4_000_000_000, n=10_000_000, pool='T1.5_d1', scheme='fixed', buffer=64,
                keep_neg=4e-3, p=16, hidden=64, depth=2, act='silu', lr=1e-3, batch=16384, max_steps=40000, seed=0)
KINDS = ['deeponet', 'taylor', 'taylor:8', 'taylor-sub:4', 'taylor-res', 'quad', 'rsm']
BUDGETS = [300_000_000, 1_000_000_000, 3_000_000_000, 10_000_000_000, 18_000_000_000]
# production surrogates scored alongside: the earlier one, and the final ensemble (lcso/train.py)
PRODUCTION = {'existing': 'lcso_deeponet.pt', 'final': 'lcso.pt'}


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


def run(c, refs, dev):
    t0 = time.time()
    arrays = {}
    if c['kind'] in PRODUCTION:
        model = Production(load_surrogate(checkpoint(PRODUCTION[c['kind']]), MuonShield().phi_0, dev))
        out = score(model, refs, arrays)
        return dict(out, seconds=time.time() - t0), arrays
    train, val = train_val(c, MuonShield().dim, dev)
    torch.manual_seed(c['seed'])
    model = make(c["kind"], MuonShield().dim, 7, p=c['p'], hidden=c['hidden'], depth=c['depth'], act=c['act']).to(dev)
    info = fit(model, train, val, c['scheme'], c['buffer'], c['max_steps'], c['batch'], c['lr'],
               evals_per_epoch=4, seed=c['seed'])
    out = score(model, refs, arrays)
    torch.save(dict(config=c, state_dict=model.state_dict()), os.path.join(OUT, 'arrays', f'{key(c)}.pt'))
    out.update(steps=info['steps'], val_bce=info['val_bce'], designs=train.U.shape[0] + val.U.shape[0],
               rows=len(train.y), hits=int(train.y.sum()), params=n_params(model), seconds=time.time() - t0)
    arrays['curve'] = np.array(info['curve'])
    return out, arrays


def load_results(out=OUT):
    return run_toys.load_results(out)


if __name__ == '__main__':
    parser = cli.parser(__doc__)
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
