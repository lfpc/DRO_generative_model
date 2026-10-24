"""The surrogate study on the muon shield: fit on a simulated pool, score on the references.

Training rows come from a pool (lcso/simulate_pools.py, lcso/data.py): budget / n designs, each with n of its muons
(hits and kept misses subsampled alike), the misses optionally thinned further to keep_neg
(case-control: the logit offset -log(keep_neg) makes the model one of p(hit | u, x)). The trunk
sees (px, py, pz, charge).

Scores, on the reference designs simulated on stored muons (shield_ref.py), which the surrogate
predicts on the same muons. Two references (REF_LABEL): 'T1', the nominal muons, and 'T1.5', a
hot check sample drawn like the T = 1.5 pools (many more hits per design). On each: predicted vs.
simulated hits at u = 0 (h0_err), in the half box and in the box (pred_half, pred_box: median
|log pred / sim|); on the hot check also the one-at-a-time secants (H(+e_i) - H(-e_i)) / 2 and
curvatures H(+e_i) + H(-e_i) - 2 H(0) of the 43 parameters (oat_slope_err, oat_curv_err:
relative vector errors, and their cosines). The gradient and Hessian at u = 0 on the nominal
muons are saved for the convergence plots.

Experiments, one setting at a time away from DEFAULTS: budget (every architecture), n (muons
per design), delta and temperature (pools), shape (DeepONet), rank (Taylor), scheme (stream),
keep_neg, and hot (momentum temperature 2 and 3, 1e6 muons x up to 9000 designs). Results:
<out>/results_<worker>.jsonl and <out>/arrays/<key>.npz, resumable.

    python run_shield.py --list
    python run_shield.py --worker 0          # as many workers as wanted; each claims the jobs left
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

import run_toys
import shield_ref

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # after this folder: both have a plots.py
from lcso.data import to_x, train_val  # noqa: E402
from lcso.models import derivatives, make, n_params  # noqa: E402
from lcso.surrogate import load_surrogate  # noqa: E402
from lcso.training import fit  # noqa: E402
from shield import MuonShield  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs/surrogate_study/shield')
DEFAULTS = dict(kind='deeponet', budget=4_000_000_000, n=10_000_000, pool='T1.5_d1', scheme='fixed', buffer=64,
                keep_neg=4e-3, p=16, hidden=64, depth=2, act='silu', lr=1e-3, batch=16384, max_steps=40000, seed=0)
KINDS = ['deeponet', 'taylor', 'taylor:8', 'taylor-sub:4', 'taylor-res', 'quad', 'rsm']
BUDGETS = [300_000_000, 1_000_000_000, 3_000_000_000, 10_000_000_000, 18_000_000_000]
REFS = ('T1', 'T1.5')
REF_LABEL = {'T1': 'nominal muons', 'T1.5': 'hot check sample'}
# production surrogates scored alongside: the earlier one, and the final ensemble (lcso/train.py)
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
        U = self.U.to(dev)
        out = torch.zeros(len(U), dtype=torch.float64, device=dev)
        if not model.separable:     # the whole network per design and muon; TF32 here (~1e-3 on the sums)
            chunk = 2 ** 24 // len(U)
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                for c in range(0, self.n, chunk):
                    out += torch.sigmoid(model.logits(U, self.X[c:c + chunk].to(dev, non_blocking=True))).double().sum(1)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf32
            return out.cpu().numpy()
        chunk = 2 ** 27 // (model.P + 3 * len(U))       # ~0.5 GB of features and logits at a time
        b = model.branch(U)
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
    if not model.separable:                  # (final_compare.py takes these on the grid, by autograd)
        return out
    X = (refs['T1'] if 'T1' in refs else next(iter(refs.values()))).X          # derivatives on the nominal muons
    H0, g, H = derivatives(model, X, chunk=min(2 ** 20, 2 ** 27 // model.P))
    arrays.update(H0=H0 * len(X), g=g.numpy() * len(X), H=H.numpy() * len(X))   # in hits, not rates
    return out


# ---------------------------------------------------------------------------
# A production surrogate (surrogate.py, e.g. outputs/lcso.pt) in the study's interface
# ---------------------------------------------------------------------------
def as_study(model):
    """A production surrogate as a study network for score(): a ModelSurrogate's own network
    (its u and inputs are the study's), an LCSONet through Production."""
    return model.net if hasattr(model, 'net') else Production(model)


class Production(torch.nn.Module):
    separable = True

    def __init__(self, model):
        super().__init__()
        self.m, self.bias, self.D, self.P = model, model.bias, model.dim, model.P

    def branch(self, u):                       # its u is ours: +-1 = +-10% of phi_0
        return self.m.branch_net(u)

    def trunk(self, x):                        # and so are its inputs (px, py, pz, charge)
        return self.m.trunk4(x)


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
    return run_toys.load_results(out)


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
