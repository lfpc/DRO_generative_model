"""The surrogate study on the toy problems, where the truth is exact.

Each job simulates a training set (budget samples: budget / n designs x n samples each, the
designs in the region delta around u = 0), fits one surrogate and scores it at u = 0 against
the truth (metrics.py). Experiments vary one setting at a time away from DEFAULTS:

  budget    every architecture, budgets 1e5 .. 1e7 (convergence)
  scheme    fixed vs. stream (rolling buffer) training, over the budgets
  n         samples per design at a fixed budget (many designs with few samples, or the reverse)
  delta     size of the design region the training designs are drawn from; radial pull on/off
  shape     DeepONet width x depth, latent size p, branch activation
  rank      rank of the Taylor curvature (taylor:r) and of the design manifold (taylor-sub:r)
  gate      Taylor + gated residual: the gate (mean(u^2))^2 or mean(u^4) (-res4), and the
            near-weighted loss (near: the Taylor part learns from rows near u = 0 only)
  converge  the DeepONet against the Taylor surrogates at larger budgets (3e7, 1e8; the step cap
            raised to 1e5 so that the fits are not cut short): how many simulated samples each
            needs for a given gradient and Hessian error

Jobs shared between experiments run once. Results: <out>/results_<worker>.jsonl, resumable.

    python experiments/run_toys.py --list                                   # count the jobs
    python experiments/run_toys.py --worker gpu0 --device cuda          # as many workers as wanted, each
    python experiments/run_toys.py --worker cpu0 --device cpu           # claims the jobs left (<out>/claims)
    python experiments/run_toys.py --quick --problems synthetic              # smoke test
"""
import glob
import hashlib
import json
import os
import time

import numpy as np
import torch

import metrics
import toys

import cli
from lcso.models import derivatives, features, make, n_params, predict
from lcso.training import Rows, draw_designs, fit, split
from paths import EXPERIMENTS

OUT = os.path.join(EXPERIMENTS, 'toys')
DEFAULTS = dict(kind='deeponet', budget=1_000_000, n=1000, delta=1.0, radial=True, scheme='fixed',
                buffer=64, p=16, hidden=64, depth=2, act='silu', lr=1e-3, batch=4096,
                max_steps=40000, seed=0)
KINDS = ['deeponet', 'taylor', 'taylor:8', 'taylor-sub:4', 'taylor-res', 'quad', 'rsm', 'fd']
BUDGETS = [100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]


def experiments():
    """{experiment: [config overrides]}."""
    E = {'budget': [dict(kind=k, budget=b) for k in KINDS for b in BUDGETS],
         'scheme': [dict(kind=k, budget=b, scheme='stream') for k in ('deeponet', 'taylor') for b in BUDGETS],
         'n': [dict(kind=k, n=n) for k in ('deeponet', 'taylor', 'taylor:8') for n in (10, 100, 1000, 10_000, 100_000)],
         'delta': [dict(kind=k, delta=d) for k in ('deeponet', 'taylor', 'taylor:8', 'quad')
                   for d in (0.25, 0.5, 1.0, 2.0, 4.0)]
         + [dict(kind=k, radial=False) for k in ('deeponet', 'taylor')],
         'shape': [dict(hidden=h, depth=d) for h in (16, 32, 64, 128, 256) for d in (1, 2, 3, 4)]
         + [dict(p=p) for p in (2, 4, 8, 16, 32, 64)]
         + [dict(act=a) for a in ('relu', 'silu', 'gelu', 'tanh', 'softplus')],
         'rank': [dict(kind=f'{k}:{r}') for k in ('taylor', 'taylor-sub') for r in (1, 2, 4, 8, 16)]
         + [dict(kind='taylor-sub')],
         'gate': [dict(kind=k, budget=b, **({'near': nr} if nr else {}))
                  for k in ('taylor-res:8', 'taylor-res4:8', 'taylor-sub-res4:4') for nr in (0, 0.3) for b in BUDGETS],
         'converge': [dict(kind=k, budget=b, max_steps=100_000, **({'near': nr} if nr else {}))
                      for k, nr in (('deeponet', 0), ('taylor-res4:8', 0), ('taylor-res4:8', 0.3), ('taylor-sub-res4:4', 0))
                      for b in (30_000_000, 100_000_000)]}
    return E


def jobs(problems, seeds, only=None):
    out = {}
    for over in sum((v for k, v in experiments().items() if only is None or k in only), []):
        for prob in problems:
            for s in range(seeds):
                c = dict(DEFAULTS, **over, problem=prob, seed=s)
                out[json.dumps(c, sort_keys=True)] = c
    return list(out.values())


# ---------------------------------------------------------------------------
# Truth and simulation
# ---------------------------------------------------------------------------
def simulate(prob, U, n, seed):
    """n fresh samples per design (M, D) and their Bernoulli hits, as Rows."""
    dev = prob.dev
    g = torch.Generator(dev).manual_seed(seed)
    M = U.shape[0]
    x = prob.sample(M * n, g).view(M, n, -1)
    U = U.to(dev)
    y = torch.empty(M, n, device=dev)
    per = max(1, 2 ** 21 // n)
    f = torch.vmap(lambda u, xs: prob.logit(u[None], xs)[0])
    for c in range(0, M, per):
        p = torch.sigmoid(f(U[c:c + per], x[c:c + per]))
        y[c:c + per] = (torch.rand(p.shape, device=dev, generator=g) < p).float()
    d = torch.arange(M, device=dev).repeat_interleave(n)
    return Rows(U, d, x.view(M * n, -1), y.view(-1))


def truth(prob, n_true=1_000_000, radius=1.0, n_test=128):
    dev = prob.dev
    X = prob.sample(n_true, torch.Generator(dev).manual_seed(7))
    H0, g, H = toys.true_derivatives(prob, X)
    rng = np.random.default_rng(11)
    U_in = torch.as_tensor(rng.uniform(-0.5, 0.5, (n_test, prob.D)), dtype=torch.float32, device=dev)
    U_box = torch.as_tensor(rng.uniform(-1, 1, (n_test, prob.D)), dtype=torch.float32, device=dev)
    Xc = X[:200_000]
    F = lambda U: prob.prob(U, Xc).mean(1)                                      # noqa: E731
    # the true worst case (256 random starts; the surrogates get 64: there are local maxima)
    u_star = metrics.ball_max(F, metrics.inits(g, H, radius, n_random=256).to(dev), radius, steps=2000)
    T = dict(X=X, Xc=Xc, H0=H0, g=g, H=H, U_in=U_in, U_box=U_box, radius=radius,
             H_in=toys.true_H(prob, X, U_in).cpu().numpy(), H_box=toys.true_H(prob, X, U_box).cpu().numpy())
    T['rise_star'] = float(toys.true_H(prob, X, u_star[None])[0]) - H0
    return T


def score(prob, model, T):
    H0, g, H = derivatives(model, T['X'])
    out = metrics.derivative_scores(g, H, T['g'], T['H'])
    out.update(metrics.pred_scores(H0, T['H0'], predict(model, T['X'], T['U_in']).cpu().numpy(), T['H_in'],
                                   predict(model, T['X'], T['U_box']).cpu().numpy(), T['H_box']))
    t = features(model, T['Xc'])
    F = lambda U: torch.sigmoid(model.branch(U) @ t.T + model.bias).mean(1)     # noqa: E731
    u = metrics.ball_max(F, metrics.inits(g, H, T['radius'], n_random=64).to(t.device), T['radius'], steps=2000)
    out['worst'] = (float(toys.true_H(prob, T['X'], u[None])[0]) - T['H0']) / T['rise_star']
    return out


def finite_differences(prob, c, T):
    """Baseline without a surrogate: central differences of the simulated hit rate along each
    coordinate (+-h, h = delta / 2), budget / 2D samples per design, the same samples for +-."""
    D, m, h = prob.D, max(1, c['budget'] // (2 * prob.D)), c['delta'] / 2
    g = torch.Generator(prob.dev).manual_seed(1000 + c['seed'])
    est = torch.zeros(D, dtype=torch.float64)
    for i in range(D):
        x = prob.sample(m, g)
        e = torch.zeros(1, D, device=prob.dev)
        e[0, i] = h
        hits = [(torch.rand(m, device=prob.dev, generator=g) < prob.prob(s * e, x)[0]).double().mean() for s in (1, -1)]
        est[i] = float(hits[0] - hits[1]) / (2 * h)
    return dict(grad_err=float((est - T['g']).norm() / T['g'].norm()),
                grad_cos=float(est @ T['g'] / (est.norm() * T['g'].norm()).clamp_min(1e-300)))


def run(c, prob, T):
    t0 = time.time()
    if c['kind'] == 'fd':
        return dict(finite_differences(prob, c, T), seconds=time.time() - t0)
    M = max(4, c['budget'] // c['n'])
    rows = simulate(prob, draw_designs(M, prob.D, c['delta'], c['radial'], seed=c['seed']), c['n'], seed=c['seed'])
    train, val = split(rows, 0.1, c['seed'])
    torch.manual_seed(c['seed'])
    model = make(c['kind'], prob.D, prob.dx, p=c['p'], hidden=c['hidden'], depth=c['depth'], act=c['act']).to(prob.dev)
    model.near = c.get('near', 0.0)          # (not in DEFAULTS, so earlier results keep their keys)
    info = fit(model, train, val, c['scheme'], c['buffer'], c['max_steps'], c['batch'], c['lr'], seed=c['seed'])
    out = score(prob, model, T)
    out.update(steps=info['steps'], val_bce=info['val_bce'], designs=M, hits=int(rows.y.sum()),
               params=n_params(model), seconds=time.time() - t0)
    return out


def claim(todo, out):
    """Yield the jobs no other worker has claimed (a claim is a file created exclusively), so
    workers can be added or removed at any time. Clear <out>/claims after killing workers."""
    os.makedirs(os.path.join(out, 'claims'), exist_ok=True)
    for c in todo:
        k = hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()
        try:
            os.close(os.open(os.path.join(out, 'claims', k), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        yield c


def load_results(out=OUT):
    """Every result row written by the workers to <out>/results_*.jsonl."""
    rows = []
    for f in glob.glob(os.path.join(out, 'results_*.jsonl')):
        with open(f) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


if __name__ == '__main__':
    parser = cli.parser(__doc__)
    parser.add_argument('--problems', default=','.join(toys.PROBLEMS))
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--experiments', default=None, help='comma-separated experiments to run (default: all)')
    parser.add_argument('--worker', default='0', help='name of this worker (its results file)')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=OUT)
    parser.add_argument('--list', action='store_true', help='count the jobs and exit')
    parser.add_argument('--quick', action='store_true', help='a few small jobs, to <out>/quick')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    problems = args.problems.split(',')
    todo = jobs(problems, args.seeds, args.experiments.split(',') if args.experiments else None)
    if args.quick:
        args.out = os.path.join(args.out, 'quick')
        todo = [dict(DEFAULTS, kind=k, budget=200_000, max_steps=1000, problem=p, seed=0)
                for p in problems for k in KINDS]
    done = {json.dumps(r['config'], sort_keys=True) for r in load_results(args.out)}
    todo = [c for c in todo if json.dumps(c, sort_keys=True) not in done]
    # big jobs first, so that the workers finish together
    todo.sort(key=lambda c: (-c['budget'], c['problem'], json.dumps(c, sort_keys=True)))
    print(f'{len(todo)} jobs to do')
    if args.list:
        raise SystemExit
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'results_{args.worker}.jsonl')
    torch.set_num_threads(args.threads)
    cache = {}
    for i, c in enumerate(claim(todo, args.out)):
        if c['problem'] not in cache:
            prob = toys.load(c['problem'], args.device)
            cache[c['problem']] = (prob, truth(prob))
        prob, T = cache[c['problem']]
        res = run(c, prob, T)
        with open(path, 'a') as f:
            f.write(json.dumps(dict(config=c, **res)) + '\n')
        print(f'[{i + 1}] {c["problem"]:12s} {c["kind"]:12s} B={c["budget"]:.0e} n={c["n"]} '
              f'delta={c["delta"]} {c["scheme"]} h{c["hidden"]}x{c["depth"]} p{c["p"]} {c["act"]}: '
              + ' '.join(f'{k} {v:.3g}' for k, v in res.items() if isinstance(v, float)), flush=True)
