"""Tune LCSO on two moons, one knob group at a time.

    python3 tune_lcso.py stage1      surrogate x step
    python3 tune_lcso.py stage2      designs per fit x samples per design
    python3 tune_lcso.py stage3      surrogate capacity and fit length

Staged rather than a full grid: the knobs are not independent, but a full cross of them is
hundreds of runs, and the first stage settles which surrogate is even in contention. Every
run gets the same sample budget and is scored by the true risk of the design it returns,
against Adam on the exact gradient as the reference.
"""
import itertools
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optb.lcso import Budget, lcso                                # noqa: E402
from train_lcso import MoonsForLCSO, exact_gradient_descent       # noqa: E402

BUDGET = 2_000_000


def run(problem, seeds=2, **kw):
    out = []
    for s in range(seeds):
        b = Budget(BUDGET)
        phi, _ = lcso(problem, b, seed=s, **kw)
        out.append(problem.true_objective(phi))
    return sum(out) / len(out)


def show(name, rows):
    print(f'\n--- {name}')
    for label, value, secs in sorted(rows, key=lambda r: r[1]):
        print(f'    {label:46s} {value:8.4f}   ({secs:.0f}s)')


def stage1(problem):
    rows = []
    for sur, step in [('mlp', 'gradient'), ('taylor', 'gradient'),
                      ('taylor', 'tr-newton')]:
        t0 = time.time()
        v = run(problem, surrogate=sur, step=step, scheme='resample', sampler='sobol',
                n_samples=20000, epochs=150)
        rows.append((f'{sur} + {step}', v, time.time() - t0))
    show('stage 1: surrogate x step', rows)


def stage2(problem, sur='taylor', step='tr-newton'):
    rows = []
    D = problem.dim
    for mult, n_samp in itertools.product((1, 2, 4), (800, 3200, 12800, 51200)):
        t0 = time.time()
        v = run(problem, surrogate=sur, step=step, scheme='resample', sampler='sobol',
                n_designs=mult * D, n_samples=n_samp, epochs=150)
        per = max(200, n_samp // 8)
        cost = mult * D * per
        rows.append((f'{mult}D designs ({mult * D}) x {per}/design = {cost // 1000}k/round,'
                     f' ~{BUDGET // max(cost, 1)} steps', v, time.time() - t0))
    show('stage 2: designs per fit x samples per design', rows)


def stage3(problem, sur='taylor', step='tr-newton', n_designs=None, n_samples=20000):
    rows = []
    for epochs, rank in itertools.product((80, 200, 500), (4, 8, 16)):
        t0 = time.time()
        v = run(problem, surrogate=sur, step=step, scheme='resample', sampler='sobol',
                n_designs=n_designs, n_samples=n_samples, epochs=epochs, rank=rank)
        rows.append((f'{epochs} epochs, rank {rank}', v, time.time() - t0))
    show('stage 3: fit length x surrogate capacity', rows)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    stage = sys.argv[1] if len(sys.argv) > 1 else 'stage1'
    kind = sys.argv[2] if len(sys.argv) > 2 else 'poly'
    par = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    problem = MoonsForLCSO(kind=kind, par=par)
    _, tr = exact_gradient_descent(problem, BUDGET)
    print(f'{kind}{par}: design dim {problem.dim};  exact gradient reaches {tr[-1][1]:.4f};'
          f'  initialisation {problem.true_objective(problem.init_design()):.4f}')
    {'stage1': stage1, 'stage2': stage2, 'stage3': stage3}[stage](problem)
