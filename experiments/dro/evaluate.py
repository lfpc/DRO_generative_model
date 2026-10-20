"""Evaluation helpers: design sampling, regret against held-out shifts, aggregation over seeds.

Not imported by any current script.
"""
from collections import defaultdict

import numpy as np
import torch

from .baselines import erm


def sobol_designs(lo, hi, m, seed=0):
    """m space-filling designs in the box [lo, hi]."""
    eng = torch.quasirandom.SobolEngine(dimension=len(lo), scramble=True, seed=seed)
    u = eng.draw(m).to(lo.dtype)
    return lo + u * (hi - lo)


def radial_designs(center, lo, hi, m, seed=0):
    """Sobol designs pulled toward `center` by a uniform radial factor (as train_surrogate.py)."""
    eng = torch.quasirandom.SobolEngine(dimension=len(lo), scramble=True, seed=seed)
    u = eng.draw(m).to(lo.dtype) * 2 - 1
    g = torch.Generator().manual_seed(seed)
    u = u * torch.rand(m, 1, generator=g, dtype=lo.dtype)
    half = (hi - lo) / 2
    return (center + u * half).clamp(min=lo, max=hi)


def reference_best(problem, x, n_grid=61):
    """Best achievable risk on the sample x: exact optimisation for exact-loss problems,
    a grid in 2D, otherwise a Sobol search."""
    if not getattr(problem, 'uses_surrogate', True):
        lr = getattr(problem, 'ref_lr', 0.05)
        phi = erm(problem, problem.loss, x[:40000], steps=2500, lr=lr)
        return problem.risk(phi, x)
    lo, hi = problem.design_bounds()
    if problem.design_dim == 2:
        bs = torch.linspace(float(lo[0]), float(hi[0]), n_grid, dtype=x.dtype)
        hs = torch.linspace(float(lo[1]), float(hi[1]), n_grid, dtype=x.dtype)
        cand = torch.stack(torch.meshgrid(bs, hs, indexing='ij'), -1).reshape(-1, 2)
    else:
        cand = sobol_designs(lo, hi, 4000, seed=0)
    best = float('inf')
    for i in range(0, cand.shape[0], 256):
        best = min(best, float(problem.risk(cand[i:i + 256], x).min()))
    return best


def regrets(problem, phi, eval_sets, best):
    """{shift: risk - best} for every evaluation set."""
    return {k: float(problem.risk(phi, x)) - best[k] for k, x in eval_sets.items()}


def worst_regret(problem, phi, eval_sets, best, precomputed=None):
    """max over held-out shifts of (risk - best achievable on that shift)."""
    r = precomputed if precomputed is not None else regrets(problem, phi, eval_sets, best)
    return max(r[s] for s in problem.held_out)


def worst_objective(problem, phi, eval_sets):
    """max over held-out shifts of the raw risk (when the per-shift optimum is unreliable)."""
    return max(float(problem.risk(phi, eval_sets[s])) for s in problem.held_out)


def make_evaluation(problem, seed, n=200000):
    """(eval_sets, best) for one seed."""
    ev = problem.eval_sets(seed, n)
    return ev, {k: reference_best(problem, x) for k, x in ev.items()}


def aggregate(rows, keys, by=('method', 'knob')):
    """Mean and standard error over seeds, grouped by `by`."""
    acc = defaultdict(list)
    for r in rows:
        acc[tuple(r[b] for b in by)].append(r)
    out = []
    for key, rs in acc.items():
        rec = dict(zip(by, key))
        rec['n_seeds'] = len(rs)
        for k in keys:
            vals = [r[k] for r in rs if k in r and r[k] is not None
                    and np.isfinite(r[k])]
            if vals:
                rec[k] = float(np.mean(vals))
                rec[k + '_se'] = (float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                                  if len(vals) > 1 else 0.0)
        out.append(rec)
    return out
