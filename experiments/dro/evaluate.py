"""Shared evaluation: held-out shifts, the regret metric, design sampling, aggregation.

Every study uses these, so there is exactly one definition of "worst-case regret" in the
codebase rather than one per script.
"""
import numpy as np
import torch

from .baselines import erm


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------
def sobol_designs(lo, hi, m, seed=0):
    """Space-filling designs in the box [lo, hi].

    Space-filling rather than uniform random: with only a handful of designs, uniform
    draws routinely leave the interesting region uncovered, the surrogate then predicts
    almost nothing happening anywhere, and the optimizer walks to the cheapest corner.
    """
    eng = torch.quasirandom.SobolEngine(dimension=len(lo), scramble=True, seed=seed)
    u = eng.draw(m).to(lo.dtype)
    return lo + u * (hi - lo)


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------
def radial_designs(center, lo, hi, m, seed=0):
    """Space-filling designs pulled toward `center` by a random radial factor.

    Box-filling sequences concentrate near the surface as the dimension grows -- in 16
    dimensions almost every Sobol point sits in a thin shell, so a surrogate fitted on
    them never sees the neighbourhood of the nominal design where the optimum lives.
    Scaling each offset by a uniform factor spreads the radii from 0 outward instead.
    This is what lcso_resample.py does.
    """
    eng = torch.quasirandom.SobolEngine(dimension=len(lo), scramble=True, seed=seed)
    u = eng.draw(m).to(lo.dtype) * 2 - 1                     # box in [-1, 1]
    g = torch.Generator().manual_seed(seed)
    u = u * torch.rand(m, 1, generator=g, dtype=lo.dtype)    # pull toward the centre
    half = (hi - lo) / 2
    return (center + u * half).clamp(min=lo, max=hi)


def reference_best(problem, x, n_grid=61):
    """Best achievable risk on the distribution sampled by `x`.

    Subtracting this turns a raw objective into a regret, which removes the part of a
    shift that penalizes every design equally -- without it the metric largely restates
    nominal performance.
    """
    if not getattr(problem, 'uses_surrogate', True):
        # Exact-loss problems: optimize directly rather than searching the design box. A
        # space-filling search is hopeless once the design has more than two or three
        # coordinates -- on the six-item newsvendor it left the reference 49 units above
        # the achievable optimum, which is larger than every difference between methods
        # and made the whole regret column meaningless.
        lr = getattr(problem, 'ref_lr', 0.05)
        phi = erm(problem, problem.loss, x[:40000], steps=2500, lr=lr)
        return problem.risk(phi, x)
    lo, hi = problem.design_bounds()
    if problem.design_dim == 2:                      # exact: grid the design space
        bs = torch.linspace(float(lo[0]), float(hi[0]), n_grid, dtype=x.dtype)
        hs = torch.linspace(float(lo[1]), float(hi[1]), n_grid, dtype=x.dtype)
        cand = torch.stack(torch.meshgrid(bs, hs, indexing='ij'), -1).reshape(-1, 2)
    else:                                            # approximate: space-filling search
        cand = sobol_designs(lo, hi, 4000, seed=0)
    best = float('inf')
    for i in range(0, cand.shape[0], 256):
        best = min(best, float(problem.risk(cand[i:i + 256], x).min()))
    return best


def regrets(problem, phi, eval_sets, best):
    """Regret of a design on every evaluation distribution, in one pass.

    Returned as a dict so a caller that wants both the per-shift breakdown and the worst
    case pays for one evaluation of each distribution rather than two.
    """
    return {k: float(problem.risk(phi, x)) - best[k] for k, x in eval_sets.items()}


def worst_regret(problem, phi, eval_sets, best, precomputed=None):
    """max over held-out shifts of (risk - best achievable on that shift)."""
    r = precomputed if precomputed is not None else regrets(problem, phi, eval_sets, best)
    return max(r[s] for s in problem.held_out)


def worst_objective(problem, phi, eval_sets):
    """max over held-out shifts of the raw risk.

    Use this instead of regret whenever the per-shift optimum cannot be computed
    reliably -- above two design dimensions `reference_best` falls back to a
    space-filling search, which a method can beat, making "regret" go negative and
    meaningless. The raw objective is an unknown constant away from regret, but it is the
    *same* constant for every method on a given shift, so within-problem comparisons are
    unaffected.
    """
    return max(float(problem.risk(phi, eval_sets[s])) for s in problem.held_out)


def make_evaluation(problem, seed, n=200000):
    """(eval_sets, best) for one seed. `best` is what regret is measured against."""
    ev = problem.eval_sets(seed, n)
    return ev, {k: reference_best(problem, x) for k, x in ev.items()}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(rows, keys, by=('method', 'knob')):
    """Mean and standard error over seeds, grouped by `by`."""
    from collections import defaultdict
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
