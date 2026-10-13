import torch
from scipy.stats import qmc


def _make_index(row, cols):
    return [(row, c) for c in cols]


def _apply_index(params, index):
    """params[(row, col) for each row of index], for a (k, 2) tensor index."""
    return params[index[:, 0], index[:, 1]]


def sample_phi(lb, ub, n_samples=100, dim=43):
    """Scrambled Sobol designs in the box [lb, ub] (float64)."""
    u = torch.from_numpy(qmc.Sobol(d=dim, scramble=True).random(n_samples)) * 2 - 1
    return (u + 1) / 2 * (ub - lb) + lb
