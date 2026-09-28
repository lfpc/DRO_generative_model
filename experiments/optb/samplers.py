"""How the designs around the incumbent are drawn.

`models.py::sample_phi` in the project root offers normal / uniform / lhs / sobol through
`scipy.stats.qmc`; scipy is not installed here, so the two quasi-random ones are written
out. All four return points in the normalized box [-1, 1]^d, which the caller scales by
the trust radius and centres on the incumbent.

The radial modifier is applied on top, as `lcso_resample.py` does:

    u = sample_phi(k) * rand(k, 1)

Its effect grows with dimension and is easy to miss. A box-filling sequence in sixteen
dimensions puts almost every point near the surface -- the mean radius of a uniform draw
in [-1,1]^d is close to its maximum -- so a surrogate fitted on them never sees the
neighbourhood of the incumbent, which is precisely where the gradient is wanted. Scaling
each offset by a uniform factor spreads the radii from zero outward instead.
"""
import torch


def _sobol(k, d, g):
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g))
    eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    return eng.draw(k) * 2 - 1


def _lhs(k, d, g):
    """Latin hypercube: one point per stratum in every coordinate, strata permuted
    independently per coordinate. Guarantees one-dimensional uniformity exactly, which
    plain uniform sampling only achieves in expectation."""
    edges = torch.arange(k, dtype=torch.get_default_dtype()).unsqueeze(1).repeat(1, d)
    for j in range(d):
        edges[:, j] = edges[torch.randperm(k, generator=g), j]
    jitter = torch.rand(k, d, generator=g, dtype=torch.get_default_dtype())
    return ((edges + jitter) / k) * 2 - 1


def _normal(k, d, g):
    """Gaussian about the incumbent, scaled so that most mass lands inside the box."""
    return torch.randn(k, d, generator=g, dtype=torch.get_default_dtype()) * 0.5


def _uniform(k, d, g):
    return torch.rand(k, d, generator=g, dtype=torch.get_default_dtype()) * 2 - 1


SAMPLERS = {'sobol': _sobol, 'lhs': _lhs, 'normal': _normal, 'uniform': _uniform}


def draw_designs(kind, k, d, g, radial=True):
    """k points in [-1, 1]^d, optionally pulled toward the centre by a uniform factor."""
    u = SAMPLERS[kind](k, d, g).to(torch.get_default_dtype())
    if radial:
        u = u * torch.rand(k, 1, generator=g, dtype=u.dtype)
    return u
