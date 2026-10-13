"""Designs around the incumbent, as points in [-1, 1]^d (the caller scales and centres them).

radial=True multiplies each point by a uniform factor, as lcso_resample.py does: in high
dimension box-filling points all sit near the surface, and the fit would never see the
neighbourhood of the incumbent, where the gradient is wanted.
"""
import torch


def _sobol(k, d, g):
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g))
    eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    return eng.draw(k) * 2 - 1


def _uniform(k, d, g):
    return torch.rand(k, d, generator=g, dtype=torch.get_default_dtype()) * 2 - 1


SAMPLERS = {'sobol': _sobol, 'uniform': _uniform}


def draw_designs(kind, k, d, g, radial=True):
    """k points in [-1, 1]^d, optionally pulled toward the centre by a uniform factor."""
    u = SAMPLERS[kind](k, d, g).to(torch.get_default_dtype())
    if radial:
        u = u * torch.rand(k, 1, generator=g, dtype=u.dtype)
    return u
