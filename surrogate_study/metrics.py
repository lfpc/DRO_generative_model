"""Scores of a surrogate's H(u), gradient and Hessian at u = 0 against a reference.

  h0_err      H_hat(0) / H(0) - 1
  pred_in     median |log H_hat / H| over designs uniform in the half box (|u_i| <= 1/2)
  pred_box    the same over the whole box (|u_i| <= 1)
  grad_err    |g_hat - g| / |g|;  grad_cos = cos(g_hat, g)
  hess_err    |H_hat - H|_F / |H|_F;  eig_cos = |cos| of the top-|eigenvalue| eigenvectors
  worst       the TRUE rise at the surrogate's worst design in the ball |u| <= radius (default 1:
              one parameter alone at its tolerance), over the true worst rise there (1 = the
              surrogate finds the worst case)
"""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # after this folder: both have a plots.py
from analysis import trust_region_max  # noqa: E402


def ball_max(F, inits, radius, steps=500, lr=0.02):
    """argmax of F (B, D) -> (B,) over |u| <= radius: projected Adam ascent from each init,
    keeping the best point visited (the iterates need not settle)."""
    u = inits.clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr * radius)
    best_f, best_u = -np.inf, None
    for _ in range(steps + 1):
        f = F(u)
        i = int(f.argmax())
        if float(f[i]) > best_f:
            best_f, best_u = float(f[i]), u[i].detach().clone()
        opt.zero_grad()
        (-f.sum()).backward()
        opt.step()
        with torch.no_grad():
            u.mul_((radius / u.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1))
    return best_u


def inits(g, H, radius, n_random=6, seed=0):
    """Starts of the ascent on the sphere of the given radius: the trust-region maximiser of
    the quadratic model, +-gradient, +-the top two eigenvectors, and random directions."""
    g, H = g.detach().double(), H.detach().double()
    with np.errstate(all='ignore'):
        v = trust_region_max(g.numpy(), H.numpy(), radius)
    Q = torch.linalg.eigh(H)[1][:, -2:].T.numpy()
    rng = np.random.default_rng(seed)
    starts = [v, g.numpy(), -g.numpy(), *Q, *-Q] + list(rng.standard_normal((n_random, len(v))))
    return torch.as_tensor(np.stack([radius * s / max(np.linalg.norm(s), 1e-12) for s in starts]), dtype=torch.float32)


def top_eigvec(H):
    ev, Q = torch.linalg.eigh(H)
    return Q[:, ev.abs().argmax()]


def derivative_scores(g_hat, H_hat, g, H):
    return dict(grad_err=float((g_hat - g).norm() / g.norm()),
                grad_cos=float(g_hat @ g / (g_hat.norm() * g.norm()).clamp_min(1e-300)),
                hess_err=float((H_hat - H).norm() / H.norm()),
                eig_cos=float((top_eigvec(H_hat) @ top_eigvec(H)).abs()))


def pred_scores(H0_hat, H0, H_in_hat, H_in, H_box_hat, H_box):
    return dict(h0_err=H0_hat / H0 - 1,
                pred_in=float(np.median(np.abs(np.log(np.maximum(H_in_hat, 1e-300) / H_in)))),
                pred_box=float(np.median(np.abs(np.log(np.maximum(H_box_hat, 1e-300) / H_box)))))
