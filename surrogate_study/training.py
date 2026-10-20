"""Fitting a surrogate to (design, sample, hit) rows.

Rows hold the designs U (M, D) and, per row, its design index d, sample x and hit y. offset is
added to every logit in the loss: -log(keep_neg) when only a fraction keep_neg of the misses
were kept (case-control), so that the model itself is of p(hit | u, x).

Schemes, over the same rows:
  fixed    every training design from the start. The learning rate drops 3x whenever the
           validation loss has not improved for `patience` evaluations; training stops after
           `drops` drops
  stream   the rolling buffer of train_surrogate.py: --buffer designs at a time, half of them
           replaced by unseen ones at each refresh, the refreshes spread evenly over
           max_steps, with a cosine learning rate
Both keep the weights that did best on the validation designs.
"""
import copy
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import qmc


@dataclass
class Rows:
    U: torch.Tensor          # (M, D)
    d: torch.Tensor          # (R,) long
    x: torch.Tensor          # (R, dx)
    y: torch.Tensor          # (R,) float
    offset: float = 0.0

    def take(self, ids):
        """The rows of designs ids, re-indexed."""
        ids = torch.as_tensor(ids, device=self.d.device)
        new = torch.full((self.U.shape[0],), -1, dtype=torch.long, device=self.d.device)
        new[ids] = torch.arange(len(ids), device=self.d.device)
        m = new[self.d] >= 0
        return Rows(self.U[ids], new[self.d[m]], self.x[m], self.y[m], self.offset)

    def to(self, dev):
        return Rows(self.U.to(dev), self.d.to(dev), self.x.to(dev), self.y.to(dev), self.offset)


def draw_designs(M, D, delta=1.0, radial=True, seed=0):
    """M designs u = delta * Sobol in [-1, 1]^D, pulled towards 0 by a U(0, 1) factor if radial
    (in high dimension box samples all sit near the surface, far from u = 0)."""
    with warnings.catch_warnings():                     # M need not be a power of 2
        warnings.simplefilter('ignore')
        u = qmc.Sobol(d=D, scramble=True, seed=seed).random(M) * 2 - 1
    if radial:
        u *= np.random.default_rng(seed).random((M, 1))
    return torch.as_tensor(delta * u, dtype=torch.float32)


def split(rows, val_frac, seed):
    """(train, val): a fraction of the designs (at least 2) held out for validation."""
    M = rows.U.shape[0]
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(seed))
    k = max(2, int(round(val_frac * M)))
    return rows.take(perm[k:]), rows.take(perm[:k])


def bce(model, rows, idx):
    z = model.logit_rows(rows.U, rows.d[idx], rows.x[idx]) + rows.offset
    return torch.nn.functional.binary_cross_entropy_with_logits(z, rows.y[idx])


@torch.no_grad()
def val_loss(model, val, chunk=2 ** 16):
    model.eval()
    R = val.y.shape[0]
    out = sum(float(bce(model, val, torch.arange(c, min(c + chunk, R), device=val.y.device))) * min(chunk, R - c)
              for c in range(0, R, chunk)) / R
    model.train()
    return out


def fit(model, train, val, scheme='fixed', buffer=64, max_steps=40000, batch=4096, lr=1e-3,
        eval_every=250, patience=6, drops=3, evals_per_epoch=None, seed=0, log=None):
    """Train in place; returns a dict of what happened. The weights end at the best validation.
    evals_per_epoch: validate that often per pass over the training rows instead (at least
    every eval_every steps), so that patience counts epochs on large training sets."""
    dev = train.y.device
    g = torch.Generator(device=dev).manual_seed(seed)
    model.set_norm(train.x[torch.randint(len(train.y), (200_000,), device=dev, generator=g)])
    if hasattr(model, 'gate'):
        model.gate.fill_(float(train.U.pow(2).mean().sqrt().clamp_min(1e-3)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    M = train.U.shape[0]
    if evals_per_epoch:
        eval_every = max(eval_every, len(train.y) // batch // evals_per_epoch)

    # the rows each phase draws from: fixed, one phase; stream, a sliding window over the designs
    if scheme == 'fixed' or buffer >= M:
        phases = [torch.arange(len(train.y), device=dev)]
    else:
        order = torch.randperm(M, generator=torch.Generator().manual_seed(seed)).to(dev)
        half = max(1, buffer // 2)
        starts = range(0, M - buffer + 1, half)
        rank = torch.empty(M, dtype=torch.long, device=dev)
        rank[order] = torch.arange(M, device=dev)
        r = rank[train.d]
        phases = [torch.nonzero((r >= s) & (r < s + buffer))[:, 0] for s in starts]
    per_phase = max(1, max_steps // len(phases))
    sched = None if scheme == 'fixed' else torch.optim.lr_scheduler.CosineAnnealingLR(opt, per_phase * len(phases))
    n_drops, stop = 0, False

    best, best_state, bad, step, curve = np.inf, None, 0, 0, []
    model.train()
    for k, rows_k in enumerate(phases):
        for _ in range(per_phase):
            idx = rows_k[torch.randint(len(rows_k), (min(batch, len(rows_k)),), device=dev, generator=g)]
            loss = bce(model, train, idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            step += 1
            if step % eval_every == 0:
                v = val_loss(model, val)
                curve.append((step, v))
                if v < best:
                    best, bad, best_state = v, 0, copy.deepcopy(model.state_dict())
                else:
                    bad += 1
                if log:
                    log(f'    step {step:6d} phase {k + 1}/{len(phases)} val {v:.6g} best {best:.6g}')
                if sched is None and bad >= patience:
                    if n_drops == drops:
                        stop = True
                        break
                    n_drops, bad = n_drops + 1, 0
                    model.load_state_dict(best_state)            # continue from the best, slower
                    for grp in opt.param_groups:
                        grp['lr'] /= 3
        if stop:
            break
    if best_state is None:
        best_state, best = copy.deepcopy(model.state_dict()), val_loss(model, val)
    model.load_state_dict(best_state)
    model.eval()
    return dict(steps=step, val_bce=best, phases=len(phases), curve=curve)
