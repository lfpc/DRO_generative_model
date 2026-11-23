"""Fitting a surrogate to (design, sample, hit) rows.

Rows hold the designs U (M, D) and, per row, its design index d, sample x and hit y. offset is
added to every logit in the loss: -log(keep_neg) when only a fraction keep_neg of the misses
were kept (case-control), so that the model itself is of p(hit | u, x); a tensor (R,), one per
row, when the rows come from pools that kept different fractions.

Every training design is used from the start. The learning rate drops 3x whenever the validation
loss has not improved for `patience` evaluations; training stops after `drops` drops, keeping the
weights that did best on the validation designs.
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Rows:
    U: torch.Tensor          # (M, D)
    d: torch.Tensor          # (R,) long
    x: torch.Tensor          # (R, dx)
    y: torch.Tensor          # (R,) float
    offset: float | torch.Tensor = 0.0

    def off(self, idx):
        """The logit offset of rows idx."""
        return self.offset[idx] if torch.is_tensor(self.offset) else self.offset

    def take(self, ids):
        """The rows of designs ids, re-indexed."""
        ids = torch.as_tensor(ids, device=self.d.device)
        new = torch.full((self.U.shape[0],), -1, dtype=torch.long, device=self.d.device)
        new[ids] = torch.arange(len(ids), device=self.d.device)
        m = new[self.d] >= 0
        return Rows(self.U[ids], new[self.d[m]], self.x[m], self.y[m], self.off(m))

    def to(self, dev):
        off = self.offset.to(dev) if torch.is_tensor(self.offset) else self.offset
        return Rows(self.U.to(dev), self.d.to(dev), self.x.to(dev), self.y.to(dev), off)


def split(rows, val_frac, seed):
    """(train, val): a fraction of the designs (at least 2) held out for validation."""
    M = rows.U.shape[0]
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(seed))
    k = max(2, int(round(val_frac * M)))
    return rows.take(perm[k:]), rows.take(perm[:k])


def bce(model, rows, idx):
    z = model.logit_rows(rows.U, rows.d[idx], rows.x[idx]) + rows.off(idx)
    return torch.nn.functional.binary_cross_entropy_with_logits(z, rows.y[idx])


@torch.no_grad()
def val_loss(model, val, chunk=2 ** 16):
    model.eval()
    R = val.y.shape[0]
    out = sum(float(bce(model, val, torch.arange(c, min(c + chunk, R), device=val.y.device))) * min(chunk, R - c)
              for c in range(0, R, chunk)) / R
    model.train()
    return out


def fit(model, train, val, max_steps=40000, batch=4096, lr=1e-3, eval_every=250, patience=6, drops=3,
        evals_per_epoch=None, seed=0, log=None):
    """Train in place; returns a dict of what happened. The weights end at the best validation.
    evals_per_epoch: validate that often per pass over the training rows instead (at least
    every eval_every steps), so that patience counts epochs on large training sets."""
    dev = train.y.device
    g = torch.Generator(device=dev).manual_seed(seed)
    model.set_norm(train.x[torch.randint(len(train.y), (200_000,), device=dev, generator=g)])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if evals_per_epoch:
        eval_every = max(eval_every, len(train.y) // batch // evals_per_epoch)
    R = len(train.y)
    best, best_state, bad, n_drops, curve = np.inf, None, 0, 0, []
    model.train()
    for step in range(1, max_steps + 1):
        loss = bce(model, train, torch.randint(R, (min(batch, R),), device=dev, generator=g))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every:
            continue
        v = val_loss(model, val)
        curve.append((step, v))
        if v < best:
            best, bad, best_state = v, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if log:
            log(f'    step {step:6d} val {v:.6g} best {best:.6g}')
        if bad >= patience:
            if n_drops == drops:
                break
            n_drops, bad = n_drops + 1, 0
            model.load_state_dict(best_state)            # continue from the best, slower
            for grp in opt.param_groups:
                grp['lr'] /= 3
    if best_state is None:
        best_state, best = copy.deepcopy(model.state_dict()), val_loss(model, val)
    model.load_state_dict(best_state)
    model.eval()
    return dict(steps=step, val_bce=best, curve=curve)
