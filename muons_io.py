"""Muon blocks and the surrogate-vs-simulator sweep shared by the muon-shield test scripts
(active_subspace_test.py, grad_test.py, optimization_test.py)."""
import time

import h5py
import numpy as np
import torch

COLS = ['px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight']


def param_labels(shield):
    """'M<magnet>:<param>' for each free design parameter."""
    return [f'M{m}:{shield.idx_mag[i].split("[")[0]}' for m, i in shield.params_idx.tolist()]


def count_muons(shield, n_muons, gen):
    """Resolve --n_muons: 0 means the size of the muon file; real data is capped at it."""
    if gen and n_muons > 0:
        return n_muons
    with h5py.File(shield.muons_file, 'r') as f:
        n_file = f['px'].shape[0]
    return min(n_muons, n_file) if n_muons > 0 else n_file


class Muons:
    """Muons [lo, hi) from the generative flow ('gen', unweighted) or the real file ('data',
    weighted). A block depends only on its offsets, so two sweeps see the same muons."""

    def __init__(self, shield, source, seed=0, temperature=1.0):
        self.shield, self.gen = shield, source == 'gen'
        self.seed, self.temperature = seed, temperature
        if self.gen:
            # Build the flow before any seeding: its init draws from the global RNG, so a
            # lazy build inside the first block would shift that block's muons.
            shield._load_flow()

    def describe(self):
        if self.gen:
            return f'generative flow at temperature {self.temperature:g} (unweighted)'
        return f'{self.shield.muons_file} (weighted)'

    def read(self, lo, hi):
        """(raw muons, weights); weights is None for 'gen'."""
        if self.gen:
            torch.manual_seed(self.seed + lo)
            return self.shield.sample_gen(n_samples=hi - lo, temperature=self.temperature), None
        with h5py.File(self.shield.muons_file, 'r') as f:
            x = np.stack([np.asarray(f[k][lo:hi], dtype=np.float32) for k in COLS], axis=1)
        x[:, 3], x[:, 4], x[:, 5] = 0.0, 0.0, -1.0       # same origin reset as _load_muons
        x = torch.from_numpy(x)
        return x, x[:, 7]


def evaluate_designs(model, shield, muons, PHI, n, block, chunk, n_repeats, simulate,
                     device, label='sweep 2'):
    """Surrogate and simulator at every physical design in PHI, on the same n muons.

    Returns (pred, pvar, sim, svar, sumw), all weighted by the muon weights (1 for 'gen'):
      pred = sum w p, pvar = sum w^2 p(1-p)         surrogate and its Bernoulli spread
      sim[r] = sum w hit, svar[r] = sum w^2 hit     simulator per repeat, its counting error
      sumw = sum w
    """
    n_des = PHI.shape[0]
    pred, pvar = np.zeros(n_des), np.zeros(n_des)
    sim, svar = np.zeros((n_repeats, n_des)), np.zeros((n_repeats, n_des))
    sumw = 0.0
    PHI_D = PHI.to(device)
    t0 = time.time()
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        raw, w = muons.read(lo, hi)
        sumw += float(raw.shape[0] if w is None else w.double().sum())
        with torch.no_grad():
            for c in range(0, raw.shape[0], chunk):
                xb = raw[c:c + chunk, :7].to(device)
                pr = model.predict_proba(PHI_D, xb.unsqueeze(0))       # (n_des, chunk)
                q = pr * (1 - pr)
                if w is not None:
                    wb = w[c:c + chunk].to(device)
                    pr, q = pr * wb, q * wb ** 2
                pred += pr.sum(1).double().cpu().numpy()
                pvar += q.sum(1).double().cpu().numpy()
        if simulate:
            wd = None if w is None else w.double()
            for i in range(n_des):
                for r in range(n_repeats):
                    hits = shield(PHI[i], raw).double()   # a new transport seed every call
                    sim[r, i] += float(hits.sum() if wd is None else (hits * wd).sum())
                    svar[r, i] += float(hits.sum() if wd is None else (hits * wd ** 2).sum())
            del wd
        print(f'  {label}: {hi:,}/{n:,} muons ({time.time() - t0:.0f}s)')
        del raw, w
    return pred, pvar, sim, svar, sumw
