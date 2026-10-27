"""Build the real-muon pool for NOMINAL = 'pool' in problem.py.

    python3 build_pool.py                     # CPU and disk only, reads ~74 GB once

Draws --size muons with replacement, in proportion to their weights, from
generator_muons/muons_data.h5 and writes models/pool.npy, float32 (M, 7): px, py, pz (GeV),
x, y, z (m, shield frame), pdg_id. Two streaming passes: chunk weight sums, then draws per chunk.
The file's effective sample size is only ~2.7e7, so a 1e8 pool repeats heavy muons.
"""
import argparse
import os
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, '..', '..', '..', '..', 'generator_muons', 'muons_data.h5')
COLUMNS = ('px', 'py', 'pz', 'x', 'y', 'z', 'pdg')
CM = ('x', 'y', 'z')
Z_OFFSET = -2.14   # m, the target start in the shield frame (MuonsGenerator.z_offset)


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--data', default=DEFAULT_DATA, help='weighted muon file (h5)')
    ap.add_argument('--out', default=os.path.join(HERE, 'models', 'pool.npy'))
    ap.add_argument('--size', type=float, default=1e8, help='muons in the pool')
    ap.add_argument('--chunk', type=float, default=2e7, help='rows read at a time')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    M, C = int(a.size), int(a.chunk)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    with h5py.File(a.data, 'r') as f:
        N = f['weight'].shape[0]
        starts = list(range(0, N, C))
        sums, sq = np.empty(len(starts)), 0.0
        for i, s in enumerate(starts):                       # pass 1: chunk weight sums
            w = f['weight'][s:s + C].astype(np.float64)
            sums[i], sq = w.sum(), sq + (w ** 2).sum()
        W = sums.sum()
        print(f'{N:,} muons, sum of weights {W:.4g}, effective sample size {W ** 2 / sq:,.0f} '
              f'({time.time() - t0:.0f} s)')

        counts = rng.multinomial(M, sums / W)                # pass 2: draws per chunk
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        pool = np.lib.format.open_memmap(a.out, mode='w+', dtype=np.float32, shape=(M, 7))
        at = 0
        for s, k in zip(starts, counts):
            if k == 0:
                continue
            w = f['weight'][s:s + C].astype(np.float64)
            c = np.cumsum(w)
            idx = np.searchsorted(c, rng.random(k) * c[-1], side='right').clip(max=len(w) - 1)
            for j, col in enumerate(COLUMNS):
                v = f[col][s:s + C][idx].astype(np.float32)
                pool[at:at + k, j] = v / 100 if col in CM else v
            pool[at:at + k, 5] += Z_OFFSET
            at += k
        pool.flush()
    print(f'wrote {M:,} muons to {a.out} ({time.time() - t0:.0f} s)')
