"""The SHiP muon shield as a design problem: 65 free parameters -> hits on the sensitive plane.

The design u is normalized to the tolerance box: u = 0 is the nominal design, u_i = +-1 puts
parameter i at +-DELTA (10%) of its nominal value. The muons come from a muon source (muons.py,
by default the generative flow); the transport is cuda_muons_ship (MuonsAndMatter, GPU).

    python shield.py --n_muons 100000000        # hits of the nominal design
"""
import os

import numpy as np
import torch

from muons import make_muons
from paths import PROJECTS_DIR

# columns of one magnet's row; with NI_from_B the NI column is the field target B, in tesla
COLUMNS = ['Z_gap', 'Z_len', 'dXIn', 'dXOut', 'dYIn', 'dYOut', 'gapIn', 'gapOut',
           'dXYokeIn', 'dXYokeOut', 'dY_yokeIn', 'dY_yokeOut', 'XmgapIn', 'XmgapOut', 'NI']
UNITS = {c: 'cm' for c in COLUMNS} | {'NI': 'T'}

# the nominal design (of 2026-10-28; the earlier analyses used 'try_opt'): one row per magnet
NOMINAL = torch.tensor([
    [0.0, 115.0, 40.0, 40.0, 119.0, 119.0, 61.5, 61.5, 61.5, 61.5, 50.0, 50.0, 0.0, 0.0, 1.9875],
    [15.0, 150.0, 62.0, 62.0, 11.0, 11.0, 12.0, 12.0, 62.0, 62.0, 69.0, 69.0, 0.0, 0.0, 1.875],
    [15.0, 225.0, 70.0, 70.0, 12.0, 12.0, 9.0, 9.0, 70.0, 70.0, 77.0, 77.0, 0.0, 0.0, 1.87],
    [16.0, 225.0, 64.0, 64.0, 14.0, 14.0, 8.0, 8.0, 83.0, 83.0, 71.0, 71.0, 0.0, 0.0, 1.9],
    [15.0, 225.0, 42.0, 42.0, 13.0, 13.0, 8.0, 8.0, 109.0, 109.0, 47.0, 47.0, 0.0, 0.0, 1.89],
    [16.0, 131.0, 20.0, 20.0, 30.0, 30.0, 0.0, 0.0, 110.0, 110.0, 22.0, 22.0, 0.0, 0.0, 0.0],
    [15.0, 168.0, 20.0, 30.0, 30.0, 30.0, 22.0, 12.0, 108.0, 108.0, 33.0, 33.0, 0.0, 0.0, -1.32],
    [19.0, 200.0, 38.0, 69.0, 51.0, 51.0, 8.0, 8.0, 125.0, 94.0, 76.0, 76.0, 0.0, 0.0, -2.03]])

# the free parameters, as (magnet, column), in the order they were added, so a design of an earlier
# analysis is a prefix with the rest nominal: the 43 of the first analysis (magnets 1-7); the height
# dY of magnets 1-7 (dYOut = dYIn; 50, the second); the hadron absorber M0, the gap before M1 and
# the yoke height dY_yoke of every magnet (dY_yokeOut = dY_yokeIn; 65, the third)
_rect = [0, 1, 2, 6, 8, 14]                    # magnets 0-4: no Out parameters (full_design)
_free = [0, 1, 2, 3, 6, 7, 8, 9, 14]
FREE = ([(1, c) for c in [1, 2, 6, 8, 14]] + [(m, c) for m in (2, 3, 4) for c in _rect]
        + [(5, 0), (5, 1)] + [(m, c) for m in (6, 7) for c in _free] + [(m, 4) for m in range(1, 8)]
        + [(0, c) for c in [1, 2, 6, 8, 14, 4]] + [(1, 0)] + [(m, 10) for m in range(8)])


class MuonShield:
    """The design problem: its free parameters (phi_0, labels, units), the simulator (__call__)
    and the muons it is simulated with (self.muons, any source of muons.py)."""
    PLANES = [{'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 82},      # sensitive planes (m);
              {'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 91}]      # a hit is a muon inside the last
    DELTA = 0.1                                                    # u_i = +-1 is +-10%

    def __init__(self, muons='flow', uniform_fields=True):
        self.muons = make_muons(muons) if isinstance(muons, str) else muons
        self.uniform_fields = uniform_fields
        self.idx = torch.tensor(FREE)
        self.phi_0 = NOMINAL[self.idx[:, 0], self.idx[:, 1]].clone()      # (65,)
        self.dim = len(FREE)
        self.labels = [f'M{m}:{COLUMNS[c]}' for m, c in FREE]
        self.units = [UNITS[COLUMNS[c]] for _, c in FREE]

    def sample(self, n, temperature=1.0, device='cpu'):
        """n muons from the shield's source."""
        return self.muons.sample(n, temperature, device=device)

    def phi(self, u):
        """The physical design of a normalized one."""
        return self.phi_0.numpy() * (1 + self.DELTA * np.asarray(u))

    # ---- simulation ---------------------------------------------------------
    def full_design(self, phi):
        """65 free parameters (or a prefix of them, an earlier analysis' design, the rest nominal) ->
        the (8, 15) design, with the tied columns filled."""
        full = NOMINAL.clone()
        phi = torch.as_tensor(phi, dtype=full.dtype).cpu()
        idx = self.idx[:len(phi)]
        full[idx[:, 0], idx[:, 1]] = phi
        full[:, 13] = full[:, 12]                            # XmgapOut = XmgapIn
        full[:, 5] = full[:, 4]                              # dYOut = dYIn (one height per magnet)
        full[:, 11] = full[:, 10]                            # dY_yokeOut = dY_yokeIn
        full[:5, 3], full[:5, 7], full[:5, 9] = full[:5, 2], full[:5, 6], full[:5, 8]   # M0-M4 rectangular
        # (before the 65-parameter analysis only M0-M3, around 'try_opt': the r1_, r2_ pools)
        return full

    def __call__(self, phi, muons):
        """Per-muon hit flag (bool) of the physical design phi (65 free parameters) for muons (n, 7)."""
        from cuda_muons_ship import run_from_params
        out = run_from_params(
            self.full_design(phi).numpy(), muons,
            sensitive_plane=self.PLANES,
            n_steps=max(5000, int(np.ceil(self.PLANES[0]['position'] / 0.02)) + 100),
            fSC_mag=False,
            field_mode='uniform' if self.uniform_fields else 'simulate',
            field_map_file=None, NI_from_B=True, use_diluted=True, add_cavern=True, SND=False,
            return_all=True,
            histogram_dir=os.path.join(PROJECTS_DIR, 'MuonsAndMatter/cuda_muons/data'),
            seed=None)
        p = self.PLANES[-1]
        return ((out['x'].abs() <= p['dx'] / 2) & (out['y'].abs() <= p['dy'] / 2)
                & ((out['z'] - p['position']).abs() <= p['dz']))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_muons', type=int, default=int(1e8))
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--muons', default='flow', help='muon source (muons.make_muons)')
    parser.add_argument('--field_map', action='store_true', help='simulate the field map instead of uniform fields')
    args = parser.parse_args()
    shield = MuonShield(args.muons, uniform_fields=not args.field_map)
    hits = shield(shield.phi_0, shield.sample(args.n_muons, args.temperature))
    print(f'nominal design: {int(hits.sum())} hits over {args.n_muons:,} muons')
