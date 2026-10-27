"""The SHiP muon shield as a design problem: 50 free parameters -> hits on the sensitive plane.

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

# the nominal design ('try_opt'): one row per magnet
NOMINAL = torch.tensor([
    [0.0000, 115.0000, 40.0000, 40.0000, 119.0000, 119.0000, 61.5000, 61.5000, 61.5000, 61.5000, 50.0000, 50.0000, 0.0000, 0.0000, 1.9000],
    [15.0000, 151.5035, 63.1169, 63.1169, 10.0000, 10.0000, 8.1495, 9.1550, 64.1224, 63.1169, 69.4286, 69.4286, 0.0000, 0.0000, 1.9000],
    [15.0000, 224.9372, 68.0437, 68.0437, 10.0000, 10.0000, 8.4314, 8.5295, 70.6235, 70.5253, 74.8481, 74.8481, 0.0000, 0.0000, 1.9000],
    [16.1245, 225.0000, 64.8050, 64.8050, 13.9322, 13.9322, 8.9857, 9.1093, 85.8846, 85.7609, 71.2855, 71.2855, 0.0000, 0.0000, 1.9000],
    [15.0000, 223.9999, 39.0189, 39.0189, 11.0774, 11.0774, 8.3273, 8.9986, 97.5319, 96.8606, 46.2724, 46.2724, 0.0000, 0.0000, 1.9000],
    [15.0000, 120.3744, 20.0000, 20.0000, 30.0000, 30.0000, 0.0000, 0.0000, 102.7509, 102.7509, 22.0000, 22.0000, 0.0000, 0.0000, -0.0000],
    [15.0001, 168.0000, 20.0000, 30.0000, 40.0000, 40.0000, 24.9153, 9.3875, 97.1824, 102.7103, 33.0000, 33.0000, 0.0000, 0.0000, -1.2000],
    [15.0000, 193.0909, 37.9502, 67.8055, 47.7714, 47.7714, 8.0002, 8.0000, 124.7799, 94.9248, 74.5861, 74.5861, 0.0000, 0.0000, -1.9000]])

# the free parameters, as (magnet, column): the 43 of the first analysis, then the height dY of
# magnets 1-7 (dYOut = dYIn), so a 43-parameter design is the first 43 with the heights nominal
_rect = [0, 1, 2, 6, 8, 14]                    # magnets 1-4 are rectangular: In = Out
_free = [0, 1, 2, 3, 6, 7, 8, 9, 14]
FREE = ([(1, c) for c in [1, 2, 6, 8, 14]] + [(m, c) for m in (2, 3, 4) for c in _rect]
        + [(5, 0), (5, 1)] + [(m, c) for m in (6, 7) for c in _free] + [(m, 4) for m in range(1, 8)])


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
        self.phi_0 = NOMINAL[self.idx[:, 0], self.idx[:, 1]].clone()      # (50,)
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
        """50 free parameters (or the first analysis' 43, the heights nominal) -> the (8, 15) design,
        with the tied columns filled."""
        full = NOMINAL.clone()
        phi = torch.as_tensor(phi, dtype=full.dtype).cpu()
        idx = self.idx[:len(phi)]
        full[idx[:, 0], idx[:, 1]] = phi
        full[0, 3], full[0, 5] = full[0, 2], full[0, 4]      # magnet 0: Out = In
        full[:, 13] = full[:, 12]                            # XmgapOut = XmgapIn
        full[:, 5] = full[:, 4]                              # dYOut = dYIn (one height per magnet)
        full[:4, 3], full[:4, 7], full[:4, 9] = full[:4, 2], full[:4, 6], full[:4, 8]   # rectangular
        return full

    def __call__(self, phi, muons):
        """Per-muon hit flag (bool) of the physical design phi (50 free parameters) for muons (n, 7)."""
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
