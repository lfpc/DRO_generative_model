"""The SHiP muon shield as a design problem: 65 free parameters -> hits on the sensitive plane.

The design u is normalized to the tolerance box: u = 0 is the nominal design, u_i = +-1 puts
parameter i at +-DELTA (10%) of its nominal value. The muons come from a muon source (muons.py,
by default the generative flow); the transport is cuda_muons_ship (MuonsAndMatter, GPU), in the
field map of snoopy's FEM (fields='map', the default) or a uniform field per iron block
(fields='uniform'). A design's map is computed once, on the CPU, and kept in paths.FIELD_MAPS
(one folder per design; delete them when snoopy changes).

    python shield.py --n_muons 100000000        # hits of the nominal design
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from muons import make_muons
from paths import FIELD_MAPS, PROJECTS_DIR, ROOT

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

USE_DILUTED = True            # snoopy's yoke with diluted steel (its non-diluted FEM fails on M5)
# a field map, in a fresh process: snoopy's process pool hangs in one that already runs CUDA
_WRITE_MAP = ("import os, sys, numpy as np\n"
              "from utils_cuda_muons.get_magnetic_field import get_magnetic_field_from_params\n"
              "d = sys.argv[1]\n"
              "get_magnetic_field_from_params(np.load(os.path.join(d, 'design.npy')), field_mode='simulate',\n"
              "    field_map_file=os.path.join(d, 'field.h5'), fSC_mag=False, NI_from_B=True,\n"
              f"    use_diluted={USE_DILUTED}, cores_field=int(sys.argv[2]))\n")


class MuonShield:
    """The design problem: its free parameters (phi_0, labels, units), the simulator (__call__)
    and the muons it is simulated with (self.muons, any source of muons.py)."""
    PLANES = [{'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 82},      # sensitive planes (m);
              {'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 91}]      # a hit is a muon inside the last
    DELTA = 0.1                                                    # u_i = +-1 is +-10%

    def __init__(self, muons='flow', fields='map', map_dir=FIELD_MAPS, map_workers=4, map_cores=8):
        if fields not in ('map', 'uniform'):
            raise ValueError(f"fields must be 'map' or 'uniform', not {fields!r}")
        self.muons = make_muons(muons) if isinstance(muons, str) else muons
        self.fields = fields
        self.map_dir, self.map_cores = map_dir, map_cores       # map_workers maps at once, map_cores each
        self._maps = ThreadPoolExecutor(map_workers)
        self._pending = {}                                      # map file -> its future (prefetch)
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

    # ---- designs ------------------------------------------------------------
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

    # ---- field maps ---------------------------------------------------------
    def map_file(self, phi):
        """Where the field map of design phi is (or will be): a folder per design (the design as
        the simulator sees it, rounded to 0.01)."""
        full = np.round(self.full_design(phi).numpy().astype(np.float64), 2) + 0.0     # no -0.0
        return os.path.join(self.map_dir, hashlib.sha1(full.tobytes()).hexdigest()[:20], 'field.h5')

    def prefetch(self, phis):
        """Start computing, in the background, the field maps of designs phis that are not there
        yet; field_map() then waits for them."""
        if self.fields != 'map':
            return
        for phi in phis:
            path = self.map_file(phi)
            if path not in self._pending and not os.path.exists(path):
                self._pending[path] = self._maps.submit(self._make_map, self.full_design(phi).numpy(), path)

    def field_map(self, phi):
        """The field map file of design phi, computed if it is not there yet."""
        path = self.map_file(phi)
        if path in self._pending:
            self._pending.pop(path).result()
        if not os.path.exists(path):
            self._make_map(self.full_design(phi).numpy(), path)
        return path

    def drop_map(self, phi):
        """Delete the field map of design phi (a pool's designs are simulated once)."""
        shutil.rmtree(os.path.dirname(self.map_file(phi)), ignore_errors=True)

    def _make_map(self, full, path):
        os.makedirs(self.map_dir, exist_ok=True)
        tmp = tempfile.mkdtemp(dir=self.map_dir, prefix='tmp_')
        try:
            np.save(os.path.join(tmp, 'design.npy'), full)
            r = subprocess.run([sys.executable, '-c', _WRITE_MAP, tmp, str(self.map_cores)], cwd=ROOT,
                               env=os.environ | {'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
            if r.returncode or not os.path.exists(os.path.join(tmp, 'field.h5')):
                raise RuntimeError(f'the field map of\n{full}\nfailed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}')
            try:
                os.rename(tmp, os.path.dirname(path))
            except OSError:                       # another process wrote it meanwhile
                pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ---- simulation ---------------------------------------------------------
    def __call__(self, phi, muons):
        """Per-muon hit flag (bool) of the physical design phi (65 free parameters) for muons (n, 7)."""
        from cuda_muons_ship import run_from_params
        field_map = self.field_map(phi) if self.fields == 'map' else None
        out = run_from_params(
            self.full_design(phi).numpy(), muons,
            sensitive_plane=self.PLANES,
            n_steps=max(5000, int(np.ceil(self.PLANES[0]['position'] / 0.02)) + 100),
            fSC_mag=False,
            field_mode='read_file' if field_map else 'uniform',
            field_map_file=field_map, NI_from_B=True, use_diluted=USE_DILUTED, add_cavern=True, SND=False,
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
    parser.add_argument('--fields', default='map', choices=['map', 'uniform'],
                        help="snoopy's FEM field map, or a uniform field per iron block")
    args = parser.parse_args()
    shield = MuonShield(args.muons, fields=args.fields)
    hits = shield(shield.phi_0, shield.sample(args.n_muons, args.temperature))
    print(f'nominal design: {int(hits.sum())} hits over {args.n_muons:,} muons')
