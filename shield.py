"""The SHiP muon shield as a design problem: 43 free parameters -> hits on the sensitive plane.

Muons come from the generative flow in ~/projects/generator_muons; the transport is
cuda_muons_ship (GPU). Set PROJECTS_DIR if the projects are not in ~/projects.

    python shield.py --n_muons 100000000        # hits of the nominal design
"""
import importlib.util
import os
import sys

import numpy as np
import torch

PROJECTS_DIR = os.path.expanduser(os.getenv('PROJECTS_DIR', '~/projects'))
GENERATOR_DIR = os.path.join(PROJECTS_DIR, 'generator_muons')

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

# the free parameters, as (magnet, column)
_rect = [0, 1, 2, 6, 8, 14]                    # magnets 1-4 are rectangular: In = Out
_free = [0, 1, 2, 3, 6, 7, 8, 9, 14]
FREE = ([(1, c) for c in [1, 2, 6, 8, 14]] + [(m, c) for m in (2, 3, 4) for c in _rect]
        + [(5, 0), (5, 1)] + [(m, c) for m in (6, 7) for c in _free])


class MuonShield:
    # kinematic envelope of the flow muons (GeV); outside it the CUDA kernel can crash
    P_MAX, PT_MAX, PZ_MIN = 400.0, 14.0, 1.0
    # sensitive planes; a hit is a muon inside the last one
    PLANES = [{'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 82},
              {'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 91}]

    def __init__(self, uniform_fields=True):
        self.uniform_fields = uniform_fields
        self.idx = torch.tensor(FREE)
        self.phi_0 = NOMINAL[self.idx[:, 0], self.idx[:, 1]].clone()      # (43,)
        self.labels = [f'M{m}:{COLUMNS[c]}' for m, c in FREE]
        self.units = [UNITS[COLUMNS[c]] for _, c in FREE]
        self._flow = None

    # ---- muons: columns px, py, pz, x, y, z, pdg ----------------------------
    def _load_flow(self):
        """generator_muons' generator (src.flows.load_generator). Its package is called 'src',
        so it is imported with GENERATOR_DIR first on the path, then sys.modules is restored."""
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'src' or k.startswith('src.')}
        sys.path.insert(0, GENERATOR_DIR)
        try:
            flows = importlib.import_module('src.flows')
        finally:
            sys.path.remove(GENERATOR_DIR)
            for k in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
                del sys.modules[k]
            sys.modules.update(saved)
        self._flow = flows.load_generator(os.path.join(GENERATOR_DIR, 'outputs', 'muon_flow.pt'),
                                          device='cuda' if torch.cuda.is_available() else 'cpu')

    def sample(self, n, temperature=1.0):
        """n muons from the flow, its momentum latent scaled by `temperature`, inside the
        kinematic envelope and on axis at z = -1. Follows torch.manual_seed."""
        if self._flow is None:
            self._load_flow()
        parts, got = [], 0
        while got < n:
            rng = np.random.default_rng(int(torch.randint(2 ** 62, (1,))))
            x = torch.from_numpy(self._flow.sample(n - got, momentum_temperature=temperature, rng=rng))
            pt = x[:, :2].norm(dim=1)
            x = x[(x[:, :3].norm(dim=1) <= self.P_MAX) & (pt <= self.PT_MAX) & (x[:, 2] > self.PZ_MIN)]
            x[:, 3:5], x[:, 5] = 0.0, -1.0
            parts.append(x[:, :7])
            got += x.shape[0]
        return torch.cat(parts)

    @torch.no_grad()
    def grid(self, n_z=801, n_phi=64, n_ref=1e9, temperature=1.0, L=7.0, batch=2 ** 21):
        """The flow's muons as a quadrature grid instead of a sample: yields (muons (k, 7) on the
        flow's device, weights (k,) float64), the weights summing to n_ref over the envelope.

        On axis, a muon is its momentum, azimuth and charge, independent in the flow: (pz, pt) is
        the momentum flow's image of a 2-d normal latent (scaled by `temperature`), the azimuth is
        uniform and mu+ has the data's fraction. So: a trapezoid grid of n_z^2 latent points on
        [-L, L]^2 with Gaussian weights, n_phi azimuth midpoints, and both charges. Sums over it
        are expectations over the flow, with no sampling noise (surrogate_study/quadrature.py)."""
        if self._flow is None:
            self._load_flow()
        gen = self._flow
        dev = next(gen.flow.parameters()).device
        zs = torch.linspace(-L, L, n_z, dtype=torch.float64, device=dev)
        wz = torch.exp(-zs ** 2 / 2) * (zs[1] - zs[0]) / np.sqrt(2 * np.pi)
        Z = torch.stack(torch.meshgrid(zs, zs, indexing='ij'), -1).view(-1, 2)
        W = (wz[:, None] * wz[None, :]).reshape(-1)
        keep = W > 1e-18 * W.max()
        y = (Z[keep] * temperature).float()
        for layer in reversed(gen.flow.momentum.layers):
            y = layer.inverse(y, None)
        uv = y.double() * torch.as_tensor(gen.std[:2], device=dev) + torch.as_tensor(gen.mean[:2], device=dev)
        m = gen.m
        p = m['p_min'] + (m['p_max'] - m['p_min']) / (1 + torch.exp(-uv[:, 0]))
        pz = p / torch.sqrt(1 + torch.exp(2 * uv[:, 1]))
        pt = pz * torch.exp(uv[:, 1])
        ok = (p <= self.P_MAX) & (pt <= self.PT_MAX) & (pz > self.PZ_MIN)
        pz, pt, W = pz[ok], pt[ok], W[keep][ok] * n_ref / W[keep][ok].sum()
        phis = (torch.arange(n_phi, dtype=torch.float64, device=dev) + 0.5) * 2 * np.pi / n_phi
        per = max(1, batch // n_phi)
        for c in range(0, len(pz), per):
            a, b, w = (v[c:c + per].repeat_interleave(n_phi) for v in (pz, pt, W))
            ph = phis.repeat(len(a) // n_phi)
            z0 = torch.zeros_like(a)
            for q, pq in ((1.0, gen.mu_plus), (-1.0, 1 - gen.mu_plus)):
                x = torch.stack([b * torch.cos(ph), b * torch.sin(ph), a, z0, z0, z0 - 1, z0 - 13 * q], 1)
                yield x.float(), w * pq / n_phi

    # ---- simulation ---------------------------------------------------------
    def full_design(self, phi):
        """43 free parameters -> the (8, 15) design, with the tied columns filled."""
        full = NOMINAL.clone()
        full[self.idx[:, 0], self.idx[:, 1]] = torch.as_tensor(phi, dtype=full.dtype).cpu()
        full[0, 3], full[0, 5] = full[0, 2], full[0, 4]      # magnet 0: Out = In
        full[:, 13] = full[:, 12]                            # XmgapOut = XmgapIn
        full[:4, 3], full[:4, 7], full[:4, 9] = full[:4, 2], full[:4, 6], full[:4, 8]   # rectangular
        return full

    def __call__(self, phi, muons):
        """Per-muon hit flag (bool) of design phi (43 free parameters)."""
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
    parser.add_argument('--field_map', action='store_true', help='simulate the field map instead of uniform fields')
    args = parser.parse_args()
    shield = MuonShield(uniform_fields=not args.field_map)
    hits = shield(shield.phi_0, shield.sample(args.n_muons, args.temperature))
    print(f'nominal design: {int(hits.sum())} hits over {args.n_muons:,} muons')
