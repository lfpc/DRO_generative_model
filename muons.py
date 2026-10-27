"""Muon sources: the muons the simulator transports and the surrogate is integrated over.

A muon is a row (px, py, pz, x, y, z, pdg): momentum in GeV, production point in m, pdg -13 for
mu+ and +13 for mu-. A source has

    sample(n, temperature=1, device='cpu')     n muons, (n, 7); follows torch.manual_seed
    points(spec, n_ref=1e9, temperature=1)     weighted muons (x, w) standing for n_ref muons,
                                               to integrate the surrogate over

temperature scales the momentum distribution (training pools are drawn hot, > 1, to get more of
the muons that make the hits); the analysis always uses 1, the physical flux.

  FlowMuons   the generative flow of generator_muons (momentum, charge and production point).
              points: a quadrature grid 'n_z x n_phi[:n_pos]', or 'mc:<n>' (sampled)
  FileMuons   stored muons, an (N, 7) .npy; points: 'all' (every row) or 'mc:<n>'

make_muons('flow' | 'flow:<path>' | 'file:<path>') builds one; every script takes it as --muons.
Another generator plugs in as a class with sample() (Muons.points then samples).
"""
import importlib
import os
import sys

import numpy as np
import torch

from paths import GENERATOR_DIR, MUON_FLOW


class Muons:
    device = 'cpu'

    def sample(self, n, temperature=1.0, device='cpu'):
        raise NotImplementedError

    def load(self):
        """Build what sampling needs (called before any seeding: building may draw random numbers)."""

    def points(self, spec, n_ref=1e9, temperature=1.0, batch=2 ** 21, seed=0):
        """'mc:<n>': n sampled muons, each of weight n_ref / n."""
        kind, _, n = spec.partition(':')
        if kind != 'mc':
            raise ValueError(f'{type(self).__name__}: unknown points {spec!r}')
        n = int(float(n))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for c in range(0, n, batch):
                x = self.sample(min(batch, n - c), temperature, device=self.device)
                yield x, torch.full((len(x),), n_ref / n, dtype=torch.float64, device=x.device)


class FlowMuons(Muons):
    """The muon flow of generator_muons: (pz, pt) from a 2-d normal latent through the momentum
    flow, the azimuth uniform, mu+ with the data's fraction, the production point from the
    position flow given (pz, pt, charge). The flow works in cm, everything here in m.
    on_axis: every muon at x = y = 0, z = -1 m instead (the first analysis)."""
    # kinematic envelope (GeV); outside it the simulator's CUDA kernel can crash
    P_MAX, PT_MAX, PZ_MIN = 400.0, 14.0, 1.0

    def __init__(self, path=MUON_FLOW, on_axis=False):
        self.path, self.on_axis, self.gen = path, on_axis, None

    def load(self):
        """generator_muons' src.flows.load_generator. Its package is called 'src', so it is imported
        with GENERATOR_DIR first on the path, then sys.modules is restored."""
        if self.gen is not None:
            return
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'src' or k.startswith('src.')}
        sys.path.insert(0, GENERATOR_DIR)
        try:
            flows = importlib.import_module('src.flows')
        finally:
            sys.path.remove(GENERATOR_DIR)
            for k in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
                del sys.modules[k]
            sys.modules.update(saved)
        self.gen = flows.load_generator(self.path, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.device = next(self.gen.flow.parameters()).device

    def _std(self, s, cols):
        """Undo the flow's standardization of columns cols."""
        std, mean = (torch.as_tensor(v[cols], device=self.device) for v in (self.gen.std, self.gen.mean))
        return s * std + mean

    def momentum(self, uv):
        """(u, v) of the flow's fit space -> (pz, pt) in GeV (generator_muons src.data.fit_to_momentum)."""
        m = self.gen.m
        p = m['p_min'] + (m['p_max'] - m['p_min']) / (1 + torch.exp(-uv[:, 0]))
        pz = p / torch.sqrt(1 + torch.exp(2 * uv[:, 1]))
        return pz, pz * torch.exp(uv[:, 1])

    def position(self, abc, phi_p):
        """(a, b, c) of the fit space and the momentum azimuth -> (x, y, z) in m (src.data.fit_to_position,
        in cm): |dphi| gets a random sign, r is folded to |r| and capped at r_max."""
        m = self.gen.m
        sign = torch.where(torch.rand(len(abc), device=abc.device) < 0.5, -1.0, 1.0).double()
        dphi = np.pi / (1 + torch.exp(-abc[:, 0])) * sign
        r = torch.clamp((m['r_scale'] * torch.sinh(abc[:, 1])).abs(), max=m['r_max'])
        z = m['z_min'] + (m['z_max'] - m['z_min']) / (1 + torch.exp(-abc[:, 2]))
        return r * torch.cos(phi_p - dphi) / 100, r * torch.sin(phi_p - dphi) / 100, z / 100

    def latent_momentum(self, z):
        """Momentum latent points (k, 2) -> (flow-space momentum y (k, 2), pz, pt)."""
        y = z.float()
        for layer in reversed(self.gen.flow.momentum.layers):
            y = layer.inverse(y, None)
        return (y,) + self.momentum(self._std(y.double(), slice(0, 2)))

    def in_envelope(self, p, pt, pz):
        return (p <= self.P_MAX) & (pt <= self.PT_MAX) & (pz > self.PZ_MIN)

    @torch.no_grad()
    def _draw(self, n, temperature):
        """n muons on the flow's device, the momentum latent scaled by temperature."""
        gen, dev = self.gen, self.device
        q = torch.where(torch.rand(n, device=dev) < gen.mu_plus, 1.0, -1.0).double()
        s = self._std(gen.flow.sample(((q - gen.mean[5]) / gen.std[5]).float(), temperature).double(), slice(None))
        pz, pt = self.momentum(s[:, 0:2])
        phi_p = torch.rand(n, device=dev, dtype=torch.float64) * 2 * np.pi
        x, y, z = self.position(s[:, 2:5], phi_p)
        return torch.stack([pt * torch.cos(phi_p), pt * torch.sin(phi_p), pz, x, y, z, -13 * q], 1).float()

    def sample(self, n, temperature=1.0, device='cpu'):
        """n muons inside the kinematic envelope."""
        self.load()
        parts, got = [], 0
        while got < n:
            x = self._draw(int(1.02 * (n - got)) + 16, temperature)
            x = x[self.in_envelope(x[:, :3].norm(dim=1), x[:, :2].norm(dim=1), x[:, 2])][:n - got]
            if self.on_axis:
                x[:, 3:5], x[:, 5] = 0.0, -1.0
            parts.append(x.to(device))
            got += x.shape[0]
        return torch.cat(parts)

    def points(self, spec, n_ref=1e9, temperature=1.0, batch=2 ** 21, seed=0):
        """'n_z x n_phi[:n_pos]' (e.g. '801x64'): the quadrature grid; 'mc:<n>': sampled."""
        if spec.startswith('mc:'):
            return super().points(spec, n_ref, temperature, batch, seed)
        grid, _, n_pos = spec.partition(':')
        n_z, n_phi = map(int, grid.split('x'))
        return self.grid(n_z, n_phi, n_ref, temperature, batch=batch, n_pos=int(n_pos or 1), seed=seed)

    @torch.no_grad()
    def grid(self, n_z=801, n_phi=64, n_ref=1e9, temperature=1.0, L=7.0, batch=2 ** 21, n_pos=1, seed=0):
        """The flow's muons as a quadrature grid: yields (muons (k, 7) on the flow's device,
        weights (k,) float64), the weights summing to n_ref over the envelope.

        The momentum, azimuth and charge are independent in the flow, so: a trapezoid grid of
        n_z^2 latent points on [-L, L]^2 with Gaussian weights (scaled by temperature), n_phi
        azimuth midpoints and both charges, with no sampling noise (experiments/quadrature.py).
        The production point depends on the momentum: each node gets n_pos draws from the flow's
        position part, from a fixed seed, so the grid is the same set of muons every time."""
        self.load()
        gen, dev = self.gen, self.device
        zs = torch.linspace(-L, L, n_z, dtype=torch.float64, device=dev)
        wz = torch.exp(-zs ** 2 / 2) * (zs[1] - zs[0]) / np.sqrt(2 * np.pi)
        Z = torch.stack(torch.meshgrid(zs, zs, indexing='ij'), -1).view(-1, 2)
        W = (wz[:, None] * wz[None, :]).reshape(-1)
        keep = W > 1e-18 * W.max()
        y, pz, pt = self.latent_momentum(Z[keep] * temperature)
        ok = self.in_envelope((pz ** 2 + pt ** 2).sqrt(), pt, pz)
        y, pz, pt, W = y[ok], pz[ok], pt[ok], W[keep][ok] * n_ref / W[keep][ok].sum()
        phis = (torch.arange(n_phi, dtype=torch.float64, device=dev) + 0.5) * 2 * np.pi / n_phi
        K = n_phi * n_pos
        per = max(1, batch // K)
        with torch.random.fork_rng(devices=[dev] if dev.type == 'cuda' else []):
            torch.manual_seed(seed)
            for c in range(0, len(pz), per):
                a, b, w = (v[c:c + per].repeat_interleave(K) for v in (pz, pt, W))
                ctx = y[c:c + per].repeat_interleave(K, 0)
                ph = phis.repeat_interleave(n_pos).repeat(len(a) // K)
                z0 = torch.zeros_like(a)
                for q, pq in ((1.0, gen.mu_plus), (-1.0, 1 - gen.mu_plus)):
                    if self.on_axis:
                        pos = (z0, z0, z0 - 1)
                    else:
                        qs = torch.full((len(a), 1), (q - gen.mean[5]) / gen.std[5], device=dev)
                        abc = gen.flow.position.sample(0, context=torch.cat([ctx, qs.float()], 1)).double()
                        pos = self.position(self._std(abc, slice(2, 5)), ph)
                    x = torch.stack([b * torch.cos(ph), b * torch.sin(ph), a, *pos, z0 - 13 * q], 1)
                    yield x.float(), w * pq / K


class FileMuons(Muons):
    """Stored muons, an (N, 7) float32 .npy. sample draws rows at random (with replacement)."""

    def __init__(self, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.path, self.device, self.x = path, device, None

    def load(self):
        if self.x is None:
            self.x = torch.from_numpy(np.load(self.path, mmap_mode='r')[:].astype(np.float32))

    def sample(self, n, temperature=1.0, device='cpu'):
        if temperature != 1:
            raise ValueError('stored muons have no temperature')
        self.load()
        return self.x[torch.randint(len(self.x), (n,))].to(device)

    def points(self, spec='all', n_ref=1e9, temperature=1.0, batch=2 ** 21, seed=0):
        """'all': every stored muon, each of weight n_ref / N; 'mc:<n>': sampled."""
        if spec != 'all':
            return super().points(spec, n_ref, temperature, batch, seed)
        self.load()
        return ((self.x[c:c + batch].to(self.device), torch.full((len(self.x[c:c + batch]),), n_ref / len(self.x),
                                                                 dtype=torch.float64, device=self.device))
                for c in range(0, len(self.x), batch))


def make_muons(spec='flow'):
    """'flow' (paths.MUON_FLOW), 'flow:<checkpoint>', 'file:<muons.npy>'."""
    kind, _, path = (spec or 'flow').partition(':')
    if kind == 'flow':
        return FlowMuons(path or MUON_FLOW)
    if kind == 'file':
        return FileMuons(os.path.expanduser(path))
    raise ValueError(f'unknown muon source {spec!r}')
