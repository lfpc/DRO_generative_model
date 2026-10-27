"""The SHiP muon shield of MuonShieldChallenge, simulated with cuda_muons.

    min_phi  N_mu * [ P(muon hits the detector | phi) + lam * violation(phi) ]

    python3 problem.py                                   # hit rate of the reference design, 1e8 muons
    MUON_SHIELD_BACKEND=toy python3 problem.py --n_samples 1e5   # CPU smoke test

N_mu is the weighted number of muons in the data, so the risk reads as the expected number of
hits. The penalty (length, cost, cavern) is zero on feasible designs.

The DRO layer sees a muon as (x, y): x in R^6 standardised features, y in {+1, -1} the charge.
    f0      logit((|p| - p_min) / (p_max - p_min))
    f1, f2  log(1 + tan(theta) / t0) * (cos, sin)(azimuth)
    f3, f4  production x, y (m)
    f5      logit of the Beta variable behind z
Every point of R^6 is a valid muon, so transported or generated points can always be simulated.

The nominal P is the challenge's MuonsGenerator ('gmm', exact density) or the real-muon pool of
build_pool.py ('pool'). `log_prob` is the MuonsGenerator density in the feature coordinates.
The simulator has no gradient: gradient-based adversaries read a surrogate.
"""
import copy
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_DIR = os.path.join(HERE, '..', '..', '..', '..', 'MuonShieldChallenge')
GENERATOR_DIR = os.path.join(HERE, '..', '..', '..', '..', 'generator_muons')

# Settings shared by every script, including the cross-problem ones in ../outer etc.
CONFIG = os.environ.get('MUON_SHIELD_CONFIG', 'easy')     # easy (30 params) | hard (72)
BACKEND = os.environ.get('MUON_SHIELD_BACKEND', 'cuda')   # cuda | toy (CPU smoke test, numbers meaningless)
NOMINAL = 'gmm'                     # gmm | pool (real muons, run build_pool.py first)
POOL = os.path.join(HERE, 'models', 'pool.npy')
N_MU = 27148094403.25               # sum of weights of muons_data.h5
CHUNK = 10_000_000                  # muons per chunk of sampling / simulation / evaluation
DRO_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # adversaries, surrogate, flow

sys.path.insert(0, CHALLENGE_DIR)
from MuonsGenerator import MuonsGenerator  # noqa: E402
from muon_shield import MuonShieldProblem  # noqa: E402
from utils import load_config  # noqa: E402


class MuonShield:
    def __init__(self, config=CONFIG, device='cuda', backend=BACKEND, lam=1.0, n_ref=200_000):
        configs = os.path.join(CHALLENGE_DIR, 'configs')
        self.config, self.backend, self.lam = config, backend, lam
        self.dro_device, self.chunk, self.loss_scale = DRO_DEVICE, CHUNK, N_MU
        self.sim = MuonShieldProblem.from_config(
            load_config(os.path.join(configs, f'{config}.json'), output='hits'), device=device)
        self.generator = MuonsGenerator.from_file(os.path.join(configs, 'gmm_model.json'),
                                                  os.path.join(configs, 'z_beta_fit.json'))
        self.input_dim, self.design_dim = 6, self.sim.dim
        self.bounds = tuple(b.float() for b in self.sim.bounds)
        self._pool = None
        gen = self.generator
        self.gmm = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(probs=gen.cdf.diff(prepend=gen.cdf.new_zeros(1))),
            torch.distributions.MultivariateNormal(gen.means, scale_tril=gen.scale_tril))
        self.beta = torch.distributions.Beta(float(gen.z_beta['a']), float(gen.z_beta['b']))
        # feature scales from a fixed reference sample, so every script agrees
        ref = gen.sample(n_ref, seed=12345)
        self.t0 = float((ref[:, :2].norm(dim=1) / ref[:, 2]).median())
        self.loc, self.scale = torch.zeros(6), torch.ones(6)
        f = self._features(ref)
        self.loc, self.scale = f.mean(0), f.std(0)

    # -- the data ---------------------------------------------------------
    def _features(self, mu):
        """Raw muons (N, 7) -> standardised features (N, 6). The charge is dropped."""
        g = self.generator
        p = mu[:, :3].norm(dim=1)
        q = ((p - g.p_min) / (g.p_max - g.p_min)).clamp(1e-7, 1 - 1e-7)
        pt = mu[:, :2].norm(dim=1).clamp_min(1e-12)
        rho = torch.log1p(pt / mu[:, 2] / self.t0)
        # z is in the shield frame (m); the Beta was fitted in cm from the target start, at z_offset
        b = ((100 * (mu[:, 5] - g.z_offset) - g.z_beta['loc']) / g.z_beta['scale']).clamp(1e-7, 1 - 1e-7)
        f = torch.stack([torch.logit(q), rho * mu[:, 0] / pt, rho * mu[:, 1] / pt,
                         mu[:, 3], mu[:, 4], torch.logit(b)], 1)
        return (f - self.loc.to(f.device)) / self.scale.to(f.device)

    def features(self, mu):
        """Raw muons (N, 7) -> (standardised features, charge)."""
        return self._features(mu), -mu[:, 6] / 13

    def to_muons(self, x, y):
        """Features (N, 6) and charge (N,) -> the simulator's rows (N, 7)."""
        g = self.generator
        f = x * self.scale.to(x.device) + self.loc.to(x.device)
        p = g.p_min + (g.p_max - g.p_min) * torch.sigmoid(f[:, 0])
        rho = f[:, 1:3].norm(dim=1).clamp_min(1e-12)
        tan = self.t0 * torch.expm1(rho)
        pz = p / torch.sqrt(1 + tan ** 2)
        px, py = pz * tan * f[:, 1] / rho, pz * tan * f[:, 2] / rho
        z = (g.z_beta['loc'] + g.z_beta['scale'] * torch.sigmoid(f[:, 5])) / 100 + g.z_offset
        return torch.stack([px, py, pz, f[:, 3], f[:, 4], z, -13 * y.to(x.dtype)], 1)

    @property
    def pool(self):
        """The real-muon pool, (M, 7) raw muons, memory-mapped."""
        if self._pool is None:
            self._pool = np.load(POOL, mmap_mode='r')
        return self._pool

    def sample(self, n, generator=None):
        """(x, y): standardised features and charge of n muons from P."""
        if NOMINAL == 'gmm':
            return self._sample_gmm(n, generator)
        return self._sample_pool(n, generator)

    def _sample_gmm(self, n, generator=None):
        """Chunk i is drawn with seed + i."""
        seed = None if generator is None else int(torch.randint(2**62, (), generator=generator))
        xs, ys = [], []
        for i, start in enumerate(range(0, n, self.chunk)):
            mu = self.generator.sample(min(self.chunk, n - start),
                                       None if seed is None else seed + i).cpu()
            x, y = self.features(mu)
            xs.append(x)
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    def _sample_pool(self, n, generator=None):
        """n <= M: a random subset without replacement; n > M: with replacement."""
        M = self.pool.shape[0]
        idx = (torch.randperm(M, generator=generator)[:n] if n <= M
               else torch.randint(M, (n,), generator=generator))
        xs, ys = [], []
        for s in range(0, n, self.chunk):
            order, pos = idx[s:s + self.chunk].sort()          # sorted reads from the memory map
            rows = torch.from_numpy(self.pool[order.numpy()])
            mu = torch.empty_like(rows)
            mu[pos] = rows
            x, y = self.features(mu)
            xs.append(x)
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    def log_prob(self, x, y=None):
        """MuonsGenerator log-density of the features (and charge), in chunks."""
        out = torch.cat([self._log_prob(x[s:s + self.chunk].to(torch.float32).cpu())
                         for s in range(0, x.shape[0], self.chunk)])
        return out if y is None else out + math.log(0.5)

    def _log_prob(self, x):
        """Exact log-density of standardised features, by change of variables."""
        g = self.generator
        f = x * self.scale + self.loc
        u, rho = f[:, 0], f[:, 1:3].norm(dim=1).clamp_min(1e-12)
        v = math.log(self.t0) + torch.log(torch.expm1(rho))
        ab = torch.stack([(u - g.mean[0]) / g.std[0], (v - g.mean[1]) / g.std[1]], 1)
        lp = self.gmm.log_prob(ab) - torch.log(g.std).sum()           # density of (u, v)
        lp = lp - torch.log(-torch.expm1(-rho))                         # dv / drho
        lp = lp - math.log(2 * math.pi) - torch.log(rho)                # polar, uniform azimuth
        s, R, sig = f[:, 3:5].norm(dim=1), g.radius, g.sigma            # smeared ring: Bessel I0
        lp = lp - math.log(2 * math.pi * sig ** 2) - (s - R) ** 2 / (2 * sig ** 2) \
            + torch.log(torch.special.i0e(R * s / sig ** 2))
        b = torch.sigmoid(f[:, 5]).clamp(1e-7, 1 - 1e-7)
        lp = lp + self.beta.log_prob(b) + torch.log(b) + torch.log1p(-b)
        return lp + torch.log(self.scale).sum()

    # -- the shield -------------------------------------------------------
    def simulate(self, phi, x, y, chunk_index=0):
        """The muons after the shield, (N, 7) on the CPU. One chunk."""
        mu = self.to_muons(x.detach().float().cpu(), y.detach().cpu())
        if self.backend == 'toy':
            return toy_transport(self.sim.design_from_phi(phi), mu, self.sim.sensitive_plane)
        self.sim._muons, self.sim.n_samples = mu, len(mu)
        seed = self.sim.seed
        if seed is not None:                  # a distinct fixed seed per chunk
            self.sim.seed = seed + 1000 * chunk_index
        out = self.sim.simulate(phi.detach().cpu()).cpu()
        self.sim.seed = seed
        return out

    def hits(self, phi, x, y):
        """(N,) 1 for each muon that reaches the detector, 0 otherwise, on the CPU."""
        return torch.cat([self.sim.hits(self.simulate(phi, x[s:s + self.chunk], y[s:s + self.chunk], i)).float()
                          for i, s in enumerate(range(0, x.shape[0], self.chunk))])

    def penalty(self, phi):
        """lam * sum of relative constraint violations, 0 if feasible."""
        g = self.sim.constraints(phi.detach()).float()
        norm = torch.ones_like(g)
        norm[0], norm[1] = self.sim.L0, self.sim.W0
        return self.lam * (g / norm).clamp_min(0).sum()

    def loss(self, phi, x, y):
        """Per-muon N_mu * (hit + penalty)."""
        return self.loss_scale * (self.hits(phi, x, y) + self.penalty(phi))

    def risk(self, phi, x, y):
        return self.loss(phi, x, y).mean()

    def hit_rate(self, phi, x, y):
        return float(self.hits(phi, x, y).mean())

    def constraints(self, phi):
        return self.sim.constraints(phi)

    def project(self, phi):
        return torch.maximum(torch.minimum(phi, self.bounds[1]), self.bounds[0])

    def init_design(self):
        return self.sim.initial_phi.float().clone()

    def wrap_surrogate(self, model):
        """A surrogate of the packed (features, charge) row as a loss (phi, x, y) -> (N,).

        With a graph (an adversary's ascent, one chunk at a time) it returns the differentiable
        output on self.dro_device; without one it evaluates in chunks and returns on the CPU.
        """
        dev, chunk = self.dro_device, self.chunk
        m = copy.deepcopy(model).to(dev).eval()
        k = float(getattr(model, 'y_scale', 1.0))      # the fit is on the per-muon scale

        def row(x, y):
            return torch.cat([x.to(dev), y.to(dev).unsqueeze(1).to(x.dtype)], 1)

        def loss_fn(p, x, y):
            p = p.to(dev)
            if torch.is_grad_enabled() and x.requires_grad:
                return k * m(p, row(x, y))
            with torch.no_grad():
                return torch.cat([k * m(p, row(x[s:s + chunk], y[s:s + chunk])).cpu()
                                  for s in range(0, x.shape[0], chunk)])
        loss_fn.model = model
        return loss_fn


def toy_transport(design, mu, plane, n_slices=20, z0=5.0, dedx=1.3, p_stop=1.0):
    """A crude CPU stand-in for cuda_muons, only to smoke-test the pipeline without a GPU.

    Straight lines between thin slices of each magnet; in the iron a p_T kick 0.3 B dz and a
    constant energy loss. Nothing is calibrated.
    """
    d = design.float() / 100                                   # cm -> m (B untouched below)
    B = design[:, 14].float()
    px, py, pz, x, y, z = (mu[:, k].clone() for k in range(6))
    q = -mu[:, 6] / 13
    alive = torch.ones(len(mu), dtype=torch.bool)
    zc = z0
    for m in range(d.shape[0]):
        zc += d[m, 0]
        dz = 2 * d[m, 1] / n_slices
        for s in range(n_slices):
            t = (s + 0.5) / n_slices
            zs = zc + (s + 0.5) * dz
            step = (zs - z).clamp_min(0)
            x, y, z = x + px / pz * step, y + py / pz * step, torch.maximum(z, zs.expand_as(z))
            lerp = lambda a, b: d[m, a] + t * (d[m, b] - d[m, a])
            dx, dy, gap, yoke = lerp(2, 3), lerp(4, 5), lerp(6, 7), lerp(8, 9)
            ax, ay = x.abs() - d[m, 12] / 2, y.abs()
            core = (ax >= 0) & (ax < dx) & (ay < dy + yoke)
            ret = (ax >= dx + gap) & (ax < dx + gap + yoke) & (ay < dy + yoke)
            by = torch.where(core, B[m], torch.where(ret, -B[m], torch.zeros(())))
            px = px - 0.3 * q * by * dz
            iron = core | ret
            p = torch.sqrt(px ** 2 + py ** 2 + pz ** 2)
            k = ((p - dedx * dz * iron) / p).clamp_min(0)
            px, py, pz = px * k, py * k, (pz * k).clamp_min(1e-6)
            alive &= p * k > p_stop
        zc += 2 * d[m, 1]
    step = (plane['position'] - z).clamp_min(0)
    x, y = x + px / pz * step, y + py / pz * step
    zf = torch.where(alive, torch.full_like(z, plane['position']), torch.zeros_like(z))
    x = torch.where(alive, x, torch.full_like(x, 1e3))
    return torch.stack([px, py, pz, x, y, zf, mu[:, 6]], 1)


if __name__ == '__main__':  # simulate a design on muons from P
    import argparse
    import time

    class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    parser.add_argument('--config', default=CONFIG, help='name in MuonShieldChallenge/configs')
    parser.add_argument('--phi', default=None, help='text file with the free parameters (default: reference)')
    parser.add_argument('--n_samples', type=float, default=1e8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--backend', default=BACKEND, choices=['cuda', 'toy'])
    args = parser.parse_args()

    problem = MuonShield(args.config, device=f'cuda:{args.gpu}', backend=args.backend)
    phi = problem.init_design() if args.phi is None else torch.as_tensor(np.loadtxt(args.phi)).float()
    x, y = problem.sample(int(args.n_samples), torch.Generator().manual_seed(args.seed))

    t0 = time.time()
    hits = problem.hits(phi, x, y)
    print(f'\nProblem: {problem.config} ({problem.design_dim} free parameters), {len(x):,} muons, '
          f'backend {problem.backend}')
    print(f'  n_hits           {hits.sum().item():.6g}')
    print(f'  hit rate         {hits.mean().item():.6g}')
    print(f'  simulation_time  {time.time() - t0:.3g} s')
    print(f'  expected hits    {problem.loss_scale * hits.mean().item():.6g}  '
          f'(hit rate x N_mu = {problem.loss_scale:.5g} weighted muons)')
    print(f'  penalty          {problem.penalty(phi).item():.4g}')
    g = problem.constraints(phi)  # length, cost, then the cavern overlaps if simulated
    print('Constraints (feasible iff all <= 0):')
    for name, value in zip(['length', 'cost', 'cavern (max)'], [g[0], g[1]] + ([g[2:].max()] if len(g) > 2 else [])):
        print(f'  {name:<16} {value.item():.4g}' + ('  VIOLATED' if value > 0 else ''))
    n = int(hits.sum())
    print(f'\nWith {n} hits in {len(x):,} muons, the relative MC error of the hit rate is '
          f'{1 / max(n, 1) ** .5:.0%}.')
