"""Set up the generator: the pretrained flow of generator_muons, validated against P.

    python3 train_generator.py                # CPU is enough
    python3 train_generator.py --no_data      # skip the 1e8-muon models/data.pt

Nothing is trained. `PretrainedFlow` wraps the generator_muons spline flow as a differentiable
map latent -> problem.py features. The flow's fit space folds away the global azimuth and the
sign of dphi; two extra latents set them, so the generator is one map of an 8-D normal and
the latent adversary moves only the flow's six.

Writes models/generator.pt (a copy of the checkpoint), models/data.pt (nominal muons in
features, for the integration study) and figs/generator.png.
"""
import argparse
import importlib.util
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from problem import CHUNK, DRO_DEVICE, GENERATOR_DIR, MuonShield   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figs')
MODELS = os.path.join(HERE, 'models')
DEFAULT_FLOW = os.path.join(GENERATOR_DIR, 'outputs', 'flow_model_complete_ploss.pt')


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _nsf_module():
    """generator_muons/models.py, imported under its own name so it clashes with nothing."""
    spec = importlib.util.spec_from_file_location('generator_muons_models',
                                                  os.path.join(GENERATOR_DIR, 'models.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PretrainedFlow(nn.Module):
    """(z (n, 6), extra (n, 2)) -> (features (n, 6), charge (n,)), differentiable in z.

    extra[:, 0] sets the global azimuth through the normal CDF, the sign of extra[:, 1] the
    sign of dphi. The flow's fit space is (logit_p, log_tan_theta, dphi, r, z, charge),
    positions in cm from the target start.
    """

    dim = 6

    def __init__(self, ckpt, problem):
        super().__init__()
        self.nsf = _nsf_module().NeuralSplineFlow.from_config(ckpt['config'])
        self.nsf.load_state_dict(ckpt['state_dict'])
        self.nsf.eval()
        for p in self.nsf.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.as_tensor(ckpt['mean'], dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(ckpt['std'], dtype=torch.float32))
        self.m = {k: float(ckpt[k]) for k in ('p_min', 'p_max', 'r_scale', 'z0', 'z_scale', 'r_max')}
        self.problem = problem

    def fit_space(self, z):
        for layer in reversed(self.nsf.layers):
            z = layer.inverse(z)
        return z * self.std + self.mean

    def muons(self, z, extra):
        """Raw muons (n, 7) in the simulator's units: GeV, m, pdg_id."""
        s, m = self.fit_space(z), self.m
        p = m['p_min'] + (m['p_max'] - m['p_min']) * torch.sigmoid(s[:, 0])
        pz = p / torch.sqrt(1 + torch.exp(2 * s[:, 1]))
        pt = pz * torch.exp(s[:, 1])
        dphi = s[:, 2].abs() % (2 * math.pi)
        dphi = torch.minimum(dphi, 2 * math.pi - dphi) * torch.where(extra[:, 1] > 0, 1.0, -1.0)
        r = (m['r_scale'] * torch.sinh(s[:, 3])).clamp(0, m['r_max']) / 100
        zpos = (m['z0'] + m['z_scale'] * torch.sinh(s[:, 4])) / 100 + self.problem.generator.z_offset  # cm -> m, shield frame
        charge = torch.where(s[:, 5] >= 0, 1.0, -1.0)
        phi_r = 2 * math.pi * torch.special.ndtr(extra[:, 0])
        phi_p = phi_r + dphi
        return torch.stack([pt * torch.cos(phi_p), pt * torch.sin(phi_p), pz,
                            r * torch.cos(phi_r), r * torch.sin(phi_r), zpos, -13 * charge], 1)

    def forward(self, z, extra):
        return self.problem.features(self.muons(z, extra))


class Generator:
    """The pretrained flow as one map of an 8-D standard normal: packed (features, charge)."""

    latent_dim = 8

    def __init__(self, flow, ckpt=None):
        self.flow, self.ckpt = flow, ckpt

    def __call__(self, z):
        """On the CPU, computed in chunks on DRO_DEVICE."""
        flow = self.flow.to(DRO_DEVICE)
        out = []
        with torch.no_grad():
            for s in range(0, z.shape[0], CHUNK):
                zc = z[s:s + CHUNK].to(DRO_DEVICE, torch.float32)
                x, y = flow(zc[:, :6], zc[:, 6:8])
                out.append(torch.cat([x, y.unsqueeze(1)], 1).cpu())
        self.flow.cpu()
        return torch.cat(out)

    def save(self, path):
        torch.save(self.ckpt, path)

    @staticmethod
    def load(path, problem=None):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        problem = problem or MuonShield(device='cpu', backend='toy')   # only its feature map
        return Generator(PretrainedFlow(ckpt, problem), ckpt)


def physical(mu):
    return {'|p| (GeV)': mu[:, :3].norm(dim=1), '$p_T$ (GeV)': mu[:, :2].norm(dim=1),
            'production r (m)': mu[:, 3:5].norm(dim=1), 'production z (m)': mu[:, 5]}


def validate(problem, gen, n, seed=0):
    """Nominal muons against generated ones: the MuonsGenerator log-density and the marginals."""
    g = torch.Generator().manual_seed(seed)
    xr, yr = problem.sample(n, g)
    xf = gen(torch.randn(n, 8, generator=g))
    xf, yf = xf[:, :6], xf[:, 6]
    lr, lf = problem.log_prob(xr, yr), problem.log_prob(xf, yf)
    print(f'  MuonsGenerator log-density: nominal {lr.mean():.3f}   generated {lf.mean():.3f}   '
          f'(gap {lr.mean() - lf.mean():+.3f} nats)')
    print(f'  charge +1 fraction: nominal {(yr > 0).float().mean():.4f}   '
          f'generated {(yf > 0).float().mean():.4f}')
    print(f'  max |feature|: nominal {xr.abs().max():.1f}   generated {xf.abs().max():.1f}')
    mr, mf = problem.to_muons(xr, yr), problem.to_muons(xf, yf)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), constrained_layout=True)
    for j, name in enumerate(['logit |p|', 'slope x (log-radial)', 'slope y (log-radial)', 'x0']):
        ax = axes[0, j]
        bins = torch.linspace(-5, 5, 90)
        ax.hist(xr[:, j], bins=bins, density=True, histtype='stepfilled', alpha=.35, label='nominal')
        ax.hist(xf[:, j], bins=bins, density=True, histtype='step', lw=1.5, ls='--', color='C3',
                label='flow')
        ax.set(title=f'feature: {name}', yticks=[])
        ax.legend(fontsize=7)
    pr, pf = physical(mr), physical(mf)
    for j, k in enumerate(pr):
        ax = axes[1, j]
        hi = float(torch.quantile(pr[k][:1_000_000], .999))
        bins = torch.linspace(float(pr[k].min()), hi, 90)
        ax.hist(pr[k], bins=bins, density=True, histtype='stepfilled', alpha=.35, label='nominal')
        ax.hist(pf[k], bins=bins, density=True, histtype='step', lw=1.5, ls='--', color='C3',
                label='flow')
        ax.set(title=k, yscale='log' if j < 2 else 'linear')
        ax.legend(fontsize=7)
    fig.suptitle(f'pretrained flow vs {n:,} nominal muons -- log-density nominal {lr.mean():.2f}, '
                 f'generated {lf.mean():.2f}', fontsize=10)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'generator.png'), dpi=150)
    print(f'  wrote {OUT}/generator.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--flow', default=DEFAULT_FLOW, help='generator_muons flow checkpoint')
    ap.add_argument('--n_val', type=float, default=1e6, help='muons in the validation')
    ap.add_argument('--n_data', type=float, default=1e8, help='nominal muons in models/data.pt')
    ap.add_argument('--no_data', action='store_true', help='skip writing models/data.pt')
    a = ap.parse_args()
    problem = MuonShield(device='cpu', backend='toy')     # the feature map and the judge only
    ckpt = torch.load(a.flow, map_location='cpu', weights_only=False)
    gen = Generator(PretrainedFlow(ckpt, problem), ckpt)
    print(f'loaded {a.flow}: {ckpt["config"]}')
    validate(problem, gen, int(a.n_val))
    os.makedirs(MODELS, exist_ok=True)
    gen.save(os.path.join(MODELS, 'generator.pt'))
    if not a.no_data:
        torch.save(problem.sample(int(a.n_data), torch.Generator().manual_seed(7)),
                   os.path.join(MODELS, 'data.pt'))
    print(f'  saved generator{"" if a.no_data else " and the nominal sample"} to {MODELS}/')
