"""The SHiP muon shield as a design problem: phi -> hits on the sensitive plane (cuda_muons).

Run the nominal design once:

    python problems.py --n_samples 1000000
    python problems.py --uniform_muons --save_data
"""
import os

import h5py
import numpy as np
import torch

from utils import _make_index, _apply_index

PROJECTS_DIR = os.path.expanduser(os.getenv('PROJECTS_DIR', '~/projects'))
GENERATOR_DIR = os.path.join(PROJECTS_DIR, 'generator_muons')


class ShipMuonShieldCuda:
    idx_mag = {0: 'Z_gap[cm]', 1: 'Z_len[cm]',
               2: 'dXIn[cm]', 3: 'dXOut[cm]',
               4: 'dYIn[cm]', 5: 'dYOut[cm]',
               6: 'gapIn[cm]', 7: 'gapOut[cm]',
               8: 'ratio_yokesIn', 9: 'ratio_yokesOut',
               10: 'dY_yokeIn[cm]', 11: 'dY_yokeOut[cm]',
               12: 'XmgapIn[cm]', 13: 'XmgapOut[cm]',
               14: 'NI[A]'}

    # nominal design ('try_opt'): one row per magnet, columns as in idx_mag
    DEFAULT_PHI = torch.tensor([
        [0.0000, 115.0000, 40.0000, 40.0000, 119.0000, 119.0000, 61.5000, 61.5000, 61.5000, 61.5000, 50.0000, 50.0000, 0.0000, 0.0000, 1.9000],
        [15.0000, 151.5035, 63.1169, 63.1169, 10.0000, 10.0000, 8.1495, 9.1550, 64.1224, 63.1169, 69.4286, 69.4286, 0.0000, 0.0000, 1.9000],
        [15.0000, 224.9372, 68.0437, 68.0437, 10.0000, 10.0000, 8.4314, 8.5295, 70.6235, 70.5253, 74.8481, 74.8481, 0.0000, 0.0000, 1.9000],
        [16.1245, 225.0000, 64.8050, 64.8050, 13.9322, 13.9322, 8.9857, 9.1093, 85.8846, 85.7609, 71.2855, 71.2855, 0.0000, 0.0000, 1.9000],
        [15.0000, 223.9999, 39.0189, 39.0189, 11.0774, 11.0774, 8.3273, 8.9986, 97.5319, 96.8606, 46.2724, 46.2724, 0.0000, 0.0000, 1.9000],
        [15.0000, 120.3744, 20.0000, 20.0000, 30.0000, 30.0000, 0.0000, 0.0000, 102.7509, 102.7509, 22.0000, 22.0000, 0.0000, 0.0000, -0.0000],
        [15.0001, 168.0000, 20.0000, 30.0000, 40.0000, 40.0000, 24.9153, 9.3875, 97.1824, 102.7103, 33.0000, 33.0000, 0.0000, 0.0000, -1.2000],
        [15.0000, 193.0909, 37.9502, 67.8055, 47.7714, 47.7714, 8.0002, 8.0000, 124.7799, 94.9248, 74.5861, 74.5861, 0.0000, 0.0000, -1.9000]])
    n_params = 15
    MUON = 13
    # kinematic envelope of the flow sampler (GeV); outside it the CUDA kernel can crash
    P_MAX = 400.0
    PT_MAX = 14.0
    PZ_MIN = 1.0

    # free parameters (magnet, column), keyed by name; picked by their count, dimensions_phi
    parametrization = {
        'robustness': (
            _make_index(1, [1, 2, 6, 8, 14]) +
            _make_index(2, [0, 1, 2, 6, 8, 14]) +
            _make_index(3, [0, 1, 2, 6, 8, 14]) +
            _make_index(4, [0, 1, 2, 6, 8, 14]) +
            _make_index(5, [0, 1]) +
            _make_index(6, [0, 1, 2, 3, 6, 7, 8, 9, 14]) +
            _make_index(7, [0, 1, 2, 3, 6, 7, 8, 9, 14])),
        'all_7': sum((_make_index(i, list(range(12)) + [12, 14]) for i in range(7)), []),
        'easy_robustness': _make_index(1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]),
    }

    def __init__(self,
                 muons_file=os.path.join(PROJECTS_DIR, 'MuonsAndMatter/data/muons/full_sample.h5'),
                 n_samples=0,
                 n_steps=5000,
                 sensitive_plane=[{'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 82},
                                  {'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 91}],
                 fSC_mag=False,
                 uniform_fields=False,
                 fields_file=None,
                 use_B_goal=True,
                 use_diluted=True,
                 cavern=True,
                 SND=False,
                 seed=None,
                 initial_phi=None,
                 dimensions_phi=43,
                 ):
        """n_samples = 0 uses every muon in muons_file. dimensions_phi is the number of free
        parameters (looked up in `parametrization`, else all of them) or an explicit
        list of (magnet, column)."""
        self.muons_file = muons_file
        self.n_samples = n_samples
        self.sensitive_plane = sensitive_plane
        self.fSC_mag = fSC_mag
        self.uniform_fields = uniform_fields
        self.fields_file = fields_file
        self.use_B_goal = use_B_goal
        self.use_diluted = use_diluted
        self.cavern = cavern
        self.SND = SND
        self.seed = seed
        # enough 2 cm steps to reach the first sensitive plane
        self.n_steps = max(n_steps, int(np.ceil(sensitive_plane[0]['position'] / 0.02)) + 100)

        if initial_phi is not None:
            self.DEFAULT_PHI = torch.as_tensor(initial_phi).view(-1, self.n_params)
        self.DEFAULT_PHI = self.DEFAULT_PHI.clone()
        self.n_magnets = len(self.DEFAULT_PHI)

        if isinstance(dimensions_phi, list):
            self.params_idx = torch.tensor(dimensions_phi)
        else:
            match = [v for v in self.parametrization.values() if len(v) == dimensions_phi]
            every = sum((_make_index(i, list(range(self.n_params))) for i in range(self.n_magnets)), [])
            self.params_idx = torch.tensor(match[0] if match else every)
        self.muons = None           # loaded on the first sample_x()

    @property
    def initial_phi(self):
        """The free parameters of DEFAULT_PHI, flattened."""
        return _apply_index(self.DEFAULT_PHI, self.params_idx).flatten()

    # ---- muons: columns px, py, pz, x, y, z, pdg[, weight] -----------------
    def _load_muons(self, muons_file):
        print(f'Loading muons from {muons_file}...')
        if muons_file.endswith('.npy'):
            x = np.load(muons_file, mmap_mode='r').copy()
        else:
            with h5py.File(muons_file, 'r') as f:
                x = np.stack([np.array(f[k], dtype=np.float32)
                              for k in ['px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight']], axis=1)
        x[..., 3:5] = 0.0           # every muon starts on axis, at z = -1
        x[..., 5] = -1.0
        return torch.from_numpy(x)

    def sample_x(self):
        """n_samples random muons from muons_file (all of them if n_samples is 0)."""
        if self.muons is None:
            self.muons = self._load_muons(self.muons_file)
        if 0 < self.n_samples < self.muons.size(0):
            return self.muons[torch.randperm(self.muons.size(0))[:self.n_samples]].clone()
        return self.muons

    def _muons_from_pz_pt(self, pz, pt):
        """Muons on axis at z = -1 with random azimuth and charge."""
        n = len(pz)
        azimuth = torch.rand(n) * 2 * np.pi
        charge = torch.randint(0, 2, (n,)) * 2 - 1
        zero = torch.zeros_like(pz)
        return torch.stack([pt * torch.cos(azimuth), pt * torch.sin(azimuth), pz,
                            zero, zero, zero - 1, (-self.MUON * charge).to(pz.dtype)], dim=1)

    def sample_uniform(self, n_samples=None):
        """pt ~ U(0, 5), pz ~ U(10, 380)."""
        n = self.n_samples if n_samples is None else n_samples
        pt = torch.rand(n) * 5.0
        pz = torch.rand(n) * 370.0 + 10.0
        return self._muons_from_pz_pt(pz, pt)

    def _load_flow(self, model='flow_model_complete_old_sample.pt'):
        """The (pz, pt) Neural Spline Flow from generator_muons/outputs."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'generator_muons_models', os.path.join(GENERATOR_DIR, 'models.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ckpt = torch.load(os.path.join(GENERATOR_DIR, 'outputs', model),
                          map_location='cpu', weights_only=False)
        entry = next(iter(ckpt['groups'].values()))   # single combined flow
        flow = mod.NeuralSplineFlow.from_config(entry['config'])
        flow.load_state_dict(entry['state_dict'])
        self._flow = flow.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        self._flow_mean = np.asarray(entry['mean'], dtype=np.float32).ravel()
        self._flow_std = np.asarray(entry['std'], dtype=np.float32).ravel()

    def _sample_pz_pt(self, n, temperature, batch=2_000_000):
        """n flow samples of (pz, pt) inside the kinematic envelope, by rejection."""
        parts, got = [], 0
        while got < n:
            s = self._flow.sample(min(batch, n - got), temperature=temperature,
                                  mean=self._flow_mean, std=self._flow_std).cpu()
            s[:, 1].clamp_(min=0.0)
            s = s[(s.pow(2).sum(1) <= self.P_MAX ** 2)
                  & (s[:, 1] <= self.PT_MAX)
                  & (s[:, 0] > self.PZ_MIN)]
            parts.append(s)
            got += s.shape[0]
        return torch.cat(parts)

    def sample_gen(self, n_samples=None, temperature=1.5, model='flow_model_complete_old_sample.pt'):
        """Muons from the generative flow, base normal scaled by `temperature`."""
        if not hasattr(self, '_flow'):
            self._load_flow(model)
        n = self.n_samples if n_samples is None else n_samples
        s = self._sample_pz_pt(n, temperature)
        return self._muons_from_pz_pt(s[:, 0], s[:, 1])

    # ---- simulation ---------------------------------------------------------
    def add_fixed_params(self, phi):
        """Free parameters -> the full (n_magnets, 15) design, with the tied columns filled."""
        if phi.numel() == self.n_magnets * self.n_params:
            return phi.view(self.n_magnets, self.n_params)
        new_phi = self.DEFAULT_PHI.clone().to(phi.device)
        new_phi[self.params_idx[:, 0], self.params_idx[:, 1]] = phi
        new_phi[0, 3] = new_phi[0, 2]           # magnet 0: dXOut = dXIn, dYOut = dYIn
        new_phi[0, 5] = new_phi[0, 4]
        new_phi[:, 13] = new_phi[:, 12]         # XmgapOut = XmgapIn
        if self.use_diluted:                    # magnets 0-3 are rectangular
            new_phi[:4, 3] = new_phi[:4, 2]
            new_phi[:4, 7] = new_phi[:4, 6]
            new_phi[:4, 9] = new_phi[:4, 8]
        return new_phi

    def simulate(self, phi, muons=None, return_all=False):
        from cuda_muons_ship import run_from_params
        phi = self.add_fixed_params(phi).detach().cpu()
        if muons is None:
            muons = self.sample_x()
        return run_from_params(
            phi.numpy(),
            muons,
            sensitive_plane=self.sensitive_plane,
            n_steps=self.n_steps,
            fSC_mag=self.fSC_mag,
            simulate_fields=not self.uniform_fields,
            field_map_file=self.fields_file,
            NI_from_B=self.use_B_goal,
            use_diluted=self.use_diluted,
            add_cavern=self.cavern,
            SND=self.SND,
            return_all=return_all,
            histogram_dir=os.path.join(PROJECTS_DIR, 'MuonsAndMatter/cuda_muons/data'),
            seed=self.seed,
        )

    def is_hit(self, x, y, z):
        plane = self.sensitive_plane[-1]
        mask = ((torch.abs(x) <= plane['dx'] / 2) & (torch.abs(y) <= plane['dy'] / 2)
                & (torch.abs(z - plane['position']) <= plane['dz']))
        return mask.int()

    def __call__(self, phi, muons=None):
        """Per-muon hit flag (0/1) on the last sensitive plane."""
        output = self.simulate(phi, muons=muons, return_all=True)
        return self.is_hit(output['x'], output['y'], output['z'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fields_map', dest='uniform_fields', action='store_false',
                        help='simulate the field map instead of uniform fields')
    parser.add_argument('--save_data', action='store_true', help='save phi, muons, hits to outputs/')
    parser.add_argument('--uniform_muons', action='store_true')
    parser.add_argument('--n_samples', type=int, default=0, help='0 = all (50M if --uniform_muons)')
    args = parser.parse_args()

    n_samples = int(50e6) if (args.uniform_muons and args.n_samples == 0) else args.n_samples
    shield = ShipMuonShieldCuda(uniform_fields=args.uniform_fields, n_samples=n_samples)
    phi = shield.initial_phi
    muons = shield.sample_uniform() if args.uniform_muons else shield.sample_x()
    hits = shield(phi, muons)
    print(f'NUMBER OF HITS: {hits.sum().item()}')
    if args.save_data:
        os.makedirs('outputs', exist_ok=True)
        tag = '_uniform' if args.uniform_muons else ''
        np.save(f'outputs/phis{tag}.npy', phi.numpy().astype(np.float32)[None])
        np.save(f'outputs/muons{tag}.npy', muons.numpy().astype(np.float32)[None][..., :7])
        np.save(f'outputs/hits{tag}.npy', hits.numpy().astype(np.float32)[None])
        print(f'saved outputs/{{phis,muons,hits}}{tag}.npy')
