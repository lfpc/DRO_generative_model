from html import parser
import os
import h5py
import numpy as np
import torch
from utils import _make_index, _apply_index
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PROJECTS_DIR = os.getenv('PROJECTS_DIR', '~/projects')
GENERATOR_DIR = os.path.expanduser(os.path.join(PROJECTS_DIR, 'generator_muons'))


class ShipMuonShieldCuda:
    idx_mag = {0: 'Z_gap[cm]', 1: 'Z_len[cm]',
            2: 'dXIn[cm]', 3: 'dXOut[cm]',
            4: 'dYIn[cm]', 5: 'dYOut[cm]',
            6: 'gapIn[cm]', 7: 'gapOut[cm]',
            8: 'ratio_yokesIn', 9: 'ratio_yokesOut',
            10: 'dY_yokeIn[cm]', 11: 'dY_yokeOut[cm]',
            12: 'XmgapIn[cm]', 13: 'XmgapOut[cm]',
            14: 'NI[A]'}

    params = {
        'try_opt': [
            [0.0000, 115.0000, 40.0000, 40.0000, 119.0000, 119.0000, 61.5000, 61.5000, 61.5000, 61.5000, 50.0000, 50.0000, 0.0000, 0.0000, 1.9000],
            [15.0000, 151.5035, 63.1169, 63.1169, 10.0000, 10.0000, 8.1495, 9.1550, 64.1224, 63.1169, 69.4286, 69.4286, 0.0000, 0.0000, 1.9000],
            [15.0000, 224.9372, 68.0437, 68.0437, 10.0000, 10.0000, 8.4314, 8.5295, 70.6235, 70.5253, 74.8481, 74.8481, 0.0000, 0.0000, 1.9000],
            [16.1245, 225.0000, 64.8050, 64.8050, 13.9322, 13.9322, 8.9857, 9.1093, 85.8846, 85.7609, 71.2855, 71.2855, 0.0000, 0.0000, 1.9000],
            [15.0000, 223.9999, 39.0189, 39.0189, 11.0774, 11.0774, 8.3273, 8.9986, 97.5319, 96.8606, 46.2724, 46.2724, 0.0000, 0.0000, 1.9000],
            [15.0000, 120.3744, 20.0000, 20.0000, 30.0000, 30.0000, 0.0000, 0.0000, 102.7509, 102.7509, 22.0000, 22.0000, 0.0000, 0.0000, -0.0000],
            [15.0001, 168.0000, 20.0000, 30.0000, 40.0000, 40.0000, 24.9153, 9.3875, 97.1824, 102.7103, 33.0000, 33.0000, 0.0000, 0.0000, -1.2000],
            [15.0000, 193.0909, 37.9502, 67.8055, 47.7714, 47.7714, 8.0002, 8.0000, 124.7799, 94.9248, 74.5861, 74.5861, 0.0000, 0.0000, -1.9000]],
    }

    DEFAULT_PHI = torch.tensor(params['try_opt'])
    n_params = 15
    MUON = 13
    P_MAX = 400.0 
    PT_MAX = 14.0
    PZ_MIN = 1.0

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
        'easy_robustness': (
            _make_index(1,[0,1,2,3,4,5,6,7,8,9,10,14])
        )
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
                 validate_inputs=False,
                 **kwargs,
                 ):
        self.n_samples = n_samples
        self.validate_inputs = validate_inputs
        self.sensitive_plane = sensitive_plane
        self.fSC_mag = fSC_mag
        self.uniform_fields = uniform_fields
        self.fields_file = fields_file
        self.use_B_goal = use_B_goal
        self.use_diluted = use_diluted
        self.cavern = cavern
        self.SND = SND
        self.seed = seed
        self.n_steps = max(n_steps, int(np.ceil(sensitive_plane[0]['position'] / 0.02)) + 100)

        if initial_phi is not None:
            self.DEFAULT_PHI = torch.as_tensor(initial_phi).view(-1, self.n_params)
        self.DEFAULT_PHI = self.DEFAULT_PHI.clone()

        if isinstance(dimensions_phi, list):
            self.params_idx = torch.tensor(dimensions_phi)
        else:
            for indexes in self.parametrization.values():
                if len(indexes) == dimensions_phi:
                    self.params_idx = torch.tensor(indexes)
                    break
            else:
                self.params_idx = torch.tensor(
                    sum((_make_index(i, list(range(self.n_params))) for i in range(len(self.DEFAULT_PHI))), [])
                )

        self.n_magnets = len(self.DEFAULT_PHI)

        try: 
            from cuda_muons_ship import run_from_params
            self.run_muonshield = run_from_params
        except ImportError as e:
            print(f"Error importing cuda_muons_ship: {e}")
        self.muons_file = muons_file
        self.muons = None# self._load_muons(muons_file)

    def _load_muons(self, muons_file):
        print(f'Loading muons from {muons_file}...')
        if muons_file.endswith('.npy'):
            x = np.load(muons_file, mmap_mode='r').copy()
        elif muons_file.endswith('.h5'):
            with h5py.File(muons_file, 'r') as f:
                cols = [np.array(f[feat], dtype=np.float32)
                        for feat in ['px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight']]
            x = np.stack(cols, axis=1)
        else:
            raise ValueError(f'Unsupported muons file format: {muons_file}')
        # For now
        x[..., 3] = 0.0
        x[..., 4] = 0.0
        x[..., 5] = -1.0
        return torch.from_numpy(x)

    def sample_x(self, phi=None, idx=None):
        if self.muons is None:
            self.muons = self._load_muons(self.muons_file)
        if 0 < self.n_samples < self.muons.size(0):
            indices = torch.randperm(self.muons.size(0))[:self.n_samples]
            return self.muons[indices].clone()
        return self.muons
    def sample_uniform(self, n_samples=None):
        n = self.n_samples if n_samples is None else n_samples
        Pt_min, Pt_max = torch.tensor([0.0, 5.0], dtype=torch.float32)
        Pz_min, Pz_max = torch.tensor([10.0, 380.0], dtype=torch.float32)
        Pt_sample = torch.rand(n) * (Pt_max - Pt_min) + Pt_min
        Pz_sample = torch.rand(n) * (Pz_max - Pz_min) + Pz_min
        phi_sample = torch.rand(n) * 2 * np.pi
        px_sample = Pt_sample * torch.cos(phi_sample)
        py_sample = Pt_sample * torch.sin(phi_sample)
        x_sample = torch.zeros_like(px_sample)
        y_sample = torch.zeros_like(py_sample)
        z_sample = -1*torch.ones_like(px_sample)
        charge_sample = torch.randint(0, 2, (n,), dtype=torch.int32) * 2 - 1
        pdg_sample = charge_sample * (-1) * self.MUON
        return torch.stack([px_sample, py_sample, Pz_sample, x_sample, y_sample, z_sample, pdg_sample], dim=1)

    def _load_flow(self, model='flow_model_complete_old_sample.pt'):
        """Load the (pz, pt) Neural Spline Flow from ~/projects/generator_muons/outputs."""
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
        """Flow samples of (pz, pt), `batch` at a time so a huge n does not OOM, rejecting
        any muon outside the kinematic envelope (|p| > P_MAX, pt > PT_MAX, pz <= PZ_MIN)
        and redrawing until n valid ones are collected."""
        parts, got = [], 0
        while got < n:
            s = self._flow.sample(min(batch, n - got), temperature=temperature,
                                  mean=self._flow_mean, std=self._flow_std).cpu()
            s[:, 1].clamp_(min=0.0)                                   # pt >= 0
            s = s[(s.pow(2).sum(1) <= self.P_MAX ** 2)                # pz^2 + pt^2 = |p|^2
                  & (s[:, 1] <= self.PT_MAX)
                  & (s[:, 0] > self.PZ_MIN)]
            parts.append(s)
            got += s.shape[0]
        return torch.cat(parts)

    def sample_gen(self, n_samples=None, temperature=1.5, model='flow_model_complete_old_sample.pt'):
        """Sample muons from the generative flow trained in generator_muons, with the base
        normal scaled by `temperature`. flow.sample() (in eval mode) already denormalizes
        and undoes the log transform, so it returns physical (pz, pt)."""
        if not hasattr(self, '_flow'):
            self._load_flow(model)
        n = self.n_samples if n_samples is None else n_samples
        samples = self._sample_pz_pt(n, temperature)
        Pz_sample, Pt_sample = samples[:, 0], samples[:, 1]
        phi_sample = torch.rand(n) * 2 * np.pi
        px_sample = Pt_sample * torch.cos(phi_sample)
        py_sample = Pt_sample * torch.sin(phi_sample)
        x_sample = torch.zeros_like(px_sample)
        y_sample = torch.zeros_like(py_sample)
        z_sample = torch.zeros_like(px_sample) - 1
        charge_sample = torch.randint(0, 2, (n,)) * 2 - 1
        pdg_sample = (charge_sample * (-1) * self.MUON).to(px_sample.dtype)
        return torch.stack([px_sample, py_sample, Pz_sample, x_sample, y_sample, z_sample, pdg_sample], dim=1)

    def add_fixed_params(self, phi: torch.Tensor):
        if phi.numel() != (self.n_magnets * self.n_params):
            new_phi = self.DEFAULT_PHI.clone().to(phi.device)
            new_phi = new_phi.index_put((self.params_idx[:, 0], self.params_idx[:, 1]), phi)
            new_phi = new_phi.index_put((torch.tensor([0]), torch.tensor([3])), new_phi[0, 2])
            new_phi = new_phi.index_put((torch.tensor([0]), torch.tensor([5])), new_phi[0, 4])
            all_rows = torch.arange(new_phi.size(0))
            new_phi = new_phi.index_put((all_rows, torch.tensor(13)), new_phi[:, 12])
            if self.use_diluted:
                rect_rows = torch.tensor([0, 1, 2, 3])
                new_phi = new_phi.index_put((rect_rows, torch.tensor(3)), new_phi[rect_rows, 2])
                new_phi = new_phi.index_put((rect_rows, torch.tensor(7)), new_phi[rect_rows, 6])
                new_phi = new_phi.index_put((rect_rows, torch.tensor(9)), new_phi[rect_rows, 8])
        else:
            new_phi = phi
        return new_phi.view(self.n_magnets, self.n_params)

    def simulate(self, phi: torch.Tensor, muons=None, return_all=False):
        phi = self.add_fixed_params(phi).detach().cpu()
        if muons is None:
            muons = self.sample_x()
        self._sum_weights = muons[:, -1].sum()
        assert phi.shape[1] == 15, f'Expected phi to have 15 columns, got {phi.shape}'
        output = self.run_muonshield(
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
        return output
        px, py, pz = output['px'], output['py'], output['pz']
        x, y, z = output['x'], output['y'], output['z']
        particle = output['pdg_id']
        weight = output.get('weight', torch.ones_like(px))
        return torch.stack([px, py, pz, x, y, z, particle, weight]).T

    def is_hit(self,x, y, z):
        x_margin = self.sensitive_plane[-1]['dx'] / 2
        y_margin = self.sensitive_plane[-1]['dy'] / 2
        mask = (torch.abs(x) <= x_margin) & (torch.abs(y) <= y_margin) 
        mask = mask & (torch.abs(z - self.sensitive_plane[-1]['position']) <= self.sensitive_plane[-1]['dz'])
        return mask.int()
    @property
    def initial_phi(self):
        return _apply_index(self.DEFAULT_PHI, self.params_idx).flatten()
    def __call__(self, phi: torch.Tensor, muons=None):
        output = self.simulate(phi, muons=muons, return_all=True)
        return self.is_hit(output['x'], output['y'], output['z'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields_map', dest='uniform_fields', action='store_false', help='Use uniform magnetic fields')
    parser.add_argument('--save_data', action='store_true', help='Save simulation data')
    parser.add_argument('--uniform_muons', action='store_true', help='Use uniform muon distribution')
    parser.add_argument('--n_samples', type=int, default=0, help='Number of muons to simulate')
    args = parser.parse_args()
    n_samples = int(50e6) if (args.uniform_muons and args.n_samples == 0) else args.n_samples
    muon_shield = ShipMuonShieldCuda(uniform_fields=args.uniform_fields, n_samples=n_samples)
    phi = muon_shield.initial_phi
    if args.uniform_muons:
        muons = muon_shield.sample_uniform()
    else:
        muons = muon_shield.sample_x()
    hits = muon_shield(phi, muons)
    print(f'NUMBER OF HITS: {hits.sum().item()}')
    if args.save_data:
        os.makedirs('outputs', exist_ok=True)
        phi= phi.numpy().astype(np.float32, copy=False)[None]
        muons= muons.numpy().astype(np.float32, copy=False)[None][...,:7]
        hits= hits.numpy().astype(np.float32, copy=False)[None]
        tag = '_uniform' if args.uniform_muons else ''
        np.save(f'outputs/phis{tag}.npy', phi)
        np.save(f'outputs/muons{tag}.npy', muons)
        np.save(f'outputs/hits{tag}.npy', hits)
        print(f'Saved phi {phi.shape}, muons {muons.shape}, hits {hits.shape} '
              f'to outputs/{{phi,muons,hits}}{tag}.npy')
