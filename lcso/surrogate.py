"""The LCSO surrogate of the muon shield as the analysis uses it: p(hit | phi, x) for a design phi
and a muon x,

    logit(phi, x) = <branch(u), trunk(x)> + bias,   u = phi in the [-1, 1] tolerance box,

where u = 0 is the nominal design and u_i = +-1 puts parameter i at +-delta of its nominal
value. The trunk sees (px, py, pz, charge): the flow muons all start on axis at z = -1.

  LCSONet         the DeepONet; stack() turns an ensemble of them into one LCSONet.
                  Checkpoints of the earlier 7-input version load too.
  ModelSurrogate  any network of lcso.models (e.g. a Taylor surrogate, 'taylor-res4:8'), or an
                  ensemble of them, behind the same interface.

The analysis uses: dim, bias, branch_net(u), features(raw muons), denormalize_phi(u).
"""
import os

import torch

from .models import Ensemble, make


def _mlp(in_dim, hidden, out_dim, act):
    return torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), act(),
                               torch.nn.Linear(hidden, hidden), act(),
                               torch.nn.Linear(hidden, out_dim))


def muon_inputs(x):
    """(px, py, pz, charge) of raw muons (..., 7): px, py, pz, x, y, z, pdg (mu+ = -13)."""
    return torch.cat([x[..., :3], -torch.sign(x[..., 6:7])], -1)


class _Box(torch.nn.Module):
    """The tolerance box, phi <-> u, and the forward pass shared by both wrappers."""

    def _box(self, phi_0, delta):
        phi_0 = torch.as_tensor(phi_0, dtype=torch.float32).view(-1)
        self.dim, self.delta = phi_0.numel(), delta
        # buffers: they follow .to() and are saved with the weights
        self.register_buffer('lower_bound', phi_0 * (1 - delta))
        self.register_buffer('upper_bound', phi_0 * (1 + delta))

    def normalize_phi(self, phi):
        return (phi - self.lower_bound) / (self.upper_bound - self.lower_bound) * 2 - 1

    def denormalize_phi(self, u):
        return (u + 1) / 2 * (self.upper_bound - self.lower_bound) + self.lower_bound

    def features(self, x):
        """Trunk features of raw muons (..., 7)."""
        return self.trunk4(muon_inputs(x))

    def forward(self, phi, x):
        """Logits (B, N) of PHYSICAL designs phi (B, d) and raw muons x (B, N, 7) or (N, 7)."""
        b = self.branch_net(self.normalize_phi(phi).view(-1, self.dim))
        t = self.features(x)
        return (torch.einsum('bp,np->bn', b, t) if t.dim() == 2
                else torch.einsum('bp,bnp->bn', b, t)) + self.bias


class LCSONet(_Box):
    def __init__(self, phi_0, hidden=64, p=16, delta=0.1):
        super().__init__()
        self._box(phi_0, delta)
        self.hp = {'hidden': hidden, 'p': p}
        self.branch_net = _mlp(self.dim, hidden, p, torch.nn.SiLU)
        self.trunk_net = _mlp(4, hidden, p, torch.nn.ReLU)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('x_loc', torch.zeros(4))
        self.register_buffer('x_scale', torch.ones(4))

    @property
    def P(self):
        return self.hp['p']

    def trunk4(self, x4):
        """Trunk features of (px, py, pz, charge) rows."""
        return self.trunk_net((x4 - self.x_loc) / self.x_scale)


class ModelSurrogate(_Box):
    """A network of lcso.models (or an Ensemble of them) on (px, py, pz, charge), in the box."""

    def __init__(self, net, phi_0, delta=0.1, hp=None):
        super().__init__()
        self._box(phi_0, delta)
        self.net, self.hp = net, dict(hp or {})
        self.bias = net.bias

    @property
    def P(self):
        return self.net.P

    def branch_net(self, u):
        return self.net.branch(u)

    def trunk4(self, x4):
        return self.net.trunk(x4)


@torch.no_grad()
def stack(models):
    """One LCSONet whose logit is the mean of the models' logits (an ensemble): the branch and
    trunk features are concatenated, with block-diagonal layers and each trunk's input
    standardization folded into its first layer."""
    K, m0 = len(models), models[0]
    out = LCSONet(torch.zeros(m0.dim), hidden=K * m0.hp['hidden'], p=K * m0.hp['p'], delta=m0.delta)
    out.lower_bound.copy_(m0.lower_bound)
    out.upper_bound.copy_(m0.upper_bound)
    for name, scale in (('branch_net', 1 / K), ('trunk_net', 1)):        # 1/K: the mean of the K logits
        net, srcs = getattr(out, name), [getattr(m, name) for m in models]
        W0, b0 = [s[0].weight for s in srcs], [s[0].bias for s in srcs]
        if name == 'trunk_net':                                         # (x - loc) / scale, folded in
            W0 = [W / m.x_scale for W, m in zip(W0, models)]
            b0 = [b - W @ m.x_loc for W, b, m in zip(W0, b0, models)]
        net[0].weight.copy_(torch.cat(W0))
        net[0].bias.copy_(torch.cat(b0))
        for i, c in ((2, 1), (4, scale)):
            net[i].weight.copy_(torch.block_diag(*[s[i].weight for s in srcs]) * c)
            net[i].bias.copy_(torch.cat([s[i].bias for s in srcs]) * c)
    out.x_loc.zero_()
    out.x_scale.fill_(1.0)
    out.bias.copy_(torch.stack([m.bias for m in models]).mean(0))
    return out


def save_surrogate(model, path, **info):
    """An LCSONet, or a ModelSurrogate (its members' weights: one network, or each of an ensemble)."""
    config = {'delta': model.delta, **model.hp}
    if isinstance(model, ModelSurrogate):
        nets = list(model.net.models) if isinstance(model.net, Ensemble) else [model.net]
        torch.save({'config': config, 'members': [n.state_dict() for n in nets],
                    'box': [model.lower_bound, model.upper_bound], **info}, path)
    else:
        torch.save({'config': config, 'state_dict': model.state_dict(), **info}, path)


def _from_7_inputs(state):
    """A checkpoint of the earlier trunk on [(px, py, pz, x, y, z - mean) / std, -sign(pdg)]: the
    positions are constant for the flow muons, so fold their mean and std into the first layer."""
    W, b = state['trunk_net.0.weight'], state['trunk_net.0.bias']
    mean, std = state.pop('muon_mean'), state.pop('muon_std')
    z = torch.tensor([0.0, 0.0, -1.0])                                   # x, y, z of every muon
    b = b + W[:, 3:6] @ ((z - mean[3:6]) / std[3:6])
    state['trunk_net.0.weight'] = torch.cat([W[:, :3], W[:, 6:7]], 1)
    state['trunk_net.0.bias'] = b
    state['x_loc'] = torch.cat([mean[:3], torch.zeros(1)])
    state['x_scale'] = torch.cat([std[:3], torch.ones(1)])
    return state


def _member_nets(obj, D):
    """The lcso.models networks of a ModelSurrogate checkpoint."""
    c = obj['config']
    nets = []
    for sd in obj['members']:
        net = make(c['kind'], D, 4, p=c['p'], hidden=c['hidden'], p_res=c.get('p_res', 16))
        net.load_state_dict(sd)
        nets.append(net.eval())
    return nets


def load_surrogate(path, phi_0, device='cpu'):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if 'members' in obj:                                 # a ModelSurrogate
        nets = _member_nets(obj, len(phi_0))
        net = nets[0] if len(nets) == 1 else Ensemble(nets)
        return ModelSurrogate(net, phi_0, obj['config']['delta'], obj['config']).eval().to(device)
    state = obj['state_dict']
    if 'muon_mean' in state:
        state = _from_7_inputs(state)
    model = LCSONet(phi_0, **{k: obj['config'][k] for k in ('hidden', 'p', 'delta')})
    model.load_state_dict(state)
    return model.eval().to(device)


def load_members(path, phi_0, device='cpu'):
    """The networks an ensemble checkpoint averages, each as a surrogate; [] for one network."""
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if 'members' in obj:
        nets = _member_nets(obj, len(phi_0))
        return ([ModelSurrogate(n, phi_0, obj['config']['delta'], obj['config']).eval().to(device) for n in nets]
                if len(nets) > 1 else [])
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))   # member paths are stored relative to the repo root
    return [load_surrogate(p if os.path.exists(p) else os.path.join(root, p), phi_0, device)
            for p in obj.get('stacked', [])]
