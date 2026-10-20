"""The LCSO surrogate of the muon shield (DeepONet): p(hit | phi, x) for a design phi and a muon x.

    logit(phi, x) = <branch(u), trunk(x)> + bias,   u = phi in the [-1, 1] tolerance box

u = 0 is the nominal design and u_i = +-1 puts parameter i at +-delta of its nominal value.
The trunk sees (px, py, pz, charge), standardized (the flow muons all start on axis at z = -1).
train_surrogate.py fits it (stack() makes one LCSONet of an ensemble); checkpoints of the
earlier 7-input version load too.
"""
import torch
from scipy.stats import qmc


def _mlp(in_dim, hidden, out_dim, act):
    return torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), act(),
                               torch.nn.Linear(hidden, hidden), act(),
                               torch.nn.Linear(hidden, out_dim))


def muon_inputs(x):
    """(px, py, pz, charge) of raw muons (..., 7): px, py, pz, x, y, z, pdg (mu+ = -13)."""
    return torch.cat([x[..., :3], -torch.sign(x[..., 6:7])], -1)


class LCSONet(torch.nn.Module):
    def __init__(self, phi_0, hidden=64, p=16, delta=0.1):
        super().__init__()
        phi_0 = torch.as_tensor(phi_0, dtype=torch.float32).view(-1)
        self.dim, self.delta, self.hp = phi_0.numel(), delta, {'hidden': hidden, 'p': p}
        self.branch_net = _mlp(self.dim, hidden, p, torch.nn.SiLU)
        self.trunk_net = _mlp(4, hidden, p, torch.nn.ReLU)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        # buffers: they follow .to() and are saved with the weights
        self.register_buffer('lower_bound', phi_0 * (1 - delta))
        self.register_buffer('upper_bound', phi_0 * (1 + delta))
        self.register_buffer('x_loc', torch.zeros(4))
        self.register_buffer('x_scale', torch.ones(4))

    def normalize_phi(self, phi):
        return (phi - self.lower_bound) / (self.upper_bound - self.lower_bound) * 2 - 1

    def denormalize_phi(self, u):
        return (u + 1) / 2 * (self.upper_bound - self.lower_bound) + self.lower_bound

    def sample_u(self, n, seed=None):
        """Scrambled Sobol designs in the normalized box."""
        u = qmc.Sobol(d=self.dim, scramble=True, seed=seed).random(n) * 2 - 1
        return torch.as_tensor(u, dtype=torch.float32)

    def features(self, x):
        """Trunk features of raw muons (..., 7)."""
        return self.trunk_net((muon_inputs(x) - self.x_loc) / self.x_scale)

    def forward(self, phi, x):
        """Logits (B, N) of PHYSICAL designs phi (B, d) and raw muons x (B, N, 7) or (N, 7)."""
        b = self.branch_net(self.normalize_phi(phi).view(-1, self.dim))
        t = self.features(x)
        return (torch.einsum('bp,np->bn', b, t) if t.dim() == 2
                else torch.einsum('bp,bnp->bn', b, t)) + self.bias


@torch.no_grad()
def stack(models):
    """One LCSONet whose logit is the mean of the models' logits (an ensemble): the branch and
    trunk features are concatenated, with block-diagonal layers and each trunk's input
    standardization folded into its first layer."""
    K, m0 = len(models), models[0]
    h, p = m0.hp['hidden'], m0.hp['p']
    out = LCSONet(torch.zeros(m0.dim), hidden=K * h, p=K * p, delta=m0.delta)
    out.lower_bound.copy_(m0.lower_bound)
    out.upper_bound.copy_(m0.upper_bound)
    for k, m in enumerate(models):
        rows, cols = slice(k * h, (k + 1) * h), slice(k * p, (k + 1) * p)
        for net, src in ((out.branch_net, m.branch_net), (out.trunk_net, m.trunk_net)):
            W0, b0 = src[0].weight, src[0].bias
            if net is out.trunk_net:                                      # (x - loc) / scale, folded in
                W0, b0 = W0 / m.x_scale, b0 - (W0 / m.x_scale) @ m.x_loc
            net[0].weight[rows] = W0
            net[0].bias[rows] = b0
            net[2].weight[rows, rows] = src[2].weight
            net[2].bias[rows] = src[2].bias
            scale = 1 / K if net is out.branch_net else 1                 # the mean of the K logits
            net[4].weight[cols, rows] = src[4].weight * scale
            net[4].bias[cols] = src[4].bias * scale
    for net in (out.branch_net, out.trunk_net):                           # zero off the blocks
        for i in (0, 2, 4):
            mask = torch.zeros_like(net[i].weight)
            for k in range(K):
                r = slice(k * (p if i == 4 else h), (k + 1) * (p if i == 4 else h))
                c = slice(k * h, (k + 1) * h) if i else slice(None)
                mask[r, c] = 1
            net[i].weight.mul_(mask)
    out.x_loc.zero_()
    out.x_scale.fill_(1.0)
    out.bias.copy_(torch.stack([m.bias for m in models]).mean(0))
    return out


def save_surrogate(model, path, **info):
    torch.save({'config': {'delta': model.delta, **model.hp}, 'state_dict': model.state_dict(), **info}, path)


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


def load_surrogate(path, phi_0, device='cpu'):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    state = obj['state_dict']
    if 'muon_mean' in state:
        state = _from_7_inputs(state)
    model = LCSONet(phi_0, **{k: obj['config'][k] for k in ('hidden', 'p', 'delta')})
    model.load_state_dict(state)
    return model.eval().to(device)
