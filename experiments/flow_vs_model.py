"""Why did M2:Z_len (and others) have a large gradient in the earlier analysis and not now?
Two things changed: the surrogate and the muon flow. Separate them: the gradient at u = 0 of
each surrogate (earlier, final) integrated over each flow (earlier, current) on the quadrature
grid, total and per p_z band.

    python experiments/flow_vs_model.py --device cpu

Run in the first analysis (43 parameters, muons on axis, as here).
"""
import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

from analysis.core import branch_jacobian, derivatives, nominal_pass
import cli
from lcso.surrogate import load_surrogate
from muons import FlowMuons
from paths import EXPERIMENTS, GENERATOR_DIR, ROOT
from shield import MuonShield

BANDS = torch.tensor([30.0, 75.0, 175.0])          # <30 | 30-75 | 75-175 | >175 GeV


def load_flow(path, device):
    """The momentum part of a generator_muons flow checkpoint, in the shape FlowMuons uses
    (flow, mean, std, m, mu_plus). Muons enter on axis, so the position part is not needed, which
    also lets older checkpoints in (their position parametrization differs)."""
    sys.path.insert(0, GENERATOR_DIR)
    try:
        flows = importlib.import_module('src.flows')
    finally:
        sys.path.remove(GENERATOR_DIR)
    flow, m = flows.load_flow(path, device)
    return SimpleNamespace(flow=flow, m=m, mean=m['mean'], std=m['std'], mu_plus=m['mu_plus_fraction'])


def band(x):
    return torch.bucketize(x[:, 2].contiguous(), BANDS.to(x))


if __name__ == '__main__':
    parser = cli.parser(__doc__)
    parser.add_argument('--models', default='outputs/lcso_deeponet.pt,outputs/lcso.pt')
    parser.add_argument('--flows', default='outputs/flow_2026.pt,outputs/muon_flow.pt', help='in generator_muons')
    parser.add_argument('--params', default='M2:Z_len,M3:Z_len,M4:Z_len,M7:Z_len,M7:NI,M2:dXIn,M7:dXOut,M6:Z_len')
    parser.add_argument('--grid', default='201x32')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(32)
    shield = MuonShield(FlowMuons(on_axis=True))
    labels = shield.labels
    ks = [labels.index(p) for p in args.params.split(',')]
    nz, nphi = map(int, args.grid.split('x'))
    lines = [f'gradient at u = 0 per tolerance unit, g / H_0 (quadrature grid {args.grid}, nominal flux); '
             'bands: p_z < 30 | 30-75 | 75-175 | > 175 GeV', '']
    for fpath in args.flows.split(','):
        try:
            shield.muons.gen = load_flow(os.path.join(GENERATOR_DIR, fpath), args.device)
            shield.muons.device = torch.device(args.device)
        except Exception as e:                     # an older flow the current generator code cannot read
            lines.append(f'flow {fpath}: cannot be loaded ({e})')
            continue
        for mpath in args.models.split(','):
            model = load_surrogate(os.path.join(ROOT, mpath), shield.phi_0, args.device)
            acc = nominal_pass(model, shield.muons.grid(nz, nphi), band, 4)
            J = branch_jacobian(model)
            gb = np.array([derivatives(model, J, acc.T[b], acc.A[b])[0] for b in range(4)])   # (bands, D)
            H, Hb = acc.H.sum(), acc.H
            lines.append(f'flow {fpath}, surrogate {mpath}: H_0 = {H:.1f} per 1e9; hits per band '
                         + ' | '.join(f'{h / H:.0%}' for h in Hb))
            lines.append(f'  {"parameter":16s} {"g/H_0":>8s}   per band (dH_b/du / H_0)')
            for k in ks:
                lines.append(f'  {labels[k]:16s} {gb[:, k].sum() / H:+8.3f}   ' + ' '.join(f'{v / H:+7.3f}' for v in gb[:, k]))
            lines.append('')
    out = os.path.join(EXPERIMENTS, 'figs', 'flow_vs_model.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
