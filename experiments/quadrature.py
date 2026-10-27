"""Integrating the surrogate's expected hits over the muons: sampling vs. the quadrature grid.

A muon is its momentum (pz, pt), azimuth phi, charge q and production point. In the flow the first
three are independent: (pz, pt) is the image of a 2-d standard normal latent under the momentum
flow, phi is uniform and q is +-1 with the data's mu+ fraction; the production point is drawn from
the flow's position part given (pz, pt, q). So

    H(u) / N = E_z,phi,q E_pos|z,q [ A p(hit | u, x) ] / E_z[ A ]      (A: the kinematic envelope)

and two estimators of it (and of its gradient and Hessian at u = 0) are compared:
  mc     plain Monte Carlo: sampled muons
  grid   muons.FlowMuons.grid: a tensor grid in the latent on [-L, L]^2 with Gaussian weights, phi midpoints,
         both charges, and n_pos production points drawn per node from a fixed seed; the draws are
         the grid's only noise, measured here over seeds and against n_pos
The reference is the finest grid with several production points per node (--ref).

    python experiments/quadrature.py            # -> outputs/{figures,results}/experiments/quadrature.{png,txt,json}
    python experiments/quadrature.py --replot   # the table and the figure again, from the .json
"""
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.core import branch_jacobian, derivatives
import cli
from lcso.surrogate import load_surrogate
from paths import EXPERIMENTS, figures, save_figure
from shield import MuonShield

FIGS = figures(EXPERIMENTS)


@torch.no_grad()
def accumulate(model, x, w, acc):
    """Add the weighted H, T = sum w s t and A = sum w s (1 - 2s) t t' (s = p(1 - p)) of muons x."""
    t = model.features(x)
    b0 = model.branch_net(torch.zeros(1, model.dim, device=x.device))[0]
    p = torch.sigmoid(t @ b0 + model.bias)
    s = p * (1 - p) * w.float()
    acc['H'] += float((p.double() * w).sum())
    acc['T'] += (s[:, None] * t).double().sum(0)
    acc['A'] += ((t * (s * (1 - 2 * p))[:, None]).T @ t).double()
    acc['W'] += float(w.sum())
    acc['n'] += len(x)


def estimate(model, J, points, n_muons=1e9):
    """(H, grad, hess) per n_muons muons, and the number of points, from batches of (x, weight)."""
    P = model.branch_net[-1].out_features
    dev = model.bias.device
    acc = dict(H=0.0, W=0.0, n=0, T=torch.zeros(P, dtype=torch.float64, device=dev),
               A=torch.zeros(P, P, dtype=torch.float64, device=dev))
    for x, w in points:
        accumulate(model, x, w, acc)
    g, h = derivatives(model, J, acc['T'] / acc['W'], acc['A'] / acc['W'])
    return acc['H'] / acc['W'] * n_muons, g * n_muons, h * n_muons, acc['n']


def mc_points(shield, n, seed, batch=2 ** 21):
    """Plain Monte Carlo: sampled muons (with their production points), weight 1."""
    torch.manual_seed(seed)
    for c in range(0, n, batch):
        x = shield.sample(min(batch, n - c), device='cuda')
        yield x, torch.ones(len(x), dtype=torch.float64, device=x.device)


def grid_points(shield, spec, n_pos=1, seed=0):
    """The grid 'n_z x n_phi', n_pos production points per node drawn from `seed`."""
    return shield.muons.points(f'{spec}:{n_pos}', n_ref=1.0, seed=seed)


def errors(rs, ref):
    """rms over runs rs of the relative errors of H, grad and Hessian against ref."""
    H0, g0, h0 = ref['H'], np.array(ref['g']), np.array(ref['h'])
    return (np.sqrt(np.mean([(r['H'] / H0 - 1) ** 2 for r in rs])),
            np.sqrt(np.mean([np.sum((np.array(r['g']) - g0) ** 2) for r in rs])) / np.linalg.norm(g0),
            np.sqrt(np.mean([np.sum((np.array(r['h']) - h0) ** 2) for r in rs])) / np.linalg.norm(h0))


def report(out, name):
    """The table (results/experiments/<name>.txt) and the figure of a finished comparison."""
    ref, ref_spec = out['ref'], out.get('ref_spec', '1601x64:4')
    first_noise = next(iter(out['noise'])).split(':')[0]
    lines = [f'model {out["model"]}; reference: grid {ref_spec} (n_z x n_phi : production points per node), '
             f'H0 = {ref["H"]:.4f} hits per 1e9 muons',
             'errors: relative to the reference; rms over the repeats (Monte Carlo seeds, or position seeds)',
             f'{"estimator":34s} {"points":>9s} {"H rel. err":>11s} {"grad rel. err":>14s} {"Hess rel. err":>14s}']
    curves = {'mc': [], 'grid': [], 'noise': []}
    for size, rs in out['mc'].items():
        e = errors(rs, ref)
        curves['mc'].append((rs[0]['points'], *e))
        lines.append(f'{"mc (" + str(len(rs)) + " seeds)":34s} {rs[0]["points"]:9.1e} ' + ' '.join(f'{v:{w}.2e}' for v, w in zip(e, (11, 14, 14))))
    for spec, r in out['grid'].items():
        e = errors([r], ref)
        curves['grid'].append((r['points'], *e))
        lines.append(f'{"grid " + spec + ", 1 position":34s} {r["points"]:9.1e} ' + ' '.join(f'{v:{w}.2e}' for v, w in zip(e, (11, 14, 14))))
    for key, rs in out['noise'].items():
        e = errors(rs, ref)
        spec, k = key.split(':')
        curves['noise'].append((rs[0]['points'], *e, key))
        lines.append(f'{"grid " + spec + ", " + k + " position" + "s" * (k != "1") + " (" + str(len(rs)) + " seeds)":34s} {rs[0]["points"]:9.1e} '
                     + ' '.join(f'{v:{w}.2e}' for v, w in zip(e, (11, 14, 14))))
    with open(os.path.join(EXPERIMENTS, f'{name}.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for i, title in enumerate(('expected hits H', 'gradient at u = 0', 'Hessian at u = 0')):
        ax = axes[i]
        for kind, color, label in (('mc', '#2a78d6', 'Monte Carlo'), ('grid', '#1baf7a', 'grid, one position per node')):
            pts = sorted(curves[kind])
            ax.plot([p[0] for p in pts], [p[i + 1] for p in pts], '-o', color=color, lw=2, ms=4, label=label)
        for j, p in enumerate(curves['noise']):
            spec, k = p[4].split(':')
            ax.plot(p[0], p[i + 1], 's' if k == '1' else 'D', ms=7, mfc='none' if k == '1' else '#1baf7a', mec='#0d6e4b',
                    label=f'grid, {k} position{"s" if k != "1" else ""} per node (rms over seeds)' if spec == first_noise else None)
        ax.set(xscale='log', yscale='log', xlabel='points (muons) per estimate', title=title)
    axes[0].set_ylabel('relative error vs. the reference grid')
    axes[0].legend(fontsize=7.5)
    fig.suptitle(f'Integrating the surrogate over the muon flow: sampling vs. the grid (reference: grid {ref_spec})')
    save_figure(fig, FIGS, name, report_name=name)
    print(f'saved {name}.png, {name}.txt')


if __name__ == '__main__':
    parser = cli.add_common(cli.parser(__doc__))
    parser.add_argument('--sizes', default='1e6,1e7,1e8', help='muons per Monte Carlo estimate')
    parser.add_argument('--reps', type=int, default=8, help='Monte Carlo repeats per size, and position seeds per grid')
    parser.add_argument('--grids', default='101x8,201x16,201x64,401x32,801x64', help='n_z x n_phi, one production point per node')
    parser.add_argument('--noise_grids', default='201x64,801x64', help='grids whose position draws are repeated over seeds')
    parser.add_argument('--n_pos', default='1,4', help='production points per node of --noise_grids')
    parser.add_argument('--ref', default='1601x64:4', help='reference grid n_z x n_phi : n_pos')
    parser.add_argument('--out', default='quadrature')
    parser.add_argument('--replot', action='store_true', help='only the table and the figure, from <out>.json')
    args = parser.parse_args()
    if args.replot:
        with open(os.path.join(EXPERIMENTS, f'{args.out}.json')) as f:
            report(json.load(f), args.out)
        raise SystemExit
    torch.backends.cuda.matmul.allow_tf32 = False
    shield = MuonShield(args.muons)
    model = load_surrogate(args.model, shield.phi_0, args.device)
    J = branch_jacobian(model)
    out = dict(model=args.model, ref_spec=args.ref, mc={}, grid={}, noise={})

    def run(points):
        t0 = time.time()
        H, g, h, n = estimate(model, J, points)
        return dict(H=H, g=g.tolist(), h=h.tolist(), points=n, seconds=time.time() - t0)

    spec, n_pos = args.ref.split(':')
    out['ref'] = ref = run(grid_points(shield, spec, int(n_pos), seed=1000))
    print(f'reference grid {args.ref} ({ref["points"]:.1e} points): H = {ref["H"]:.3f} ({ref["seconds"]:.0f} s)', flush=True)
    for spec in args.grids.split(','):
        out['grid'][spec] = r = run(grid_points(shield, spec))
        print(f'grid {spec:9s} ({r["points"]:.1e} points): H = {r["H"]:.3f} ({r["seconds"]:.0f} s)', flush=True)
    for spec in args.noise_grids.split(','):
        for k in map(int, args.n_pos.split(',')):
            rs = out['noise'][f'{spec}:{k}'] = [run(grid_points(shield, spec, k, seed)) for seed in range(args.reps)]
            Hs = [r['H'] for r in rs]
            print(f'grid {spec}, {k} position{"s" * (k != 1)} per node: H = {np.mean(Hs):.3f} +- {np.std(Hs, ddof=1):.3f} '
                  f'(sd over {args.reps} position seeds)', flush=True)
    for size in args.sizes.split(','):
        rs = out['mc'][size] = [run(mc_points(shield, int(float(size)), seed)) for seed in range(args.reps)]
        Hs = [r['H'] for r in rs]
        print(f'mc n = {float(size):.0e}: H = {np.mean(Hs):.3f} +- {np.std(Hs, ddof=1):.3f} (sd over {args.reps})', flush=True)
    os.makedirs(EXPERIMENTS, exist_ok=True)
    with open(os.path.join(EXPERIMENTS, f'{args.out}.json'), 'w') as f:
        json.dump(out, f)
    report(out, args.out)
