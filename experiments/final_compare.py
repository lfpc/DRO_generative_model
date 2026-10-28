"""The production surrogate: candidate architectures compared at the same, final setup.

Each candidate is an ensemble checkpoint outputs/<name>.pt (final_compare.sh: lcso/train.py
on all pools, one --kind, several seeds, --stack). For each:
  scores        on the reference designs, simulated on the same muons (validation.validate.score:
                hit predictions on the nominal muons and the hot check sample, one-at-a-time
                secants and curvatures)
  members       on the quadrature grid: how far the ensemble's members
                agree on the gradient at u = 0 (cos) and on the top Hessian eigenvector (|cos|)
  vs. the first candidate   the cosines of the gradient, the top Hessian eigenvector and the
                worst direction (trust region, |u| = 1), and H/H_0 along the candidate's worst
                direction as predicted by the candidate and by the first one

    python experiments/final_compare.py --candidates lcso,tsub4,tsub8,tsubres4,taylor8

Run in the first analysis (43 parameters, muons on axis).
"""
import json
import os

import numpy as np
import torch

from metrics import cos, top_vec

from analysis.core import branch_jacobian, cached_hits, derivatives, feature_cache, nominal_pass, trust_region_max
import cli
from lcso.models import derivatives_autograd
from lcso.surrogate import load_members, load_surrogate, muon_inputs
from paths import EXPERIMENTS, figures, checkpoint
from shield import MuonShield
from validation.validate import REFS, Reference, as_study, score

FIGS = figures(EXPERIMENTS)


def one(x):
    return torch.zeros(len(x), dtype=torch.long)


def separable(m):
    return getattr(getattr(m, 'net', m), 'separable', True)


def x4_points(shield, grid):
    """The grid's points as the study networks' inputs (lcso.surrogate.muon_inputs)."""
    return ((muon_inputs(x), w) for x, w in shield.muons.points(grid))


def at_nominal(m, shield, grid):
    """(H0, grad, hess) of a production surrogate, integrated on the grid (autograd through the
    whole network if it is not separable)."""
    if not separable(m):
        H, g, h = derivatives_autograd(m.net, x4_points(shield, grid))
        return H, g.numpy(), h.numpy()
    acc = nominal_pass(m, shield.muons.points(grid), one, 1)
    g, h = derivatives(m, branch_jacobian(m), acc.T[0], acc.A[0])
    return float(acc.H[0]), g, h


@torch.no_grad()
def ratio(m, shield, grid, U):
    """H(u)/H(0) of designs U on the grid."""
    U = np.concatenate([np.zeros((1, m.dim)), U])
    if not separable(m):
        net = m.net
        Ut = torch.as_tensor(U, dtype=torch.float32, device=net.bias.device)
        h = sum((torch.sigmoid(net.logits(Ut, x[c:c + 2 ** 18])).double() * w[c:c + 2 ** 18]).sum(1)
                for x, w in x4_points(shield, grid) for c in range(0, len(x), 2 ** 18)).cpu().numpy()
    else:
        h = cached_hits(m, feature_cache(m, shield.muons.points(grid), one), 1, U)[:, 0]
    return h[1:] / h[0]


if __name__ == '__main__':
    parser = cli.add_common(cli.parser(__doc__), model=False)
    parser.add_argument('--candidates', default='lcso,tsub4,tsub8,tsubres4,taylor8',
                        help='checkpoints outputs/<name>.pt; the first is the baseline')
    parser.add_argument('--grid', default='401x64', help='muon points of the derivatives')
    parser.add_argument('--cache_grid', default='201x64', help='muon points of H/H_0')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    grid, cgrid = args.grid, args.cache_grid
    shield = MuonShield(args.muons, fields=args.fields)
    refs = {t: Reference(t) for t in REFS}
    names = args.candidates.split(',')
    out, base = {}, None
    for name in names:
        path = checkpoint(name)
        model = load_surrogate(path, shield.phi_0, args.device)
        cached = os.path.join(EXPERIMENTS, f'final_compare_{name}_scores.json')   # the slow part: kept across runs
        if os.path.exists(cached) and os.path.getmtime(cached) > os.path.getmtime(path):
            with open(cached) as f:
                scores = json.load(f)
        else:
            scores = score(as_study(model), refs, {})
            with open(cached, 'w') as f:
                json.dump(scores, f)
        r = dict(kind=model.hp.get('kind', 'deeponet'), scores=scores)
        print(name, 'scores', json.dumps(scores), flush=True)
        H0, g, h = at_nominal(model, shield, grid)
        mem = [at_nominal(m, shield, grid) for m in load_members(path, shield.phi_0, args.device)]
        if mem:
            G = [x[1] for x in mem]
            r['members_grad_cos'] = [cos(G[i], G[j]) for i in range(len(G)) for j in range(i + 1, len(G))]
            r['members_eig_cos'] = [abs(top_vec(x[2]) @ top_vec(h)) for x in mem]
        v = trust_region_max(g, h, 1.0)
        r.update(H0=H0, worst_ratio=float(ratio(model, shield, cgrid, v[None])[0]), g=g.tolist(), h=h.tolist(), v=v.tolist())
        if base is None:
            base = dict(model=model, g=g, h=h, v=v)
        else:
            r['vs_base'] = dict(grad_cos=cos(g, base['g']), eig_cos=abs(float(top_vec(h) @ top_vec(base['h']))),
                                worst_cos=cos(v, base['v']),
                                worst_ratio_by_base=float(ratio(base['model'], shield, cgrid, v[None])[0]))
        out[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k not in ('g', 'h', 'v')}, default=float), flush=True)
        del model

    cols = ['h0_err_T1', 'pred_half_T1', 'pred_box_T1', 'h0_err_T1.5', 'pred_box_T1.5', 'oat_slope_err', 'oat_curv_err']
    lines = [f'candidates at the final setup; baseline: {names[0]}; columns _T1: nominal muons, _T1.5: hot check sample', '',
             f'{"candidate":12s} {"kind":18s} ' + ' '.join(f'{c:>13s}' for c in cols)]
    for n, r in out.items():
        lines.append(f'{n:12s} {r["kind"]:18s} ' + ' '.join(f'{r["scores"][c]:13.3g}' for c in cols))
    lines += ['', f'{"candidate":12s} {"H0":>8s} {"members: grad cos":>18s} {"top-eig |cos|":>14s} '
              f'{"vs base: grad":>14s} {"eig":>6s} {"worst":>6s} {"H/H0 worst":>11s} {"(by base)":>10s}']
    for n, r in out.items():
        mg = f'{min(r["members_grad_cos"]):.3f}-{max(r["members_grad_cos"]):.3f}' if 'members_grad_cos' in r else '-'
        me = f'{min(r["members_eig_cos"]):.3f}' if 'members_eig_cos' in r else '-'
        vb = r.get('vs_base', {})
        lines.append(f'{n:12s} {r["H0"]:8.1f} {mg:>18s} {me:>14s} {vb.get("grad_cos", 1):14.3f} {vb.get("eig_cos", 1):6.3f} '
                     f'{vb.get("worst_cos", 1):6.3f} {r["worst_ratio"]:11.3f} {vb.get("worst_ratio_by_base", r["worst_ratio"]):10.3f}')
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(EXPERIMENTS, exist_ok=True)
    with open(os.path.join(EXPERIMENTS, 'final_compare.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(EXPERIMENTS, 'final_compare.json'), 'w') as f:
        json.dump(out, f, default=float)
    print('\n'.join(lines))
