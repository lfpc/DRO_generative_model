"""Score a surrogate on every design the robustness analyses simulated at the physical flux.

The analyses (analysis/*.py) simulate their designs on 1e9 (or 3e8) flow muons: the walks, the
worst cases, the one-at-a-time scans and the tolerance boxes, ~130 designs in all, from near the
nominal design to the worst found within +-10%. They are the designs the conclusions rest on, so
they make a check set of the surrogate at the physical flux, wider than the references
(validation/references.py), though chosen by an earlier surrogate. Each is compared on Y/Y_0 (the
surrogate on the quadrature grid, the simulator against the nominal of its own run):

    ratio  log(pred / sim) of Y/Y_0;  z = (pred - sim) / Poisson error of sim, both in Y/Y_0

Scores: median |ratio| and the fraction within 2 sigma, over all designs and over those the
simulator puts within x3 of nominal; and per group. Writes <out>/flux_check_<model>.json, .txt.

    python -m validation.flux_check --model outputs/models/lcso.pt
"""
import json
import os

import numpy as np
import torch

import cli
from analysis.core import cached_hits, feature_cache, one_group, setup
from paths import RESULTS, VALIDATION

RUNS = ('robustness', 'robustness_first')


def designs(run):
    """[(group, name, u (D,), sim Y/Y_0, its error)] of the designs simulated in results/<run>."""
    d = os.path.join(RESULTS, run)
    out = []

    def add(group, names, U, hits, h0):
        for n, u, h in zip(names, U, hits):
            r = h / h0
            out.append((group, f'{run}:{n}', np.asarray(u, np.float32), r, r * np.sqrt(1 / max(h, 1) + 1 / h0)))

    f = os.path.join(d, 'robustness.npz')
    r = np.load(f, allow_pickle=True) if os.path.exists(f) else None
    if r is not None and 'sim' in r.files and r['sim'][0] > 0:               # not a --no_sim run
        idx, t = r['dir_idx'], r['dir_t']
        names = {int(i): f'{n}@{tt:+.2f}' for n, ii, ts in zip(r['dir_names'], idx, t) for i, tt in zip(ii, ts) if i > 0}
        k = sorted(names)
        add('walks', [names[i] for i in k], r['designs'][k], r['sim'][k], float(r['sim'][0]))
    f = os.path.join(d, 'worst_cases.npz')
    if os.path.exists(f):
        w = np.load(f, allow_pickle=True)
        hits = w['sim_bin'].sum(1)
        m = w['names'] != 'nominal'
        add('worst cases', w['names'][m], w['U'][m], hits[m], float(hits[~m][0]))
    for k in (0, 1):
        f = os.path.join(d, f'scan_sim_{k}.npz')
        if os.path.exists(f):
            s = np.load(f, allow_pickle=True)
            labels = np.load(os.path.join(d, 'robustness.npz'), allow_pickle=True)['labels']
            U = np.zeros((len(s['params']), len(labels)), np.float32)
            U[np.arange(len(U)), s['params']] = s['t']
            add('scans', [f'{labels[p]}@{t:+.1f}' for p, t in zip(s['params'], s['t'])], U, s['hits'], float(s['nominal']))
    f = os.path.join(d, 'tolerance_sim_0.npz')
    if os.path.exists(f):
        s = np.load(f, allow_pickle=True)
        b = np.load(os.path.join(d, 'tolerance_boxes.npz'), allow_pickle=True)
        add('tolerance boxes', b['names'][s['idx']], b['U'][s['idx']], s['hits'], float(s['nominal']))
    return out


def scores(pred, sim, err):
    lr = np.log(pred / sim)
    z = (pred - sim) / err
    return dict(n=len(sim), median_abs_log=float(np.median(np.abs(lr))), mean_log=float(lr.mean()),
                within_2sigma=float(np.mean(np.abs(z) < 2)))


if __name__ == '__main__':
    parser = cli.add_common(cli.parser(__doc__))
    parser.add_argument('--runs', default=','.join(RUNS), help='results/<run> directories to collect')
    parser.add_argument('--grid', default='201x64', help='quadrature grid of the surrogate')
    parser.add_argument('--out', default=VALIDATION)
    args = parser.parse_args()

    D = sum((designs(r) for r in args.runs.split(',')), [])
    shield, model = setup(args)
    cache = feature_cache(model, shield.muons.points(args.grid, 1e9), one_group)
    U = np.stack([d[2] for d in D])
    pred = cached_hits(model, cache, 1, np.concatenate([np.zeros((1, U.shape[1])), U]))[:, 0]
    pred = pred[1:] / pred[0]
    sim, err = np.array([d[3] for d in D]), np.array([d[4] for d in D])
    groups = np.array([d[0] for d in D])

    res = {'all': scores(pred, sim, err), 'sim within x3': scores(pred[sim < 3], sim[sim < 3], err[sim < 3])}
    res.update({g: scores(pred[groups == g], sim[groups == g], err[groups == g]) for g in dict.fromkeys(groups)})
    name = os.path.splitext(os.path.basename(args.model))[0]
    lines = [f'model {args.model}; {len(D)} simulated designs of {args.runs}; Y/Y_0, surrogate on {args.grid}',
             f'{"set":18s} {"n":>4s} {"median|log|":>11s} {"mean log":>9s} {"|z|<2":>6s}']
    lines += [f'{k:18s} {v["n"]:4d} {v["median_abs_log"]:11.3f} {v["mean_log"]:+9.3f} {v["within_2sigma"]:6.2f}' for k, v in res.items()]
    lines += ['', f'{"design":40s} {"sim":>9s} {"pred":>9s} {"z":>6s}']
    lines += [f'{d[1]:40s} {s:9.3f} {p:9.3f} {(p - s) / e:+6.1f}' for d, s, p, e in zip(D, sim, pred, err)]
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, f'flux_check_{name}.txt'), 'w').write('\n'.join(lines) + '\n')
    json.dump(res, open(os.path.join(args.out, f'flux_check_{name}.json'), 'w'), indent=1)
    print('\n'.join(lines[:len(res) + 2]))
