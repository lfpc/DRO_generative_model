"""Score a surrogate on the reference designs (validation/references.py), on the same muons.

On each reference: predicted vs. simulated hits at u = 0 (h0_err), in the half box and in the box
(pred_half, pred_box: median |log pred / sim|); with one-at-a-time designs also the secants
(H(+e_i) - H(-e_i)) / 2 and curvatures H(+e_i) + H(-e_i) - 2 H(0) of every parameter
(oat_slope_err, oat_curv_err: relative vector errors, and their cosines). Each score is quoted
next to its noise floor: what a perfect model would read, from the simulator's Poisson noise.

Writes <out>/scores.json, report.txt, calibration.png (predicted vs. simulated hits) and oat.png
(the one-at-a-time secants).

    python -m validation.validate                       # outputs/lcso.pt -> outputs/validation/
"""
import json
import os

import numpy as np
import torch

import cli
from lcso.data import to_x
from lcso.models import derivatives
from lcso.surrogate import load_surrogate
from paths import VALIDATION
from shield import MuonShield
from validation import references

REFS = {'T1': 'nominal muons', 'T1.5': 'hot check sample'}


class Reference:
    """A finished reference run, its muons as trunk inputs (lcso.data.to_x) in pinned host memory."""

    def __init__(self, name, block=50_000_000):
        self.name = name
        self.names, self.U, self.hits, self.n, path = references.load(name)
        self.X = torch.cat([to_x(torch.from_numpy(references.read_muons(path, lo, block)))
                            for lo in range(0, self.n, block)]).pin_memory()
        self.U = torch.as_tensor(self.U)

    def group(self, prefix):
        return np.array([n.startswith(prefix) for n in self.names])

    @torch.no_grad()
    def predict(self, model):
        """Predicted hits (sum over the muons) of every reference design."""
        dev = model.bias.device
        U = self.U.to(dev)
        out = torch.zeros(len(U), dtype=torch.float64, device=dev)
        if not model.separable:     # the whole network per design and muon; TF32 here (~1e-3 on the sums)
            chunk = 2 ** 24 // len(U)
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                for c in range(0, self.n, chunk):
                    out += torch.sigmoid(model.logits(U, self.X[c:c + chunk].to(dev, non_blocking=True))).double().sum(1)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf32
            return out.cpu().numpy()
        chunk = 2 ** 27 // (model.P + 3 * len(U))       # ~0.5 GB of features and logits at a time
        b = model.branch(U)
        for c in range(0, self.n, chunk):
            t = model.trunk(self.X[c:c + chunk].to(dev, non_blocking=True))
            out += torch.sigmoid(b @ t.T + model.bias).double().sum(1)
        return out.cpu().numpy()


def oat_pairs(names, values, h0):
    """(secants, curvatures) of the one-at-a-time designs oat+<i>, oat-<i> of values."""
    idx = {n: i for i, n in enumerate(names)}
    D = sum(n.startswith('oat+') for n in names)
    p = np.array([values[idx[f'oat+{i}']] for i in range(D)])
    m = np.array([values[idx[f'oat-{i}']] for i in range(D)])
    return (p - m) / 2, p + m - 2 * h0


def ref_scores(R, pred, tag):
    nom = R.group('nominal')
    s0 = R.hits[nom].mean()
    out = {f'h0_err_{tag}': float(pred[nom].mean() / s0 - 1)}
    for grp in ('half', 'box'):
        m = R.group(grp) & (R.hits > 0)
        out[f'pred_{grp}_{tag}'] = float(np.median(np.abs(np.log(pred[m] / R.hits[m]))))
    if R.group('oat').any():
        for name, sim, hat in zip(('slope', 'curv'), oat_pairs(R.names, R.hits, s0), oat_pairs(R.names, pred, pred[nom].mean())):
            out[f'oat_{name}_err'] = float(np.linalg.norm(hat - sim) / np.linalg.norm(sim))
            out[f'oat_{name}_cos'] = float(hat @ sim / (np.linalg.norm(hat) * np.linalg.norm(sim)))
    return out


def noise_floors(R, rng):
    """What the scores of R read for a perfect model, from the Poisson noise of its simulated
    counts (expected values under Gaussian noise)."""
    out = {f'h0_err_{R.name}': float(1 / np.sqrt(R.hits[R.group('nominal')].sum()))}
    for grp in ('half', 'box'):
        h = R.hits[R.group(grp)]
        out[f'pred_{grp}_{R.name}'] = float(np.median(np.abs(np.log1p(rng.standard_normal((2000, len(h))) / np.sqrt(h)))))
    if R.group('oat').any():
        h0 = R.hits[R.group('nominal')]
        s0, v0 = h0.mean(), h0.mean() / len(h0)
        slope, curv = oat_pairs(R.names, R.hits, s0)
        hp_hm = curv + 2 * s0                                                  # H(+e_i) + H(-e_i)
        out['oat_slope_err'] = float(np.sqrt((hp_hm / 4).sum()) / np.linalg.norm(slope))
        out['oat_curv_err'] = float(np.sqrt((hp_hm + 4 * v0).sum()) / np.linalg.norm(curv))
    return out


def score(model, refs, arrays):
    """The scores of a study network (lcso.models interface) on refs; the predictions, and the
    hits' gradient and Hessian at u = 0 on the first reference's muons, go to arrays."""
    out = {}
    for tag, R in refs.items():
        pred = R.predict(model)
        arrays[f'pred_{tag}'] = pred
        out.update(ref_scores(R, pred, tag))
    if not model.separable:
        return out
    X = next(iter(refs.values())).X
    H0, g, H = derivatives(model, X, chunk=min(2 ** 20, 2 ** 27 // model.P))
    arrays.update(H0=H0 * len(X), g=g.numpy() * len(X), H=H.numpy() * len(X))   # in hits, not rates
    return out


class Production(torch.nn.Module):
    """A surrogate of lcso.surrogate (an LCSONet) in the interface of lcso.models (branch, trunk)."""
    separable = True

    def __init__(self, model):
        super().__init__()
        self.m, self.bias, self.D, self.P = model, model.bias, model.dim, model.P

    def branch(self, u):                       # its u is ours: +-1 = +-10% of phi_0
        return self.m.branch_net(u)

    def trunk(self, x):                        # and so are its inputs (lcso.data.to_x)
        return self.m.trunk_x(x)


def as_study(model):
    """A loaded surrogate as an lcso.models network: a ModelSurrogate's own network, an LCSONet
    through Production."""
    return model.net if hasattr(model, 'net') else Production(model)


def figures(refs, preds, labels, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    color = '#e34948'
    fig, axes = plt.subplots(1, len(refs), figsize=(4.2 * len(refs), 3.9), constrained_layout=True, squeeze=False)
    for ax, (tag, R) in zip(axes[0], refs.items()):
        ax.errorbar(R.hits, preds[tag], xerr=np.sqrt(R.hits), fmt='o', ms=3.5, color=color, elinewidth=0.6, alpha=0.85)
        lim = [R.hits.min() * 0.8, R.hits.max() * 1.2]
        ax.plot(lim, lim, color='0.6', lw=1, zorder=0)
        ax.set(xscale='log', yscale='log', xlabel='simulated hits', ylabel='predicted hits',
               title=f'{REFS.get(tag, tag)} ({R.n:.0e} muons)')
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.suptitle('Predicted vs. simulated hits on the reference designs, same muons')
    fig.savefig(os.path.join(out, 'calibration.png'), dpi=150)

    tag, R = next(((t, R) for t, R in refs.items() if R.group('oat').any()), (None, None))
    if R is None:
        return
    s0 = R.hits[R.group('nominal')].mean()
    sim = oat_pairs(R.names, R.hits, s0)[0]
    err = np.sqrt(oat_pairs(R.names, R.hits, s0)[1] + 2 * s0) / 2
    hat = oat_pairs(R.names, preds[tag], preds[tag][R.group('nominal')].mean())[0]
    order = np.argsort(-np.abs(sim))
    D = len(sim)
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    ax.errorbar(np.arange(D), sim[order], yerr=err[order], fmt='s', color='0.25', ms=4, label='simulator (1 s.d.)')
    ax.plot(np.arange(D), hat[order], 'o', mfc='none', mec=color, ms=6, mew=1.5, label='surrogate')
    ax.axhline(0, color='0.7', lw=0.8)
    ax.set_xticks(range(D), [labels[i] for i in order], rotation=90, fontsize=6)
    ax.set_ylabel('(H(+10%) - H(-10%)) / 2  [hits]')
    ax.legend(fontsize=7)
    ax.set_title(f'One parameter at a time, {REFS.get(tag, tag)} ({R.n:.0e} muons): simulator vs. surrogate')
    fig.savefig(os.path.join(out, 'oat.png'), dpi=150)


if __name__ == '__main__':
    parser = cli.add_common(cli.parser(__doc__))
    parser.add_argument('--refs', default=','.join(REFS), help='references (paths.REFERENCES/<name>)')
    parser.add_argument('--out', default=VALIDATION)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(args.out, exist_ok=True)
    shield = MuonShield(args.muons)
    model = as_study(load_surrogate(args.model, shield.phi_0, args.device))
    refs = {t: Reference(t) for t in args.refs.split(',')}
    preds = {t: R.predict(model) for t, R in refs.items()}
    scores, floors, rng = {}, {}, np.random.default_rng(0)
    for t, R in refs.items():
        scores.update(ref_scores(R, preds[t], t))
        floors.update(noise_floors(R, rng))
    with open(os.path.join(args.out, 'scores.json'), 'w') as f:
        json.dump(dict(model=args.model, scores=scores, noise=floors), f, indent=1)
    lines = [f'model {args.model}; references ' + ', '.join(f'{t} ({R.n:.0e} muons, {len(R.U)} designs)' for t, R in refs.items()),
             f'{"score":20s} {"surrogate":>10s} {"noise":>10s}']
    lines += [f'{k:20s} {v:10.4f} {floors.get(k, np.nan):10.4f}' for k, v in scores.items()]
    with open(os.path.join(args.out, 'report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    figures(refs, preds, shield.labels, args.out)
    print(f'saved {args.out}/scores.json, report.txt, calibration.png, oat.png')
