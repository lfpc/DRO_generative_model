"""Score a surrogate on the reference designs (validation/references.py), on the same muons.

On each reference: predicted vs. simulated hits at u = 0 (h0_err), in the half box and in the box
(pred_half, pred_box: median |log pred / sim|); with one-at-a-time designs also the secants
(H(+e_i) - H(-e_i)) / 2 and curvatures H(+e_i) + H(-e_i) - 2 H(0) of every parameter
(oat_slope_err, oat_curv_err: relative vector errors, and their cosines). Each score is quoted
next to its noise floor: what a perfect model would read, from the simulator's Poisson noise.

Writes <out>/scores.json, report.txt, predictions.npz (--replot redraws the figures from it), and
to <figs>:
    hits_<ref>.png         predicted vs. simulated hits of every design of a reference, coloured
                           by its distance from the nominal design
    calibration_<ref>.png  per muon: the simulated hit rate against the predicted p(hit), over
                           all the designs of a reference (separable surrogates)
    oat.png                the one-at-a-time secants

    python -m validation.validate                       # models/lcso.pt -> results/validation/, figures/validation/
    CUDA_VISIBLE_DEVICES= python -m validation.validate  # on CPU
"""
import json
import os

import numpy as np
import torch

import cli
from lcso.data import to_x
from lcso.models import derivatives
from lcso.surrogate import load_surrogate
from paths import VALIDATION, figures, save_figure
from shield import MuonShield
from validation import references

REFS = {'T1': 'validation: nominal muons', 'T1.5': 'hot check sample (T = 1.5, as in training)'}


class Reference:
    """A finished reference run, its muons as trunk inputs (lcso.data.to_x) in pinned host memory."""

    def __init__(self, name, block=50_000_000):
        self.name, self.block, self._X = name, block, None
        self.names, self.U, self.hits, self.n, self.path = references.load(name)
        self.U = torch.as_tensor(self.U)

    @property
    def X(self):
        """The muons as trunk inputs, read on first use."""
        if self._X is None:
            self._X = torch.cat([to_x(torch.from_numpy(references.read_muons(self.path, lo, self.block)))
                                 for lo in range(0, self.n, self.block)])
            if torch.cuda.is_available():
                self._X = self._X.pin_memory()
        return self._X

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


    @torch.no_grad()
    def calibration(self, model, edges=np.logspace(-10, 0, 201)):
        """Predicted hits of every design, and per bin of the predicted p(hit) of a muon, over
        every (design, muon) pair: (pairs, predicted hits, simulated hits), bins under/over edges
        included (a separable model)."""
        dev = model.bias.device
        e = torch.as_tensor(np.log10(edges), dtype=torch.float32, device=dev)
        b = model.branch(self.U.to(dev))
        K = len(edges) + 1
        pred = torch.zeros(len(b), dtype=torch.float64, device=dev)
        n, s, h = (torch.zeros(K, dtype=torch.float64, device=dev) for _ in range(3))
        chunk = 2 ** 27 // (model.P + 3 * len(b))
        for c in range(0, self.n, chunk):
            p = torch.sigmoid(b @ model.trunk(self.X[c:c + chunk].to(dev, non_blocking=True)).T + model.bias)
            pred += p.double().sum(1)
            k = torch.bucketize(p.log10(), e).view(-1)
            n += torch.bincount(k, minlength=K).double()
            s += torch.bincount(k, p.view(-1).double(), minlength=K)
        for i, idx in references.hit_index(self.name).items():
            p = torch.sigmoid(model.trunk(self.X[torch.as_tensor(idx)].to(dev)) @ b[i] + model.bias)
            h += torch.bincount(torch.bucketize(p.log10(), e), minlength=K).double()
        return pred.cpu().numpy(), tuple(v.cpu().numpy() for v in (n, s, h))


def merge_bins(n, s, h, min_pred=20):
    """Adjacent bins merged until each holds min_pred predicted hits: (mean p, simulated rate,
    its Poisson error) per merged bin."""
    groups, cur = [], np.zeros(3)
    for v in zip(n, s, h):
        cur += v
        if cur[1] >= min_pred:
            groups.append(cur)
            cur = np.zeros(3)
    if cur[0] and groups:
        groups[-1] = groups[-1] + cur
    n, s, h = np.array(groups).T
    return s / n, h / n, np.sqrt(np.maximum(h, 1)) / n


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


def plot(refs, preds, cals, labels, out, report=False):
    """The figures, as PNG in out; with report, also the report's as PDF (validation_<name>.pdf)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13})
    color, grey = '#e34948', '0.45'

    def log_axes(ax, which='xy'):
        for axis in [getattr(ax, f'{a}axis') for a in which]:
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    def tag_text(ax, R, tag):
        m, k = f'{R.n:.0e}'.split('e+')
        ax.text(0.04, 0.96, f'{REFS.get(tag, tag)}\n${m}\\times10^{{{int(k)}}}$ muons, {len(R.U)} designs',
                transform=ax.transAxes, va='top', fontsize=11)

    def save(fig, name):
        save_figure(fig, out, name, report_name=f'validation_{name}' if report else None, dpi=200)
        plt.close(fig)

    def figure():
        fig = plt.figure(figsize=(6.4, 6.2), layout='constrained')
        ax, rx = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1))
        return fig, ax, rx

    # per reference: predicted vs. simulated hits, coloured by the distance from the nominal design
    dist = {t: R.U.norm(dim=1).numpy() for t, R in refs.items()}
    cnorm = matplotlib.colors.Normalize(0, max(d.max() for d in dist.values()))    # one scale for all
    for tag, R in refs.items():
        fig, ax, rx = figure()
        sim, pred, se, o = R.hits, preds[tag], np.sqrt(R.hits), np.argsort(-dist[tag])   # nearest on top
        lim = np.array([sim.min() * 0.75, sim.max() * 1.3])
        ax.fill_between(lim, lim * 0.9, lim * 1.1, color='0.93', lw=0, zorder=0, label=r'$\pm10\%$')
        ax.plot(lim, lim, color=grey, lw=1, zorder=1)
        rx.axhspan(0.9, 1.1, color='0.93', lw=0, zorder=0)
        rx.axhline(1, color=grey, lw=1, zorder=1)
        for a, y, yerr in ((ax, pred, None), (rx, pred / sim, pred / sim * se / sim)):
            a.errorbar(sim, y, xerr=se, yerr=yerr, fmt='none', ecolor='0.7', elinewidth=0.7, zorder=2)
            sc = a.scatter(sim[o], y[o], c=dist[tag][o], cmap='viridis', norm=cnorm, s=34,
                           edgecolors='white', linewidths=0.7, zorder=3)
        ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim, ylabel='predicted hits')
        rx.set(xlabel='simulated hits', ylabel='pred. / sim.')
        log_axes(ax)
        log_axes(rx, 'x')
        tag_text(ax, R, tag)
        ax.legend(loc='lower right', fontsize=10, frameon=False)
        fig.colorbar(sc, ax=[ax, rx], shrink=0.8, label=r'distance from nominal $\|u\|_2$  ($u_i = \pm1$: $\pm10\%$)')
        save(fig, f'hits_{tag}')

    # per reference: per-muon calibration, simulated hit rate against predicted p(hit)
    for tag, cal in cals.items():
        fig, ax, rx = figure()
        p, rate, err = merge_bins(*cal)
        lim = np.array([p.min() / 2, min(1, p.max() * 2)])
        ax.plot(lim, lim, color=grey, lw=1, zorder=1)
        rx.axhline(1, color=grey, lw=1, zorder=1)
        for a, y, yerr in ((ax, rate, err), (rx, rate / p, err / p)):
            a.errorbar(p, y, yerr=yerr, fmt='o', ms=5, color=color, mec='white', mew=0.7, elinewidth=1, zorder=3)
        ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim, ylabel='simulated hit rate')
        rx.set(xlabel='predicted $p$(hit) of a muon', ylabel='sim. / pred.')
        log_axes(ax)
        log_axes(rx, 'x')
        tag_text(ax, refs[tag], tag)
        save(fig, f'calibration_{tag}')

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
    save(fig, 'oat')


if __name__ == '__main__':
    parser = cli.add_common(cli.parser(__doc__))
    parser.add_argument('--refs', default=','.join(REFS), help='references (paths.REFERENCES/<name>)')
    parser.add_argument('--out', default=VALIDATION)
    parser.add_argument('--figs', default=figures(VALIDATION))
    parser.add_argument('--replot', action='store_true', help='only redraw the figures, from <out>/predictions.npz')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    for d in (args.out, args.figs):
        os.makedirs(d, exist_ok=True)
    shield = MuonShield(args.muons, fields=args.fields)
    model = as_study(load_surrogate(args.model, shield.phi_0, args.device))
    refs = {t: Reference(t) for t in args.refs.split(',')}
    saved = os.path.join(args.out, 'predictions.npz')
    if args.replot:
        z = np.load(saved)
        preds = {t: z[f'pred_{t}'] for t in refs}
        cals = {t: tuple(z[f'cal_{t}']) for t in refs if f'cal_{t}' in z}
        plot(refs, preds, cals, shield.labels, args.figs, report=args.figs == figures(VALIDATION))
        raise SystemExit(f'figures redrawn in {args.figs}')
    preds, cals = {}, {}
    for t, R in refs.items():
        if model.separable:
            preds[t], cals[t] = R.calibration(model)
        else:
            preds[t] = R.predict(model)
    np.savez(saved, **{f'pred_{t}': v for t, v in preds.items()}, **{f'cal_{t}': np.stack(v) for t, v in cals.items()})
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
    plot(refs, preds, cals, shield.labels, args.figs, report=args.figs == figures(VALIDATION))
    print(f'saved {args.out}/scores.json, report.txt, predictions.npz and the figures in {args.figs}')
