"""Integration rules for the nominal risk at a fixed design: plain MC (data or generator),
scrambled Sobol QMC, tensor Gauss-Hermite and Smolyak sparse grids through the generator.
Each is scored on the true loss and the surrogate, against its own converged value.

    python3 quasi-MC.py --problem two_moons
    python3 quasi-MC.py --problem muon_shield --sizes 1024 4096 16384

Writes figs/integration_loss_<problem>.png.
"""
import argparse
import os
import sys
from importlib import import_module
from math import comb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')

PROBLEMS = {
    'two_moons': dict(cls='TwoMoons', kwargs=dict(kind='poly', degree=5),
                      pack=lambda s: torch.cat([s[0], s[1].unsqueeze(1)], 1),
                      apply=lambda f, phi, x: f(phi, x[:, :2], x[:, 2]),
                      # the third latent is read as a class: 1-D rules must have an even
                      # node count, or a node sits on z=0 and its weight goes to one class
                      grow=lambda i: 2 * i, rot=2),
    'portfolio_optimization': dict(cls='Portfolio', kwargs={},
                                   pack=lambda s: s,
                                   apply=lambda f, phi, x: f(phi, x),
                                   # no discrete latent: rules grow 1, 3, 5, ... nodes
                                   grow=lambda i: 2 * i - 1, rot=None),
    'newsvendor': dict(cls='Newsvendor', kwargs={}, pack=lambda s: s,
                       apply=lambda f, phi, x: f(phi, x),
                       grow=lambda i: 2 * i - 1, rot=None),
    'short_column': dict(cls='ShortColumn', kwargs={}, pack=lambda s: s,
                         apply=lambda f, phi, x: f(phi, x),
                         grow=lambda i: 2 * i - 1, rot=None),
    'beam_shield': dict(cls='BeamShield', kwargs={}, pack=lambda s: s,
                        apply=lambda f, phi, x: f(phi, x),
                        grow=lambda i: 2 * i - 1, rot=None),
    # six flow latents plus two symmetry coordinates (azimuth, sign of dphi -- a sign, so
    # even rules only); only the flow's six are rotated. Rare hits: log axis; the sparse
    # grid's negative weights swamp the signal, so it is off. Reference: 8 x 2**24 muons.
    'muon_shield': dict(cls='MuonShield', kwargs={},
                        pack=lambda s: torch.cat([s[0], s[1].unsqueeze(1)], 1),
                        apply=lambda f, phi, x: f(phi, x[:, :6], x[:, 6]),
                        grow=lambda i: 2 * i, rot=6, gold_n=2 ** 24, logy=True,
                        sparse=False, sizes=[4 ** k for k in range(5, 14)]),
}


def setup(name):
    """Load one problem's trained models (sets the module globals below). Returns the
    models folder and the packed nominal data. Only that problem folder goes on the path."""
    global PROBLEM, PHI, SURROGATE, APPLY, GROW, ROT, Generator
    spec = PROBLEMS[name]
    sys.path[:0] = [ROOT, os.path.join(ROOT, name)]
    models = os.path.join(ROOT, name, 'models')
    PROBLEM = getattr(import_module('problem'), spec['cls'])(**spec['kwargs'])
    Generator = import_module('train_generator').Generator
    PHI = torch.load(os.path.join(models, 'design.pt'), weights_only=False)
    SURROGATE = import_module('run_inner_cost').load_surrogate(
        os.path.join(models, 'surrogate.pt'), PROBLEM)
    APPLY, GROW, ROT = spec['apply'], spec['grow'], spec['rot']
    return models, spec['pack'](torch.load(os.path.join(models, 'data.pt'),
                                           weights_only=False))


def true_loss(x):
    """Per-sample true loss at PHI, shape (n,)."""
    return APPLY(PROBLEM.loss, PHI, x)


def surrogate_loss(x):
    """Per-sample surrogate loss at PHI, shape (n,)."""
    return APPLY(SURROGATE, PHI, x)


def sobol_normal(n, d, seed=0):
    """n scrambled Sobol points (one randomisation per seed) mapped to N(0, I_d).
    Drawn in float64 so the inverse CDF does not truncate the tails."""
    eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    return torch.special.ndtri(eng.draw(n, dtype=torch.float64)).to(torch.get_default_dtype())


def gauss_hermite(m, d):
    """Tensor-product Gauss-Hermite rule for E_{z~N(0,I_d)}: m**d nodes and weights."""
    x, w = np.polynomial.hermite_e.hermegauss(m)          # probabilists': weight exp(-x^2/2)
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    w = torch.as_tensor(w / w.sum(), dtype=torch.get_default_dtype())
    if d == 1:
        return x.unsqueeze(1), w
    return torch.cartesian_prod(*[x] * d), torch.cartesian_prod(*[w] * d).prod(1)


def _indices(d, total):
    """Every k in Z_{>=1}^d with sum(k) <= total, without enumerating the full box."""
    if d == 1:
        yield from ((i,) for i in range(1, total + 1))
        return
    for i in range(1, total - d + 2):
        for rest in _indices(d - 1, total - i):
            yield (i,) + rest


def smolyak_hermite(q, d, grow):
    """Smolyak sparse grid of 1-D Gauss-Hermite rules (grow(i) nodes at level i), |k| <= q.
    Duplicate nodes are merged. Weights can be negative (sum|w| of a few hundred)."""
    dt = torch.get_default_dtype()
    one = {}
    for m in {grow(i) for i in range(1, q - d + 2)}:
        x, w = np.polynomial.hermite_e.hermegauss(m)
        one[m] = (torch.as_tensor(x, dtype=dt), torch.as_tensor(w / w.sum(), dtype=dt))
    Z, W = [], []
    for k in _indices(d, q):
        if sum(k) < q - d + 1:
            continue
        c = (-1) ** (q - sum(k)) * comb(d - 1, q - sum(k))
        xs, ws = zip(*(one[grow(i)] for i in k))
        Z.append(torch.cartesian_prod(*xs) if d > 1 else xs[0].unsqueeze(1))
        W.append(c * (torch.cartesian_prod(*ws).prod(1) if d > 1 else ws[0]))
    z, w = torch.cat(Z), torch.cat(W)
    u, inv = torch.unique(torch.round(z * 1e5).long(), dim=0, return_inverse=True)
    return (torch.zeros(len(u), d, dtype=dt).index_copy_(0, inv, z),
            torch.zeros(len(u), dtype=dt).index_add_(0, inv, w))


# rule (where its points come from): data file or generator
MC_DATA = 'Monte Carlo (data)'
MC_MODEL = 'Monte Carlo (model)'
QMC = 'Quasi-MC (model)'
TENSOR = 'Gauss-Hermite grid (model)'
SPARSE = 'Smolyak sparse grid (model)'

COLOR = {MC_DATA: 'C0', MC_MODEL: 'C2', QMC: 'C1', TENSOR: 'C3', SPARSE: 'C4'}
STYLE = {'true': '-', 'surrogate': '--'}    # which loss evaluated it


def sample_from_data(data, n, generator=None):
    """n rows of the data file, without replacement."""
    idx = torch.randperm(data.shape[0], generator=generator)[:n]
    return data[idx], None


def sample_from_model(model, n, generator=None, qmc=False):
    """n rows of the model, with replacement."""
    d = model.latent_dim
    if qmc:
        z = sobol_normal(n, d, seed=generator.initial_seed())
    else:
        z = torch.randn(n, d, generator=generator)
    return model(z), None


def random_rotation(d, k, generator):
    """Haar-random rotation of the first k coordinates, identity on the rest."""
    q, r = torch.linalg.qr(torch.randn(k, k, generator=generator))
    m = torch.eye(d)
    m[:k, :k] = q * torch.sign(torch.diagonal(r))
    return m


def quadrature_from_model(model, rule, generator=None, rot=None):
    """Quadrature nodes pushed through the model, with their weights.

    The nodes are randomly rotated (N(0,I) is rotation-invariant), so repeats of a fixed
    rule give a spread: how much the answer depends on the choice of axes. Not unbiased.
    Only the first `rot` coordinates turn, so a discrete latent keeps its axis.
    """
    z, w = rule
    if generator is not None:
        z = z @ random_rotation(model.latent_dim, rot or model.latent_dim, generator).T
    return model(z), w


def rules(d, cap, grow=None):
    """Every rule with at most `cap` nodes, keyed by node count. grow=None: tensor
    Gauss-Hermite with even node counts; otherwise Smolyak sparse grids."""
    out = {}
    for i in range(1, 100):
        if grow is None:
            if (2 * i) ** d > cap:
                break
            z, w = gauss_hermite(2 * i, d)
        else:
            z, w = smolyak_hermite(d + i - 1, d, grow)
            if len(z) > cap:
                break
        out[len(z)] = (z, w)
    return out


@torch.no_grad()
def estimate(n, sampler, seed=42, losses=tuple(STYLE)):
    """Draw n points once and average each loss in `losses` on them (paired draw).

    A sampler returns (points, weights); weights None means equal weights 1/n.
    losses=('surrogate',) never calls the simulator.
    """
    g = torch.Generator().manual_seed(seed)
    x, w = sampler(n, generator=g)
    mean = (lambda v: v.mean() if w is None else (v * w).sum())
    fns = {'true': true_loss, 'surrogate': surrogate_loss}
    return {k: mean(fns[k](x)).item() for k in losses}


@torch.no_grad()
def converged(model, k=8, n=2 ** 19, losses=tuple(STYLE)):
    """Reference value for the model-side rules: mean of k Sobol scramblings of n points
    (far more accurate than plain MC of the same size)."""
    e = [estimate(n, lambda j, generator=None: sample_from_model(model, j, generator, qmc=True),
                  seed=s, losses=losses) for s in range(k)]
    return {loss: float(np.mean([x[loss] for x in e])) for loss in losses}


if __name__ == '__main__':
    class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter,
                       argparse.RawDescriptionHelpFormatter):
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--problem', default='two_moons', choices=sorted(PROBLEMS))
    ap.add_argument('--sizes', nargs='+', type=int, default=None,
                    help="loss evaluations per estimate; default: problem's, else 16 64 ... 65536")
    ap.add_argument('--gold_n', type=int, default=None,
                    help="Sobol points per scrambling of the reference; default: problem's, "
                         "else 2**19")
    ap.add_argument('--logy', action=argparse.BooleanOptionalAction, default=None,
                    help="log y axis; default: problem's, else off")
    ap.add_argument('--sparse', action=argparse.BooleanOptionalAction, default=None,
                    help="include the Smolyak sparse grid; default: problem's, else on")
    ap.add_argument('--repeats', type=int, default=10, help='independent draws per size')
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)

    spec = PROBLEMS[a.problem]
    a.sizes = a.sizes or spec.get('sizes', [16, 64, 256, 1024, 4096, 16384, 65536])
    logy = a.logy if a.logy is not None else spec.get('logy', False)
    use_sparse = a.sparse if a.sparse is not None else spec.get('sparse', True)
    models, data = setup(a.problem)
    model = Generator.load(os.path.join(models, 'generator.pt'))
    d = model.latent_dim
    # quadrature rules are capped at the largest MC budget, so all are priced the same
    full = rules(d, max(a.sizes))
    sparse = rules(d, max(a.sizes), grow=GROW) if use_sparse else {}
    print(f'{a.problem}: latent dim {d}, design dim {PHI.numel()}, '
          f'nominal sample {tuple(data.shape)}')
    for lab, r in [('tensor', full)] + ([('sparse', sparse)] if use_sparse else []):
        print(f'  {lab} rules within the {max(a.sizes):,}-evaluation cap: '
              + (', '.join(f'{n:,}' for n in r) or 'none'))

    samplers = {
        MC_DATA:  (a.sizes, lambda k, generator=None: sample_from_data(data, k, generator)),
        MC_MODEL: (a.sizes, lambda k, generator=None: sample_from_model(model, k, generator)),
        QMC:      (a.sizes, lambda k, generator=None: sample_from_model(model, k, generator,
                                                                        qmc=True)),
        # a quadrature rule costs its node count
        TENSOR:   (list(full), lambda k, generator=None: quadrature_from_model(
                       model, full[k], generator, ROT)),
        SPARSE:   (list(sparse), lambda k, generator=None: quadrature_from_model(
                       model, sparse[k], generator, ROT)),
    }
    # drop rules with no size (nothing under the cap, or the sparse grid switched off)
    samplers = {k: v for k, v in samplers.items() if v[0]}

    # references: the whole data file for the data side, long QMC for the model side
    with torch.no_grad():
        gold = {'data': {k: v.item() for k, v in
                         zip(STYLE, (true_loss(data).mean(), surrogate_loss(data).mean()))}}
    gold['model'] = converged(model, n=a.gold_n or spec.get('gold_n', 2 ** 19))
    print(f'converged values:  data {gold["data"]["true"]:.5f} (true) '
          f'{gold["data"]["surrogate"]:.5f} (surrogate)   '
          f'model {gold["model"]["true"]:.5f} / {gold["model"]["surrogate"]:.5f}')

    out = {}
    for src, (sizes, sampler) in samplers.items():
        e = [[estimate(n, sampler, seed=1000 * r + n) for r in range(a.repeats)] for n in sizes]
        for loss in STYLE:
            out[(src, loss)] = (np.array(sizes, float),
                                np.array([[x[loss] for x in row] for row in e]))

    for src, (sizes, _) in samplers.items():
        print('\n%-26s %s' % (src, ' '.join('%10d' % n for n in sizes)))
        for loss in STYLE:
            vals = out[(src, loss)][1]
            sd = vals.std(1, ddof=1)
            print('  %-32s %s' % (f'{loss} loss',
                                  ' '.join('%10.5f' % x for x in vals.mean(1))))
            print('  %-32s %s' % ('  sd over repeats', ' '.join('%10.5f' % x for x in sd)))

    # on a log axis each panel keeps its own limits
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=not logy, constrained_layout=True)
    for ax, loss in zip(axes, STYLE):
        # log axis floor: a factor 5 under the smallest positive estimate
        pos = np.concatenate([out[(src, loss)][1].mean(1) for src in samplers])
        floor = pos[pos > 0].min() / 5 if (pos > 0).any() else 1e-12
        for src in samplers:
            sizes, vals = out[(src, loss)]
            mu = vals.mean(1)
            sd = vals.std(1, ddof=1) if vals.shape[1] > 1 else np.zeros(len(sizes))
            lo = mu - sd
            if logy:                # drop zero estimates; a band reaching zero runs to the floor
                mu = np.where(mu > 0, mu, np.nan)
                lo = np.where(lo > floor, lo, floor)
            ax.plot(sizes, mu, color=COLOR[src], marker='o', ms=4, mfc='none', lw=1.6,
                    label=src)
            ax.fill_between(sizes, lo, mu + sd, where=np.isfinite(mu), color=COLOR[src],
                            alpha=.13, lw=0)
        ax.axhline(gold['data'][loss], color='.4', lw=1, ls=':',
                   label=f'full nominal sample ({data.shape[0]:,})')
        ax.set(xscale='log', xlabel='loss evaluations', title=f'{loss} loss')
        if logy:
            ax.set_yscale('log', nonpositive='clip')
            ax.set_ylim(bottom=floor)
        ax.legend(fontsize=7, loc='lower right')
    axes[0].set_ylabel(f'estimated total loss ({a.repeats} runs, mean $\\pm$ sd)'
                       + ('\nzero estimates not shown' if logy else ''))
    fig.suptitle(a.problem, fontsize=10)
    os.makedirs(OUT, exist_ok=True)
    name = f'integration_loss_{a.problem}.png'
    fig.savefig(os.path.join(OUT, name), dpi=150)
    print(f'\n  wrote {OUT}/{name}')
