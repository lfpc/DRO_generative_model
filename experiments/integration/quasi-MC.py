import argparse
import os
import sys
from importlib import import_module
from math import comb

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')

PROBLEMS = {
    'two_moons': dict(cls='TwoMoons', kwargs=dict(kind='poly', degree=5),
                      pack=lambda s: torch.cat([s[0], s[1].unsqueeze(1)], 1),
                      apply=lambda f, phi, x: f(phi, x[:, :2], x[:, 2]),
                      # the third latent is read as a class, so every 1-D rule must be even:
                      # an odd one puts a node exactly on the z=0 boundary and hands its
                      # whole weight -- 2/3 of the mass, for a 3-point rule -- to one class
                      grow=lambda i: 2 * i, rot=2),
    'portfolio_optimization': dict(cls='Portfolio', kwargs={},
                                   pack=lambda s: s,
                                   apply=lambda f, phi, x: f(phi, x),
                                   # no discrete coordinate here, so the rule may start from
                                   # a single node and grow 1, 3, 5, ... -- which is the only
                                   # thing that makes a sparse grid affordable in 10-D
                                   grow=lambda i: 2 * i - 1, rot=None),
    # the three below share the plain shape: one tensor of points, loss(phi, x), a latent
    # with no discrete coordinate, so every rule may start from a single node and rotate
    # freely. They differ in what makes them hard -- multimodality, a near-degenerate
    # covariance, and a heavy tail respectively -- and in latent dimension, 8, 12 and 4.
    'newsvendor': dict(cls='Newsvendor', kwargs={}, pack=lambda s: s,
                       apply=lambda f, phi, x: f(phi, x),
                       grow=lambda i: 2 * i - 1, rot=None),
    'short_column': dict(cls='ShortColumn', kwargs={}, pack=lambda s: s,
                         apply=lambda f, phi, x: f(phi, x),
                         grow=lambda i: 2 * i - 1, rot=None),
    'beam_shield': dict(cls='BeamShield', kwargs={}, pack=lambda s: s,
                        apply=lambda f, phi, x: f(phi, x),
                        grow=lambda i: 2 * i - 1, rot=None),
}


def setup(name, models=None):
    """Load one problem's already-trained models; nothing is fitted here.

    Both problem folders define problem.py, train_generator.py and run_inner_cost.py, so
    only the chosen folder is ever put on the path -- importing both would resolve those
    names to whichever came first.
    """
    global PROBLEM, PHI, SURROGATE, APPLY, GROW, ROT, Generator
    spec = PROBLEMS[name]
    sys.path[:0] = [ROOT, os.path.join(ROOT, name)]
    models = models or os.path.join(ROOT, name, 'models')
    PROBLEM = getattr(import_module('problem'), spec['cls'])(**spec['kwargs'])
    Generator = import_module('train_generator').Generator
    PHI = torch.load(os.path.join(models, 'design.pt'), weights_only=False)
    SURROGATE = import_module('run_inner_cost').load_surrogate(
        os.path.join(models, 'surrogate.pt'), PROBLEM)
    APPLY, GROW, ROT = spec['apply'], spec['grow'], spec['rot']
    return models, spec['pack'](torch.load(os.path.join(models, 'data.pt'),
                                           weights_only=False))


def true_loss(x):
    """The simulator. Per-sample loss, shape (n,)."""
    return APPLY(PROBLEM.loss, PHI, x)


def surrogate_loss(x):
    """The surrogate. Per-sample loss, shape (n,)."""
    return APPLY(SURROGATE, PHI, x)


def sobol_normal(n, d, seed=0):
    """n Sobol points in R^d pushed to a standard normal by the inverse CDF.

    Scrambled, so each seed is an independent randomisation and a spread over seeds is a
    genuine error bar -- the "randomised" in RQMC. Drawn in float64 and mapped with ndtri
    rather than float32 and erfinv(2u-1): float32 resolves u near 1 to about 6e-8, which
    truncates the far tail. It is worth ~1e-6 on the estimate here, an order of magnitude
    under the QMC error, but it costs nothing.

    Sobol's balance holds for blocks of 2**m, which is why the default sizes are powers of
    two; drawing from index 0 makes this identical to draw_base2(m).
    """
    eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    return torch.special.ndtri(eng.draw(n, dtype=torch.float64)).to(torch.get_default_dtype())

def gauss_hermite(m, d):
    """Tensor-product Gauss-Hermite for E_{z~N(0,I_d)}[f(z)]: m**d nodes and their weights.

    This is the same integral the samplers below estimate, with the Gaussian density written
    out as an explicit weight instead of being absorbed into where the points are drawn.
    Exact whenever f is a polynomial of degree < 2m in each coordinate, so it converges at a
    rate no random rule can touch -- but only while f stays smooth, and only while m**d is
    affordable, which is the whole trade-off being tested here.
    """
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
    """Sparse-grid Gauss-Hermite: Smolyak's combination of 1-D rules of level |k| <= q.

    A full tensor product needs m**d nodes, which past a handful of dimensions leaves one
    affordable rule and no way to refine it. The sparse grid keeps only the low-order mixed
    terms -- enough to stay exact on the same total-degree polynomials -- and grows like
    2**q instead of m**d. The price is weights that go negative: sum|w| reaches 265 in 3-D
    and 445 in 10-D over the levels used here, against exactly 1 for every tensor rule, and
    that factor multiplies whatever roughness the integrand has.
    """
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


# integration rule (where its points come from). The rule is the thing being compared;
# the sampler in brackets says whether those points are drawn from the data file or pushed
# through the generative model, which is what separates sampling error from model bias.
MC_DATA = 'Monte Carlo (data)'
MC_MODEL = 'Monte Carlo (model)'
QMC = 'Quasi-MC (model)'
TENSOR = 'Gauss-Hermite grid (model)'
SPARSE = 'Smolyak sparse grid (model)'

COLOR = {MC_DATA: 'C0', MC_MODEL: 'C2', QMC: 'C1', TENSOR: 'C3', SPARSE: 'C4'}


def sampler_of(method):
    """'data' or 'model' -- the bracketed half of a method name, which fixes what it
    converges to."""
    return method[method.index('(') + 1:-1]
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
    """Quadrature nodes pushed through the model, carrying their Gaussian weights.

    A fixed rule returns the same number every time, so on its own it reports no
    uncertainty at all. N(0,I) is rotation-invariant, so f(Qz) integrates to the same
    value for any orthogonal Q: rotating the nodes gives a different valid rule for the
    same integral, and the spread over random Q is an honest error bar. Unlike a scrambled
    Sobol set it is not unbiased -- each rotated rule keeps its own deterministic error and
    averaging does not cancel them -- so read the band as how much the answer depends on an
    arbitrary choice of axes, which for a rule that has resolved the integrand is nearly
    nothing. Only the first `rot` coordinates turn, so a latent read as a discrete class
    keeps its axis and its exact half-and-half split.
    """
    z, w = rule
    if generator is not None:
        z = z @ random_rotation(model.latent_dim, rot or model.latent_dim, generator).T
    return model(z), w


def rules(d, cap, grow=None):
    """Every affordable rule of one kind, keyed by what it costs in loss evaluations.

    grow=None gives the full tensor product, over even node counts only -- see the note on
    the discrete latent coordinate in PROBLEMS. Otherwise a Smolyak sparse grid built from
    1-D rules of grow(1), grow(2), ... nodes.
    """
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
def estimate(n, sampler, seed=42):
    """Draw n points once and evaluate BOTH losses on them.

    Sharing the draw is not just tidier: it makes the true-vs-surrogate comparison paired,
    so the sampling noise common to both cancels in their difference and what is left is
    the surrogate's own error. Drawing separately would bury a small bias under Monte-Carlo
    noise of the same size.

    A sampler returns (points, weights); weights None means every point counts 1/n, which is
    what a draw from the measure already buys you. A quadrature rule carries them explicitly.
    """
    g = torch.Generator().manual_seed(seed)
    x, w = sampler(n, generator=g)
    mean = (lambda v: v.mean() if w is None else (v * w).sum())
    return {'true': mean(true_loss(x)).item(),
            'surrogate': mean(surrogate_loss(x)).item()}


@torch.no_grad()
def converged(model, k=8, n=2 ** 19):
    """The value the model-side rules converge to: k independent Sobol scramblings.

    A long plain-MC run is not good enough here. Its own error, sigma/sqrt(n), is ~1e-4 at
    n = 2e6 -- the same size as the quadrature error we are trying to resolve, so it would
    manufacture a floor that is really just the reference wobbling. Averaging scrambles
    instead gets to ~1e-6 and also reports its own uncertainty.
    """
    e = [estimate(n, lambda j, generator=None: sample_from_model(model, j, generator, qmc=True),
                  seed=s) for s in range(k)]
    return {loss: float(np.mean([x[loss] for x in e])) for loss in STYLE}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', default='two_moons', choices=sorted(PROBLEMS),
                    help='which experiment folder to take the trained models from')
    ap.add_argument('--models', default=None, help='override the checkpoint folder')
    ap.add_argument('--sizes', nargs='+', type=int,
                    default=[16, 64, 256, 1024, 4096, 16384, 65536])
    ap.add_argument('--repeats', type=int, default=10,
                    help='independent draws per N: the curves are their mean, the band their sd')
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)

    models, data = setup(a.problem, a.models)
    model = Generator.load(os.path.join(models, 'generator.pt'))
    d = model.latent_dim
    # m nodes per latent dimension cost m**d evaluations. Capping that at the largest MC
    # budget keeps every rule on the figure priced the same -- and in more than a handful of
    # dimensions it is what leaves the quadrature with almost no affordable rule at all.
    full = rules(d, max(a.sizes))
    sparse = rules(d, max(a.sizes), grow=GROW)
    print(f'{a.problem}: latent dim {d}, design dim {PHI.numel()}, '
          f'nominal sample {tuple(data.shape)}')
    for lab, r in (('tensor', full), ('sparse', sparse)):
        print(f'  {lab} rules within the {max(a.sizes):,}-evaluation cap: '
              + (', '.join(f'{n:,}' for n in r) or 'none'))

    samplers = {
        MC_DATA:  (a.sizes, lambda k, generator=None: sample_from_data(data, k, generator)),
        MC_MODEL: (a.sizes, lambda k, generator=None: sample_from_model(model, k, generator)),
        QMC:      (a.sizes, lambda k, generator=None: sample_from_model(model, k, generator,
                                                                        qmc=True)),
        # a quadrature rule costs whatever its node count is, so its x-axis is not a.sizes
        TENSOR:   (list(full), lambda k, generator=None: quadrature_from_model(
                       model, full[k], generator, ROT)),
        SPARSE:   (list(sparse), lambda k, generator=None: quadrature_from_model(
                       model, sparse[k], generator, ROT)),
    }

    # every estimator is measured against the value its own sample converges to: the whole
    # data file for the data side, a long QMC run for the model side. Comparing the model's
    # rules against the data's number would mix the generator's bias into their error.
    with torch.no_grad():
        gold = {'data': {k: v.item() for k, v in
                         zip(STYLE, (true_loss(data).mean(), surrogate_loss(data).mean()))}}
    gold['model'] = converged(model)
    print(f'converged values:  data {gold["data"]["true"]:.5f} (true) '
          f'{gold["data"]["surrogate"]:.5f} (surrogate)   '
          f'model {gold["model"]["true"]:.5f} / {gold["model"]["surrogate"]:.5f}')

    out = {}
    for src, (sizes, sampler) in samplers.items():
        v = np.array([[estimate(n, sampler, seed=1000 * r + n)[loss]
                        for r in range(a.repeats)]
                      for loss in STYLE for n in sizes])
        for i, loss in enumerate(STYLE):
            out[(src, loss)] = (np.array(sizes, float), v[i * len(sizes):(i + 1) * len(sizes)])

    for src, (sizes, _) in samplers.items():
        print('\n%-26s %s' % (src, ' '.join('%10d' % n for n in sizes)))
        for loss in STYLE:
            vals = out[(src, loss)][1]
            sd = vals.std(1, ddof=1)
            print('  %-32s %s' % (f'{loss} loss',
                                  ' '.join('%10.5f' % x for x in vals.mean(1))))
            print('  %-32s %s' % ('  sd over repeats', ' '.join('%10.5f' % x for x in sd)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, constrained_layout=True)
    for ax, loss in zip(axes, STYLE):
        for src in samplers:
            sizes, vals = out[(src, loss)]
            mu = vals.mean(1)
            sd = vals.std(1, ddof=1) if vals.shape[1] > 1 else np.zeros(len(sizes))
            ax.plot(sizes, mu, color=COLOR[src], marker='o', ms=4, mfc='none', lw=1.6,
                    label=src)
            ax.fill_between(sizes, mu - sd, mu + sd, color=COLOR[src], alpha=.13, lw=0)
        # each panel gets its own nominal line: the whole data file under that same loss
        ax.axhline(gold['data'][loss], color='.4', lw=1, ls=':',
                   label=f'full nominal sample ({data.shape[0]:,})')
        ax.set(xscale='log', xlabel='loss evaluations', title=f'{loss} loss')
        ax.legend(fontsize=7, loc='lower right')
    axes[0].set_ylabel(f'estimated total loss ({a.repeats} runs, mean $\\pm$ sd)')
    fig.suptitle(a.problem, fontsize=10)
    os.makedirs(OUT, exist_ok=True)
    name = f'integration_loss_{a.problem}.png'
    fig.savefig(os.path.join(OUT, name), dpi=150)
    print(f'\n  wrote {OUT}/{name}')
