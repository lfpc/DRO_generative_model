"""DRO outer loop: train a design against an adversary, then score it against every shift.

    python3 run_outer.py --problem short_column --surrogate taylor
    python3 run_outer.py --problem muon_shield --rounds 25 --n_test 100000000

Solves min_phi sup_{Q in U(P)} E_Q[l(phi, x)] by alternating: each round the adversary
draws a worst-case sample at the current design, a local surrogate is fitted around the
design on that sample, and the design takes a trust-region step on the surrogate
(Danskin: the gradient of the sup is the gradient at the maximising distribution).
Each trained design is then scored against all ambiguity sets plus a held-out shift.
"""
import argparse
import os
import sys
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')

# Per problem: class, whether a label rides along, the design box, and the radii of the
# ambiguity sets (each in its own units; recalibrated in main). The box is explicit because
# the portfolio and newsvendor only clamp from below in project().
# muon_shield is black_box (no input gradient): adversaries ascend a fitted surrogate and
# every risk is re-simulated, on far larger samples.
PROBLEMS = {
    'two_moons': dict(cls='TwoMoons', kwargs=dict(kind='poly', degree=5), labelled=True,
                      bounds=lambda p: (-3.0, 3.0), rho=.3, eps=.15, lat=.3, frac=1.),
    'portfolio_optimization': dict(cls='Portfolio', kwargs={}, labelled=False,
                                   bounds=lambda p: ([0.0] * p.n_assets + [-0.6],
                                                     [1.0] * p.n_assets + [0.6]),
                                   rho=.3, eps=.08, lat=.3, frac=1.),
    'newsvendor': dict(cls='Newsvendor', kwargs={}, labelled=False,
                       bounds=lambda p: ([0.0] * p.n_items + [0.0],
                                         [30.0] * p.n_items + [400.0]),
                       rho=.3, eps=.5, lat=.3, frac=1.),
    'short_column': dict(cls='ShortColumn', kwargs={}, labelled=False,
                         bounds=lambda p: p.bounds, rho=.3, eps=.08, lat=.3, frac=1.),
    'beam_shield': dict(cls='BeamShield', kwargs={}, labelled=False,
                        bounds=lambda p: (-p.b_max, p.b_max), rho=.3, eps=.005,
                        lat=.3, frac=1.),
    'muon_shield': dict(cls='MuonShield', kwargs={}, labelled=True,
                        bounds=lambda p: p.bounds, rho=.3, eps=.3, lat=.3, frac=1.,
                        black_box=True, n_test=100_000_000, n_adv=10_000_000,
                        per_design=1_000_000),
}
ADVERSARIES = ('nominal', 'kl', 'wasserstein', 'latent')


class Adapter:
    """A problem behind the interface optb.lcso expects. A label, if any, is packed as the
    last column so that a sample is always one tensor."""

    link = 'identity'

    def __init__(self, p, labelled, bounds):
        self.p, self.labelled = p, labelled
        self.dim = p.design_dim
        self.loss_scale = getattr(p, 'loss_scale', 1.0)    # see optb.lcso._make_surrogate
        lo, hi = bounds(p)
        self.lo = torch.as_tensor(lo, dtype=torch.float32).expand(self.dim).clone()
        self.hi = torch.as_tensor(hi, dtype=torch.float32).expand(self.dim).clone()

    def _split(self, x):
        return (x[:, :-1], x[:, -1]) if self.labelled else (x,)

    def pack(self, s):
        return torch.cat([s[0], s[1].unsqueeze(1).to(s[0].dtype)], 1) if self.labelled else s

    def sample(self, n, generator=None):
        return self.pack(self.p.sample(n, generator))

    def loss(self, phi, x):
        a = self._split(x)
        if phi.dim() == 1:
            return self.p.loss(phi, *a)
        return torch.stack([self.p.loss(q, *a) for q in phi])

    def risk(self, phi, x):
        return self.loss(phi, x).mean()

    def project(self, phi):
        """The problem's own feasibility (a simplex is not a box), then the box."""
        if hasattr(self.p, 'project'):
            phi = self.p.project(phi)
        return torch.maximum(torch.minimum(phi, self.hi), self.lo)

    def init_design(self):
        return self.p.init_design()


def setup(name):
    """Load one problem's module, inner loop and flow. Only that folder goes on the path:
    every problem folder defines problem.py and inner.py."""
    spec = PROBLEMS[name]
    sys.path[:0] = [ROOT, os.path.join(ROOT, name)]
    problem = getattr(import_module('problem'), spec['cls'])(**spec['kwargs'])
    inner = import_module('inner')
    gen = import_module('train_generator').Generator.load(
        os.path.join(ROOT, name, 'models', 'generator.pt'))
    flows = gen.flows if hasattr(gen, 'flows') else gen.flow
    return spec, problem, inner, flows


def surrogate_as_loss(ad, model):
    """A fitted surrogate as the (phi, *split sample) -> per-sample loss the inner loops take."""
    if hasattr(ad.p, 'wrap_surrogate'):        # muon shield: device and chunking
        return ad.p.wrap_surrogate(model)
    return lambda p, *a: model(p, ad.pack(a) if ad.labelled else a[0])


def draw_from_weights(w, n, g):
    """n indices drawn with replacement from weights w. torch.multinomial refuses more than
    2**24 categories, so large samples go through the inverse CDF."""
    w = w.clamp_min(0)
    if w.numel() <= 2 ** 24:
        return torch.multinomial(w, n, replacement=True, generator=g)
    c = torch.cumsum(w.double(), 0)
    u = torch.rand(n, generator=g, dtype=torch.float64) * c[-1]
    return torch.searchsorted(c, u).clamp_max(w.numel() - 1)


def adversarial(kind, spec, problem, inner, flows, ad, phi, x_nom, g, steps=60, loss_fn=None):
    """The worst-case sample at phi, as one packed tensor. A KL reweighting is resampled.

    The result is shuffled: the latent adversary returns its sample ordered by class, and
    any downstream slice of it would otherwise be a single class.
    """
    def shuffled(z):
        return z[torch.randperm(z.shape[0], generator=g)]

    if kind == 'nominal':
        return x_nom
    a = ad._split(x_nom)
    if kind == 'kl':
        _, w = inner.kl_dro(problem, phi, *a, spec['rho'])
        return shuffled(x_nom[draw_from_weights(w, x_nom.shape[0], g)])
    if kind == 'wasserstein':
        out = inner.wasserstein_dro(problem, phi, *a, spec['eps'], steps=steps,
                                    loss_fn=loss_fn)
        return shuffled(ad.pack((out[1], a[1])) if ad.labelled else out[1])
    out = inner.latent_dro(problem, flows, phi, spec['lat'], n=x_nom.shape[0],
                           steps=steps, loss_fn=loss_fn)
    return shuffled(ad.pack(out[1:3]) if ad.labelled else out[1])


def outer(ad, spec, problem, inner, flows, kind, surrogate, rounds=25, delta0=0.30, seed=0,
          epochs=150, radial=False):
    """Alternate adversary, local surrogate fit, design step. Returns (design, trace).

    For a black-box problem the adversary ascends this loop's own surrogate, so it is fitted
    once on nominal data first; round r's adversary then reads round r-1's model.
    """
    from optb.lcso import _derivatives, _fit, _make_surrogate, _feasible
    from dro.method import trust_region_max

    g = torch.Generator().manual_seed(seed)
    D = ad.dim
    n_designs = 4 * D
    per_design = spec.get('per_design', 400)
    n_adv = spec.get('n_adv', 4000)
    black_box = spec.get('black_box', False)
    lo, hi = ad.lo, ad.hi
    scale = ((hi - lo) / 2).clamp_min(1e-3)
    # start slightly off the initial design: two moons starts at exactly zero, where every
    # perturbation is symmetric, the surrogate gradient vanishes and the run freezes
    phi = ad.project(ad.init_design()
                     + 1e-2 * torch.randn(D, generator=g) * ((hi - lo) / 2).clamp_max(1.0))
    full_rank = surrogate.startswith(('taylor-res', 'taylor-full'))
    model = _make_surrogate(ad, surrogate, seed, rank=None if full_rank else 8)
    opt, delta, trace = None, delta0, []

    def fit_on(model, opt, sample):
        # designs around the incumbent, all scored on the SAME sample so the fit sees the
        # design dependence and not the shift. radial: each offset shrunk by a uniform factor
        designs = torch.stack([_feasible(ad, phi + delta * scale *
                                         (torch.rand(D, generator=g) * 2 - 1)
                                         * (float(torch.rand(1, generator=g)) if radial else 1.0),
                                         lo, hi)
                               for _ in range(n_designs)])
        designs[0] = phi
        if opt is not None and hasattr(model, 'phi_0'):      # re-centre on the incumbent
            model.phi_0 = phi
        idx = torch.randint(sample.shape[0], (n_designs, per_design), generator=g)
        X = sample[idx]
        with torch.no_grad():
            Y = torch.stack([ad.loss(designs[i], X[i]) for i in range(n_designs)])
        return _fit(model, designs, X, Y, epochs, seed, link='identity', opt=opt)

    if black_box:
        model, opt = fit_on(model, opt, ad.sample(n_adv, g))
    adv_loss = surrogate_as_loss(ad, model) if black_box else None

    for r in range(rounds):
        x_nom = ad.sample(n_adv, g)
        x_adv = adversarial(kind, spec, problem, inner, flows, ad, phi, x_nom, g,
                            loss_fn=adv_loss)
        model, opt = fit_on(model, opt, x_adv)
        if black_box:                          # the wrapper may hold a copy: refresh it
            adv_loss = surrogate_as_loss(ad, model)

        grad, hess = _derivatives(ad, model, phi, x_adv[:2000],
                                  need_hessian=surrogate.startswith('taylor'))
        g_u = grad * scale
        if hess is not None:
            h_u = hess * scale.unsqueeze(1) * scale.unsqueeze(0)
            p_u, _, _ = trust_region_max(-g_u, -h_u, delta)
        else:
            p_u = -g_u / max(float(torch.linalg.norm(g_u)), 1e-12) * delta
        cand = _feasible(ad, phi + p_u * scale, lo, hi).detach()

        # accept on the worst case (the adversary needs its graph, so not under no_grad)
        x_c = adversarial(kind, spec, problem, inner, flows, ad, cand, x_nom, g,
                          loss_fn=adv_loss)
        with torch.no_grad():
            f_new, f_old = float(ad.risk(cand, x_c)), float(ad.risk(phi, x_adv))
        if f_new < f_old - 1e-9:
            phi, delta = cand, min(delta * 1.6, 1.0)
        else:
            delta = max(delta * 0.6, 5e-3)
        trace.append((r, f_old))
    return phi.detach(), trace


def held_out_shift(ad, phi, x, frac=1.0, k=24, seed=7):
    """Worst of k fixed random mean shifts (frac standard deviations); no method trains on it."""
    g = torch.Generator().manual_seed(seed)
    a = ad._split(x)
    sd = a[0].std(0)
    worst, arg = -float('inf'), None
    for _ in range(k):
        u = torch.randn(a[0].shape[1], generator=g)
        moved = a[0] + frac * sd * (u / u.norm())
        xs = ad.pack((moved, a[1])) if ad.labelled else moved
        with torch.no_grad():
            v = float(ad.risk(phi, xs))
        if v > worst:
            worst, arg = v, xs
    return arg


def calibrate_radii(ad, spec, problem, inner, flows, phi, x, g, ratio=1.25, iters=9,
                    loss_fn=None):
    """Bisect each set's radius (in place, in spec) until it lifts phi's risk by `ratio`,
    so the columns of the table are comparable."""
    with torch.no_grad():
        base = float(ad.risk(phi, x))
    target, out = base + (ratio - 1.0) * abs(base), {}
    for kind, key in (('kl', 'rho'), ('wasserstein', 'eps'), ('latent', 'lat'),
                      ('held-out', 'frac')):
        lo, hi = 1e-4, 20.0
        for _ in range(iters):
            mid = (lo * hi) ** 0.5
            spec[key] = mid
            if kind == 'held-out':
                xa = held_out_shift(ad, phi, x, frac=mid)
            else:
                xa = adversarial(kind, spec, problem, inner, flows, ad, phi, x, g, steps=80,
                                 loss_fn=loss_fn)
            with torch.no_grad():
                v = float(ad.risk(phi, xa))
            lo, hi = (mid, hi) if v < target else (lo, mid)
        spec[key] = hi
        out[kind] = hi
    print(f'  radii calibrated to +{100*(ratio-1):.0f}% risk ({base:.4f} -> {target:.4f}): '
          + '  '.join(f'{k}={v:.4g}' for k, v in out.items()))
    return out


def score(ad, spec, problem, inner, flows, phi, x_test, g, loss_fn=None):
    """True risk of one design under every ambiguity set, at the calibrated radii."""
    with torch.no_grad():
        out = {'nominal': float(ad.risk(phi, x_test))}
    for kind in ('kl', 'wasserstein', 'latent'):
        xa = adversarial(kind, spec, problem, inner, flows, ad, phi, x_test, g, steps=120,
                         loss_fn=loss_fn)
        with torch.no_grad():
            out[kind] = float(ad.risk(phi, xa))
    with torch.no_grad():
        xh = held_out_shift(ad, phi, x_test, frac=spec['frac'])
        out['held-out'] = float(ad.risk(phi, xh))
    return out


if __name__ == '__main__':
    class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter,
                       argparse.RawDescriptionHelpFormatter):
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    ap.add_argument('--problem', default='two_moons', choices=sorted(PROBLEMS))
    ap.add_argument('--surrogate', nargs='+', default=['taylor', 'taylor-res', 'mlp'],
                    help="taylor | taylor-res[:...] | taylor-full | mlp (branch-trunk)")
    ap.add_argument('--rounds', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_test', type=int, default=None, help="default: problem's, else 8000")
    ap.add_argument('--n_adv', type=int, default=None, help="default: problem's, else 4000")
    ap.add_argument('--per_design', type=int, default=None, help="default: problem's, else 400")
    ap.add_argument('--local_fit', type=int, nargs=2, default=[60, 100000],
                    metavar=('N_DESIGNS', 'PER_DESIGN'),
                    help='black-box only: local surrogate around each scored design')
    ap.add_argument('--design_draw', default='uniform', choices=['uniform', 'radial'])
    ap.add_argument('--tag', default='', help='output figure suffix')
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)

    spec, problem, inner, flows = setup(a.problem)
    for k in ('n_test', 'n_adv', 'per_design'):
        if getattr(a, k) is not None:
            spec[k] = getattr(a, k)
    ad = Adapter(problem, spec['labelled'], spec['bounds'])
    g = torch.Generator().manual_seed(123)
    x_test = ad.sample(spec.get('n_test', 8000), g)
    # calibrate at the nominal optimum; a simulator has none, so use the inner studies' design
    if hasattr(problem, 'solve'):
        phi_ref = problem.solve(ad.sample(20000, g))
    elif hasattr(problem, 'fit'):
        phi_ref = problem.fit(*problem.sample(20000, g))
    else:
        phi_ref = torch.load(os.path.join(ROOT, a.problem, 'models', 'design.pt'),
                             weights_only=False)

    def local(phi):
        """What the scoring adversaries ascend: the truth, or for a black box a surrogate
        fitted around phi."""
        if not spec.get('black_box'):
            return None
        return import_module('run_inner_cost').fit_surrogate(
            problem, phi, n_designs=a.local_fit[0], n_samples=a.local_fit[1])[0]

    print(f'{a.problem}: design dim {ad.dim}')
    calibrate_radii(ad, spec, problem, inner, flows, phi_ref, x_test, g,
                    loss_fn=local(phi_ref))

    cols = ['nominal', 'kl', 'wasserstein', 'latent', 'held-out']
    results = {}
    for surrogate in a.surrogate:
        for kind in ADVERSARIES:
            phi, trace = outer(ad, spec, problem, inner, flows, kind, surrogate,
                               rounds=a.rounds, seed=a.seed,
                               radial=a.design_draw == 'radial')
            r = results[(surrogate, kind)] = score(ad, spec, problem, inner, flows, phi,
                                                   x_test, g, loss_fn=local(phi))
            print(f'  {surrogate:10s} trained vs {kind:12s} ' +
                  '  '.join(f'{c}={r[c]:8.4f}' for c in cols))

    titles = {'taylor': 'Taylor (closed-form Hessian)',
              'taylor-res': 'Taylor + residual (recentred)',
              'mlp': 'branch-trunk (backprop)'}
    fig, axes = plt.subplots(1, len(a.surrogate), figsize=(7.2 * len(a.surrogate), 4.2),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    w = 0.2
    for ax, surrogate in zip(axes, a.surrogate):
        for j, kind in enumerate(ADVERSARIES):
            v = np.array([results[(surrogate, kind)][c] for c in cols])
            ax.bar(np.arange(len(cols)) + (j - 1.5) * w, v, width=w, label=f'trained vs {kind}')
        ax.set(xticks=np.arange(len(cols)), xticklabels=cols,
               ylabel='true risk (lower is better)' if surrogate == a.surrogate[0] else None,
               title=f'{a.problem} -- {titles.get(surrogate, surrogate)}')
        ax.axhline(results[(surrogate, 'nominal')]['nominal'], color='.4', lw=1, ls=':')
        ax.legend(fontsize=7)
    name = f'outer_{a.problem}{a.tag}.png'
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=150)
    print(f'  wrote {OUT}/{name}')
