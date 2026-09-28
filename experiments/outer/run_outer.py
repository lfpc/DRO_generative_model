"""The DRO outer loop: train a design against an adversary, test it against held-out shifts.

    python3 run_outer.py --problem short_column --surrogate taylor

Everything before this script solved the inner problem at a FIXED design. This one closes
the loop:

    min_phi  sup_{Q in U(P)}  E_Q[ l(phi, x) ]

by alternating. At each round the adversary produces a worst-case sample for the current
design, a surrogate is fitted in a neighbourhood of that design on that sample, and the
design steps against it. Danskin's theorem is what makes the alternation legitimate: at the
inner maximiser the gradient of the sup equals the gradient of the expectation under the
maximising distribution, so no derivative of the adversary is needed.

Two surrogates, which differ in exactly one thing that matters here:

  taylor     a quadratic in the design with a low-rank Hessian, so the gradient AND the
             curvature come out in closed form and the step can be a trust-region Newton.
  deeponet   a branch-trunk model <b(phi), t(x)>, whose design gradient needs a backward
             pass and whose Hessian is not requested at all, so the step is first order.

The comparison is whether the free curvature buys anything once the objective is a worst
case rather than an expectation -- the inner maximiser moves between rounds, so the
objective the outer step sees is not fixed, and second-order information about a moving
target may be worth less than it is on a static one.

What is measured is NOT the training objective. Each trained design is scored against every
ambiguity set, including the ones it was not trained on, plus a shift no method uses during
training. A design that is only good against its own adversary has not been made robust, it
has been fitted to a threat model.
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

# Per problem: the class, whether a label rides along, the design box, and the radii the
# ambiguity sets are given. Radii are in each set's own units and are not comparable across
# sets -- that is the point of scoring every design against all of them.
PROBLEMS = {
    # `bounds` is (lo, hi) per design coordinate, given explicitly rather than read off
    # project(). Three of these problems clamp from both sides and one box could have been
    # inferred; the portfolio and the newsvendor only clamp from below, so inferring gave a
    # half-width of 1e6, steps of that size, and a run that never moved off its start.
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
}
ADVERSARIES = ('nominal', 'kl', 'wasserstein', 'latent')


class Adapter:
    """The problem behind the interface optb.lcso expects.

    A label, where there is one, rides as a last column so that "a sample" is one tensor --
    the same packing the integration study uses, and what lets one outer loop serve five
    problems with different signatures.
    """

    link = 'identity'

    def __init__(self, p, labelled, bounds):
        self.p, self.labelled = p, labelled
        self.dim = p.design_dim
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
        """The problem's own feasibility first -- a simplex is not a box -- then the box."""
        if hasattr(self.p, 'project'):
            phi = self.p.project(phi)
        return torch.maximum(torch.minimum(phi, self.hi), self.lo)

    def init_design(self):
        return self.p.init_design()


def setup(name):
    """Load one problem's module, models and inner loop. Only the chosen folder goes on the
    path: all five define problem.py and inner.py."""
    spec = PROBLEMS[name]
    sys.path[:0] = [ROOT, os.path.join(ROOT, name)]
    models = os.path.join(ROOT, name, 'models')
    problem = getattr(import_module('problem'), spec['cls'])(**spec['kwargs'])
    inner = import_module('inner')
    gen = import_module('train_generator').Generator.load(
        os.path.join(models, 'generator.pt'))
    flows = gen.flows if hasattr(gen, 'flows') else gen.flow
    return spec, problem, inner, flows


def adversarial(kind, spec, problem, inner, flows, ad, phi, x_nom, g, steps=60):
    """The worst-case SAMPLE at the current design, as one packed tensor.

    A reweighting does not move points, so it is turned into a sample by drawing from the
    weights -- which is what the outer step needs, and which keeps all four adversaries on
    the same footing downstream.

    The result is shuffled before it is returned, and that is not tidiness. The latent
    adversary emits its sample ordered by class -- every positive, then every negative --
    so any downstream slice of it is a single class. Taking derivatives on such a slice
    gave a surrogate gradient with cosine -0.1 against the true one while the other three
    adversaries sat at +0.96, and the design froze at the trivial classifier for the whole
    budget. Nothing downstream should depend on the order a sample arrives in.
    """
    def shuffled(z):
        return z[torch.randperm(z.shape[0], generator=g)]

    if kind == 'nominal':
        return x_nom
    a = ad._split(x_nom)
    if kind == 'kl':
        _, w = inner.kl_dro(problem, phi, *a, spec['rho'])
        idx = torch.multinomial(w.clamp_min(0), x_nom.shape[0], replacement=True,
                                generator=g)
        return shuffled(x_nom[idx])
    if kind == 'wasserstein':
        out = inner.wasserstein_dro(problem, phi, *a, spec['eps'], steps=steps)
        return shuffled(ad.pack((out[1], a[1])) if ad.labelled else out[1])
    out = inner.latent_dro(problem, flows, phi, spec['lat'], n=x_nom.shape[0],
                           steps=steps)
    return shuffled(ad.pack(out[1:3]) if ad.labelled else out[1])


def outer(ad, spec, problem, inner, flows, kind, surrogate, rounds=25, n_designs=None,
          per_design=400, delta0=0.30, seed=0, epochs=150, n_adv=4000):
    """Alternate: adversary, local surrogate, design step. Returns (design, trace)."""
    from optb.lcso import _derivatives, _fit, _make_surrogate, _feasible
    from dro.method import trust_region_max

    g = torch.Generator().manual_seed(seed)
    D = ad.dim
    n_designs = n_designs or 4 * D
    lo, hi = ad.lo, ad.hi
    scale = ((hi - lo) / 2).clamp_min(1e-3)
    # Start just off the nominal initial design. Two moons' polynomial classifier starts at
    # exactly zero, where the loss is log 2 for every input and every design perturbation is
    # symmetric about it: the fitted surrogate reports no gradient, the step is zero, the
    # candidate equals the incumbent, and the run freezes at the zero classifier for the
    # whole budget. A small asymmetric offset costs nothing and removes the degeneracy.
    phi = ad.project(ad.init_design()
                     + 1e-2 * torch.randn(D, generator=g) * ((hi - lo) / 2).clamp_max(1.0))
    model = _make_surrogate(ad, surrogate, seed)
    opt, delta, trace = None, delta0, []

    for r in range(rounds):
        x_nom = ad.sample(n_adv, g)
        x_adv = adversarial(kind, spec, problem, inner, flows, ad, phi, x_nom, g)
        # designs in a shrinking neighbourhood of the incumbent, all scored on the SAME
        # adversarial sample so the fit sees the design dependence and not the shift
        designs = torch.stack([_feasible(ad, phi + delta * scale *
                                         (torch.rand(D, generator=g) * 2 - 1), lo, hi)
                               for _ in range(n_designs)])
        designs[0] = phi
        idx = torch.randint(x_adv.shape[0], (n_designs, per_design), generator=g)
        X = x_adv[idx]
        with torch.no_grad():
            Y = torch.stack([ad.loss(designs[i], X[i]) for i in range(n_designs)])
        model, opt = _fit(model, designs, X, Y, epochs, seed, link='identity', opt=opt)

        grad, hess = _derivatives(ad, model, phi, x_adv[:2000],
                                  need_hessian=(surrogate == 'taylor'))
        g_u = grad * scale
        if hess is not None:
            h_u = hess * scale.unsqueeze(1) * scale.unsqueeze(0)
            p_u, _, _ = trust_region_max(-g_u, -h_u, delta)
        else:
            p_u = -g_u / max(float(torch.linalg.norm(g_u)), 1e-12) * delta
        cand = _feasible(ad, phi + p_u * scale, lo, hi).detach()

        # accept on the worst case, which is the objective. The adversary is itself an
        # optimisation and needs its own graph, so it cannot run under no_grad.
        x_c = adversarial(kind, spec, problem, inner, flows, ad, cand, x_nom, g)
        with torch.no_grad():
            f_new, f_old = float(ad.risk(cand, x_c)), float(ad.risk(phi, x_adv))
        if f_new < f_old - 1e-9:
            phi, delta = cand, min(delta * 1.6, 1.0)
        else:
            delta = max(delta * 0.6, 5e-3)
        trace.append((r, f_old))
    return phi.detach(), trace


def held_out_shift(ad, phi, x, frac=1.0, k=24, seed=7):
    """The worst of k fixed directions, none of which any method trains against.

    A single random direction is too easy -- it costs the nominal design about one percent
    and separates nothing. Taking the worst over a fixed bundle of directions makes this a
    genuine unseen worst case while keeping it independent of every trained adversary: the
    directions come from one fixed seed and are the same for every design scored.
    """
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


def calibrate_radii(ad, spec, problem, inner, flows, phi, x, g, ratio=1.25, iters=9):
    """Radii that do equal damage, so the columns of the table can be read against each
    other.

    A KL radius is a divergence between weightings, a Wasserstein radius is a distance in
    input units and a latent radius is a divergence in latent space; nothing makes 0.3 of
    one comparable to 0.3 of another, and with arbitrary radii the table mostly reports
    which set happened to be given the bigger one. Each is bisected instead until it lifts
    the reference design's risk by the same fraction.
    """
    with torch.no_grad():
        base = float(ad.risk(phi, x))
    target, out = base + (ratio - 1.0) * abs(base), {}
    for kind, key, hi0 in (('kl', 'rho', 20.0), ('wasserstein', 'eps', 20.0),
                           ('latent', 'lat', 20.0), ('held-out', 'frac', 20.0)):
        lo, hi = 1e-4, hi0
        for _ in range(iters):
            mid = (lo * hi) ** 0.5
            spec[key] = mid
            if kind == 'held-out':
                with torch.no_grad():
                    v = float(ad.risk(phi, held_out_shift(ad, phi, x, frac=mid)))
            else:
                xa = adversarial(kind, spec, problem, inner, flows, ad, phi, x, g, steps=80)
                with torch.no_grad():
                    v = float(ad.risk(phi, xa))
            lo, hi = (mid, hi) if v < target else (lo, mid)
        spec[key] = hi
        out[kind] = hi
    print(f'  radii calibrated to +{100*(ratio-1):.0f}% risk ({base:.4f} -> {target:.4f}): '
          + '  '.join(f'{k}={v:.4g}' for k, v in out.items()))
    return out


def score(ad, spec, problem, inner, flows, phi, x_test, g):
    """The true risk of one design under every ambiguity set, at fixed radii."""
    with torch.no_grad():
        out = {'nominal': float(ad.risk(phi, x_test))}
    for kind in ('kl', 'wasserstein', 'latent'):
        xa = adversarial(kind, spec, problem, inner, flows, ad, phi, x_test, g, steps=120)
        with torch.no_grad():
            out[kind] = float(ad.risk(phi, xa))
    with torch.no_grad():
        xh = held_out_shift(ad, phi, x_test, frac=spec['frac'])
        out['held-out'] = float(ad.risk(phi, xh))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', default='two_moons', choices=sorted(PROBLEMS))
    ap.add_argument('--surrogate', nargs='+', default=['taylor', 'mlp'],
                    help="'mlp' is the branch-trunk (DeepONet) surrogate")
    ap.add_argument('--rounds', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)

    spec, problem, inner, flows = setup(a.problem)
    ad = Adapter(problem, spec['labelled'], spec['bounds'])
    g = torch.Generator().manual_seed(123)
    x_test = ad.sample(8000, g)
    # calibrate at the problem's own nominal optimum, which every method starts from
    phi_ref = (problem.solve(ad.sample(20000, g)) if hasattr(problem, 'solve')
               else problem.fit(*problem.sample(20000, g)))
    print(f'{a.problem}: design dim {ad.dim}')
    calibrate_radii(ad, spec, problem, inner, flows, phi_ref, x_test, g)

    cols = ['nominal', 'kl', 'wasserstein', 'latent', 'held-out']
    results = {}
    for surrogate in a.surrogate:
        for kind in ADVERSARIES:
            phi, trace = outer(ad, spec, problem, inner, flows, kind, surrogate,
                               rounds=a.rounds, seed=a.seed)
            results[(surrogate, kind)] = score(ad, spec, problem, inner, flows, phi,
                                               x_test, g)
            r = results[(surrogate, kind)]
            print(f'  {surrogate:7s} trained vs {kind:12s} ' +
                  '  '.join(f'{c}={r[c]:8.4f}' for c in cols))

    name = f'outer_{a.problem}.png'
    fig, axes = plt.subplots(1, len(a.surrogate), figsize=(7.2 * len(a.surrogate), 4.2),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    w = 0.2
    for ax, surrogate in zip(axes, a.surrogate):
        base = np.array([results[(surrogate, 'nominal')][c] for c in cols])
        for j, kind in enumerate(ADVERSARIES):
            v = np.array([results[(surrogate, kind)][c] for c in cols])
            ax.bar(np.arange(len(cols)) + (j - 1.5) * w, v, width=w, label=f'trained vs {kind}')
        ax.set(xticks=np.arange(len(cols)), xticklabels=cols,
               ylabel='true risk (lower is better)' if surrogate == a.surrogate[0] else None,
               title=f'{a.problem} -- {"Taylor (closed-form Hessian)" if surrogate == "taylor" else "branch-trunk (backprop)"}')
        ax.axhline(base[0], color='.4', lw=1, ls=':')
        ax.legend(fontsize=7)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=150)
    print(f'  wrote {OUT}/{name}')
