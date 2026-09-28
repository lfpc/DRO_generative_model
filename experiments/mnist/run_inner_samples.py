"""What does each adversary think MNIST looks like, at equal damage?

    python3 run_inner_samples.py

Every method is bisected to the SAME worst-case risk, so the comparison is about the shape
of the distribution each one reaches and not about radii, which are not commensurable
across ambiguity sets. The point of doing this on images rather than on the synthetic
benchmarks is that the answer can be looked at directly: an adversarial score decodes to a
picture, and a picture either is a handwritten digit or it is not.

What to expect, and what would refute the method. A transport ball moves each point in
whatever direction the loss rises fastest, which in a 64-dimensional feature space is almost
never the direction along which digits actually vary; its scores should decode to corrupted
digits. A KL ball over the sample cannot move anything at all -- it can only up-weight
images that are already hard, so it is confined to the training set by construction. The
latent ball can reach unseen images but every one of them is a push-forward of the fitted
flow, so it should decode to digits that are harder but still digits. If the latent scores
come out looking like noise, the manifold argument is wrong and this figure says so.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inner                                                       # noqa: E402
from dro.flows import load_flow                                    # noqa: E402
from problem import MNIST, CLASSES                                 # noqa: E402
from train_generator import Generator                              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
TARGET = 0.60                    # worst-case risk every method is calibrated to reach
NCOL = 6                         # sample images shown per digit


def calibrate(fn, target, lo, hi, iters=10):
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if fn(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    return hi, fn(hi)


def grid(problem, x, y, w=None, seed=0):
    """NCOL images per digit, as one array. With weights, the heaviest images -- which is
    all a reweighting adversary can offer, since it never creates a point."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for c in CLASSES:
        m = (y == c).nonzero().squeeze(1)
        if w is None:
            pick = m[torch.randperm(m.numel(), generator=g)[:NCOL]]
        else:
            pick = m[w[m].argsort(descending=True)[:NCOL]]
        rows.append(problem.to_image(x[pick]).reshape(-1, 28, 28))
    return torch.cat([torch.cat(list(r), 1) for r in rows], 0).clamp(0, 1).numpy()


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    os.makedirs(OUT, exist_ok=True)
    problem = MNIST()
    phi = torch.load(os.path.join(MODELS, 'design.pt'), weights_only=False)
    flows = Generator.load(os.path.join(MODELS, 'generator.pt')).flows
    judge = load_flow(torch.load(os.path.join(MODELS, 'judge.pt'), weights_only=False))

    x, y = problem.sample(8000, torch.Generator().manual_seed(0))
    nom = inner.nominal(problem, phi, x, y)
    print(f'nominal risk {nom:.4f}, accuracy {problem.accuracy(phi, x, y):.4f};  '
          f'calibrating every method to {TARGET}')

    r_kl, (v_kl, wt) = calibrate(lambda r: inner.kl_dro(problem, phi, x, y, r),
                                 TARGET, 1e-3, 20.0)
    r_w, (v_w, x_w) = calibrate(lambda e: inner.wasserstein_dro(problem, phi, x, y, e, steps=100),
                                TARGET, 1e-3, 20.0)
    r_l, (v_l, x_l, y_l) = calibrate(lambda r: inner.latent_dro(problem, flows, phi, r, family='affine', n=8000, steps=200),
                                     TARGET, 1e-3, 20.0)

    panels = [('nominal', x, y, None, nom, f'{r_kl:.3g}'),
              ('KL-DRO', x, y, wt, v_kl, f'$\\rho$={r_kl:.3g}'),
              ('Wasserstein', x_w, y, None, v_w, f'$\\epsilon$={r_w:.3g}'),
              ('Latent-DRO', x_l, y_l, None, v_l, f'$\\rho$={r_l:.3g}')]
    panels[0] = ('nominal', x, y, None, nom, '')

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 9.2),
                             gridspec_kw={'height_ratios': [3, 1]},
                             constrained_layout=True)
    stats = []
    for j, (name, xa, ya, w, v, r) in enumerate(panels):
        pl = inner.plausibility(judge, xa)
        if w is None:
            acc = problem.accuracy(phi, xa, ya)
            per = problem.loss(phi, xa, ya).detach()
            lw = None
        else:                       # a reweighting changes the measure, not the points
            acc = float(((problem.logits(phi, xa).argmax(1) == ya).to(torch.float32)
                         * w * len(w)).mean())
            per, lw = problem.loss(phi, xa, ya).detach(), w.numpy() * len(w)
        stats.append((name, r, v, acc, pl))
        axes[0, j].imshow(grid(problem, xa, ya, w), cmap='gray_r')
        axes[0, j].set(title=f'{name}  {r}\nrisk {v:.3f},  accuracy {acc:.3f},  '
                             f'judge log-density {pl:.1f}', xticks=[], yticks=[])
        axes[1, j].hist(per.numpy(), bins=np.linspace(0, 8, 70), weights=lw,
                        density=True, color='C0', alpha=.75)
        axes[1, j].set(xlabel='per-sample cross-entropy', yscale='log', yticks=[])
    fig.savefig(os.path.join(OUT, 'inner_samples.png'), dpi=140)

    print(f'\n  {"method":14s} {"radius":>10s} {"risk":>8s} {"accuracy":>10s} '
          f'{"judge log-density":>19s}')
    for name, r, v, acc, pl in stats:
        print(f'  {name:14s} {r:>10s} '
              f'{v:8.4f} {acc:10.4f} {pl:19.2f}')
    print(f'  wrote {OUT}/inner_samples.png')
