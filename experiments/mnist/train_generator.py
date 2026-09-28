"""Fit a class-conditional flow to the MNIST features and validate what it learned.

    python3 train_generator.py

One flow per digit, so p(x, y) = p(y) p(x | y): an adversary then moves the latent of the
relevant digit's flow, which changes how a three is written and leaves it a three. That is
the semantics adversarial training assumes, and with ten classes it is the only sensible
factorisation -- a single unconditional flow would let the adversary quietly turn the label
distribution instead of the images.

Validation is where real data costs us something. Everywhere else in this suite the
generator is scored against an exact density; MNIST has none. Two substitutes are recorded:
held-out negative log-likelihood, which detects overfitting, and a SECOND unconditional flow
fitted on the 10,000 test images -- data neither the classifier nor the class-conditional
flows ever see -- which serves as the plausibility judge later on. That judge is a weaker
instrument than an exact density and is labelled as such wherever it is used.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dro.flows import Flow, load_flow                              # noqa: E402
from problem import MNIST, CLASSES                                 # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def _fit_one(x, dim, epochs, batch, lr, seed, every, patience, val_frac=0.15):
    g = torch.Generator().manual_seed(seed)
    n_val = int(val_frac * x.shape[0])
    perm = torch.randperm(x.shape[0], generator=g)
    xv, xt = x[perm[:n_val]], x[perm[n_val:]]
    flow = Flow(dim, n_layers=12, hidden=64, seed=seed,
                split='random', interleave_actnorm=True, scale_cap=0.25)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist, best = {'epoch': [], 'train': [], 'val': []}, (float('inf'), 0, None)
    for ep in range(epochs):
        idx = torch.randperm(xt.shape[0], generator=g)[:batch]
        loss = -flow.log_prob(xt[idx]).mean()
        opt.zero_grad()
        loss.backward()
        # A coupling flow on 784 pixels is not stable at the learning rate the low-
        # dimensional benchmarks use: without this clip one digit in ten diverges early and
        # early stopping then preserves a barely-trained checkpoint, which is invisible in
        # the training curve and obvious in the held-out likelihood.
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 20.0)
        opt.step()
        sched.step()
        if ep % every == 0 or ep == epochs - 1:
            with torch.no_grad():
                v = float(-flow.log_prob(xv).mean())
            hist['epoch'].append(ep)
            hist['train'].append(float(loss))
            hist['val'].append(v)
            if v < best[0] - 1e-3:
                best = (v, ep, {k: t.detach().clone() for k, t in flow.state_dict().items()})
            elif ep - best[1] >= patience * every:
                break
    flow.load_state_dict(best[2])
    flow.eval()
    hist['stopped_at'], hist['best'] = best[1], best[0]
    return flow, hist


def train(problem, epochs=2500, batch=256, lr=4e-4, seed=0, every=25, patience=12):
    """A flow per digit, each early-stopped on its own held-out likelihood.

    Small flows -- six couplings of width 64, under a million parameters. A wider one
    (eight couplings of width 256, 5.4M parameters) fits 784 pixels of a single digit from
    fewer than five thousand images by memorising them: the training loss falls while the
    held-out negative log-likelihood climbs past 10^5, and early stopping then preserves a
    barely-trained checkpoint. The capacity, not the learning rate or the preprocessing, is
    what decides this.
    """
    flows, hist = {}, {}
    for c in CLASSES:
        x = problem.features[problem.labels == c]
        flows[c], hist[c] = _fit_one(x, problem.input_dim, epochs, batch, lr, seed + c,
                                     every, patience)
        print(f'  digit {c}: {x.shape[0]:5,d} images, kept epoch {hist[c]["stopped_at"]:4d}, '
              f'held-out NLL {hist[c]["best"]:8.2f}')
    return flows, hist


def train_judge(problem, epochs=4000, batch=256, lr=4e-4, seed=99, every=25, patience=16):
    """The plausibility judge: one unconditional flow on the TEST features.

    Held out from everything else, so "how plausible is this adversarial score" is answered
    by a model that never saw the data the adversary was built from. It stands in for the
    exact density the synthetic benchmarks have, and it is a weaker instrument: the judge
    and the defendant are the same model class.
    """
    flow, h = _fit_one(problem.test_features, problem.input_dim, epochs, batch, lr, seed,
                       every, patience)
    print(f'  judge: kept epoch {h["stopped_at"]}, held-out NLL {h["best"]:.2f}')
    return flow, h


class Generator:
    """The ten class-conditional flows, saved and loaded as one object."""

    def __init__(self, flows):
        self.flows = flows
        self.latent_dim = next(iter(flows.values())).dim

    def __call__(self, z, y):
        x = torch.empty(z.shape[0], self.latent_dim)
        for c in CLASSES:
            m = y == c
            if m.any():
                x[m], _ = self.flows[c](z[m])
        return x

    def save(self, path):
        torch.save({k: v for c, f in self.flows.items() for k, v in
                    ((str(c), f.state_dict()), (str(c) + '_config', f.config()))}, path)

    @staticmethod
    def load(path):
        ck = torch.load(path, weights_only=False)
        return Generator({c: load_flow(ck, str(c)) for c in CLASSES})


def plot(problem, flows, hist, judge_hist):
    fig = plt.figure(figsize=(16, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 6)

    ax = fig.add_subplot(gs[0, 0])
    for c in CLASSES:
        ax.plot(hist[c]['epoch'], hist[c]['val'], lw=1.0, alpha=.8)
    ax.plot(judge_hist['epoch'], judge_hist['val'], lw=2.0, color='k', ls='--',
            label='judge (test set)')
    ax.set(xlabel='epoch', ylabel='held-out NLL', title='per-digit flows + judge')
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(CLASSES, [hist[c]['best'] for c in CLASSES], color='C0')
    ax.set(xlabel='digit', ylabel='held-out NLL', title='fit quality by digit',
           xticks=CLASSES)

    # pixel marginals: the two most variable pixels and the two least, the latter being
    # the near-degenerate background where an off-manifold adversary does its work
    xr = problem.features
    xf = torch.cat([flows[c].sample(2000, torch.Generator().manual_seed(c)) for c in CLASSES])
    sd = xr.std(0)
    picks = [int(sd.argsort(descending=True)[0]), int(sd.argsort(descending=True)[1]),
             int(sd.argsort()[0]), int(sd.argsort()[40])]
    for j, k in enumerate(picks):
        ax = fig.add_subplot(gs[0, 2 + j])
        bins = torch.linspace(float(xr[:, k].min()), float(xr[:, k].max()), 60)
        ax.hist(xr[:, k], bins=bins, density=True, histtype='stepfilled', alpha=.35)
        ax.hist(xf[:, k], bins=bins, density=True, histtype='step', lw=1.5, color='C3',
                ls='--')
        ax.set(title=f'pixel {k} (sd {float(sd[k]):.2f})', yticks=[])

    # what the generator actually produces, as images: one row per digit, side by side
    for col, (lab, src) in enumerate((('real', None), ('flow', flows))):
        ax = fig.add_subplot(gs[1:, 3 * col:3 * col + 3])
        tiles = []
        for c in CLASSES:
            x = (problem.features[problem.labels == c][:10] if src is None
                 else src[c].sample(10, torch.Generator().manual_seed(100 + c)))
            tiles.append(problem.to_image(x).reshape(-1, 28, 28))
        grid = torch.cat([torch.cat(list(t), 1) for t in tiles], 0)
        ax.imshow(grid.clamp(0, 1).numpy(), cmap='gray_r')
        ax.set(title=f'{lab} digits (784 pixels)',
               xticks=[], yticks=[])

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'generator.png'), dpi=140)
    print(f'  wrote {OUT}/generator.png')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    problem = MNIST()
    flows, hist = train(problem)
    judge, judge_hist = train_judge(problem)
    plot(problem, flows, hist, judge_hist)

    os.makedirs(MODELS, exist_ok=True)
    Generator(flows).save(os.path.join(MODELS, 'generator.pt'))
    torch.save({'flow': judge.state_dict(), 'flow_config': judge.config()},
               os.path.join(MODELS, 'judge.pt'))
    torch.save(problem.fit(), os.path.join(MODELS, 'design.pt'))
    print(f'  saved generator, judge and design to {MODELS}/')
