"""Fit a normalizing flow to the input distribution and validate what it learned.

    python3 train_generator.py

Three numbers are recorded, because the training loss alone does not say whether the
generator is usable: the held-out loss says whether it is overfitting, KL(P || p_theta)
says how far the fitted density sits from the truth -- the quantity the DRO layer cares
about, since the ambiguity radii are of the same order -- and the per-variable marginals at
the end make sure an error in one of the 8 cannot hide in an average.

The demand clusters are well separated, so an overfitted flow does not blur -- it invents mass in the gaps between regimes, which is precisely the failure the ambiguity set is supposed to rule out.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dro.flows import Flow, load_flow                              # noqa: E402
from problem import Newsvendor                                  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def train(problem, n_train=60000, epochs=4000, batch=512, lr=3e-3, seed=0, every=10,
          patience=25):
    """Fit the flow, early-stopped on held-out likelihood.

    The sample size matters more than the capacity here: at 6,000 training points this
    flow stops at a KL several times worse than at 60,000, while widening or deepening
    it at the smaller sample makes matters worse rather than better. The radii the DRO
    layer uses are of the same order as that KL, so the difference is not cosmetic.
    """
    g = torch.Generator().manual_seed(seed)
    x = problem.sample(n_train, g)
    x_val = problem.sample(n_train // 4, g)
    x_kl = problem.sample(50000, torch.Generator().manual_seed(999))
    lp_true = problem.true_log_prob(x_kl)

    flow = Flow(problem.input_dim, n_layers=10, hidden=128, seed=seed)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = {'epoch': [], 'train': [], 'val': [], 'kl': []}
    best = (float('inf'), 0, None)

    for ep in range(epochs):
        idx = torch.randperm(n_train, generator=g)[:batch]
        loss = -flow.log_prob(x[idx]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if ep % every == 0 or ep == epochs - 1:
            with torch.no_grad():
                val = float(-flow.log_prob(x_val).mean())
                hist['epoch'].append(ep)
                hist['train'].append(float(loss))
                hist['val'].append(val)
                hist['kl'].append(float((lp_true - flow.log_prob(x_kl)).mean()))
            if val < best[0] - 1e-4:
                best = (val, ep, {k: v.detach().clone()
                                  for k, v in flow.state_dict().items()})
            elif ep - best[1] >= patience * every:
                break
    if best[2] is not None:
        flow.load_state_dict(best[2])
    hist['stopped_at'] = best[1]
    flow.eval()
    return flow, hist


class Generator:
    """The fitted flow behind the (latent_dim, model(z)) interface the samplers expect."""

    def __init__(self, flow):
        self.flow, self.latent_dim = flow, flow.dim

    def __call__(self, z):
        return self.flow(z)[0]

    def save(self, path):
        torch.save({'flow': self.flow.state_dict(), 'flow_config': self.flow.config()}, path)

    @staticmethod
    def load(path):
        return Generator(load_flow(torch.load(path, weights_only=False)))


def plot(problem, flow, hist, n=50000):
    xr = problem.sample(n, torch.Generator().manual_seed(1))
    xf = flow.sample(n, torch.Generator().manual_seed(2))
    d = problem.input_dim
    ncol = min(d, 6)
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol, 5.4), constrained_layout=True)

    axes[0, 0].plot(hist['epoch'], hist['train'], lw=1.4, label='train')
    axes[0, 0].plot(hist['epoch'], hist['val'], lw=1.4, ls='--', label='held out')
    axes[0, 0].axvline(hist['stopped_at'], color='.4', lw=1, ls=':')
    axes[0, 0].set(xlabel='epoch', ylabel='negative log-likelihood', title='training loss')
    axes[0, 0].legend(fontsize=8)

    i = hist['epoch'].index(min(hist['epoch'], key=lambda e: abs(e - hist['stopped_at'])))
    axes[0, 1].plot(hist['epoch'], hist['kl'], lw=1.8, color='C3')
    axes[0, 1].axvline(hist['stopped_at'], color='.4', lw=1, ls=':')
    axes[0, 1].set(xlabel='epoch', ylabel=r'$\mathrm{KL}(P\,\|\,p_\theta)$',
                   title=f'KL to the true model (kept {hist["kl"][i]:.4f})')

    # a pair plot of the two most variable coordinates says whether the joint is right,
    # which the marginals below cannot
    v = xr.var(0)
    a, b = int(v.argmax()), int(v.argsort()[-2])
    for ax, (xx, lab) in zip(axes[0, 2:4], ((xr, 'true'), (xf, 'flow'))):
        ax.scatter(xx[:4000, a], xx[:4000, b], s=3, alpha=.25)
        ax.set(title=f'{lab}: coords {a}, {b}')
    for ax in axes[0, 4:]:
        ax.axis('off')

    for j in range(ncol):
        k = j * max(1, d // ncol)
        ax = axes[1, j]
        lo, hi = float(xr[:, k].min()), float(xr[:, k].max())
        bins = torch.linspace(lo, hi, 60)
        ax.hist(xr[:, k], bins=bins, density=True, histtype='stepfilled', alpha=.35)
        ax.hist(xf[:, k], bins=bins, density=True, histtype='step', lw=1.5, ls='--',
                color='C3')
        ax.set(title=f'demand {k}', yticks=[])

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'generator.png'), dpi=150)
    print(f'  wrote {OUT}/generator.png')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    problem = Newsvendor()
    flow, hist = train(problem)
    i = hist['epoch'].index(min(hist['epoch'], key=lambda e: abs(e - hist['stopped_at'])))
    print(f'  kept epoch {hist["stopped_at"]}: held out {hist["val"][i]:.3f}   '
          f'KL {hist["kl"][i]:.4f}   (last epoch would give KL {hist["kl"][-1]:.4f})')
    plot(problem, flow, hist)
    os.makedirs(MODELS, exist_ok=True)
    Generator(flow).save(os.path.join(MODELS, 'generator.pt'))
    torch.save(problem.sample(200000, torch.Generator().manual_seed(7)),
               os.path.join(MODELS, 'data.pt'))
    print(f'  saved generator and nominal sample to {MODELS}/')
