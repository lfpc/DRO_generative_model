"""Fit a normalizing flow to the return distribution and validate what it learned.

    python3 train_generator.py

Three things are recorded, because the training loss alone does not say whether the
generator is usable. The held-out loss says whether it is overfitting; KL(P || p_theta),
available exactly here because the return model is Gaussian, says how far the fitted
density actually sits from the truth -- which is the number the DRO layer cares about,
since the ambiguity radii are of the same order. Marginals are checked per asset at the
end, so an error in one of the ten cannot hide in an average.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dro.flows import Flow, load_flow                                        # noqa: E402
from problem import Portfolio                                     # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def train(problem, n_train=4000, epochs=800, batch=512, lr=3e-3, seed=0, every=10,
          patience=15):
    """Fit the flow, recording train loss, held-out loss and KL(P || p_theta).

    Early stopping on the held-out loss is not optional here. Trained to convergence this
    flow overfits rather than plateauing: the training loss keeps falling while the
    held-out loss rises and KL(P || p_theta) climbs from 0.05 to 0.18, so the generator
    the DRO layer would anchor to is three times worse than the one available at epoch 40.
    The training loss alone never shows it.
    """
    g = torch.Generator().manual_seed(seed)
    x = problem.sample(n_train, g)
    x_val = problem.sample(n_train // 4, g)
    x_kl = problem.sample(50000, torch.Generator().manual_seed(999))
    lp_true = problem.true_log_prob(x_kl)

    flow = Flow(problem.n_assets, n_layers=4, hidden=32, seed=seed)
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
    """The fitted flow behind the (latent_dim, model(z)) interface the samplers expect.

    One latent per asset and no discrete coordinate, so unlike the two-moons generator this
    is a plain diffeomorphism of a standard normal -- which is also why a quadrature rule
    over the latent costs m**n_assets here and prices itself out immediately.
    """

    def __init__(self, flow):
        self.flow, self.latent_dim = flow, flow.dim

    def __call__(self, z):
        return self.flow(z)[0]

    def save(self, path):
        torch.save({'flow': self.flow.state_dict(), 'flow_config': self.flow.config()}, path)

    @staticmethod
    def load(path):
        return Generator(load_flow(torch.load(path, weights_only=False)))


def plot_training(hist, ax_loss, ax_kl):
    ax_loss.plot(hist['epoch'], hist['train'], lw=1.5, label='train')
    ax_loss.plot(hist['epoch'], hist['val'], lw=1.5, ls='--', label='held out')
    ax_loss.set(xlabel='epoch', ylabel='negative log-likelihood', title='training loss')
    ax_loss.legend(fontsize=8)

    ax_kl.plot(hist['epoch'], hist['kl'], lw=1.8, color='C3')
    ax_kl.axhline(0, color='0.6', lw=.8, ls=':')
    stop = hist.get('stopped_at')
    if stop is not None:
        k_at = hist['kl'][hist['epoch'].index(min(hist['epoch'], key=lambda e: abs(e - stop)))]
        for ax in (ax_loss, ax_kl):
            ax.axvline(stop, color='0.4', lw=1.0, ls=':')
        ax_loss.text(stop, ax_loss.get_ylim()[1], ' kept', color='0.4', fontsize=7,
                     va='top')
        ax_kl.set(title=f'KL to the true model (kept {k_at:.4f})')
    else:
        ax_kl.set(title=f"KL to the true model (final {hist['kl'][-1]:.4f})")
    ax_kl.set(xlabel='epoch', ylabel=r'$\mathrm{KL}(P\,\|\,p_\theta)$', yscale='log')


def plot_marginals(problem, flow, n=50000):
    """One panel per asset: an error in any single marginal is visible, not averaged away."""
    real = problem.sample(n, torch.Generator().manual_seed(1))
    fake = flow.sample(n, torch.Generator().manual_seed(2))
    d = problem.n_assets
    fig, axes = plt.subplots(2, (d + 1) // 2, figsize=(2.3 * ((d + 1) // 2), 4.4),
                             constrained_layout=True)
    for i, ax in enumerate(axes.ravel()[:d]):
        lo, hi = float(real[:, i].min()), float(real[:, i].max())
        bins = torch.linspace(lo, hi, 60)
        ax.hist(real[:, i], bins=bins, density=True, histtype='stepfilled',
                color='C0', alpha=.35)
        ax.hist(fake[:, i], bins=bins, density=True, histtype='step', color='C1', lw=1.5)
        ax.set(title=f'asset {i + 1}   $\\sigma$={problem.sd[i]:.3f}', yticks=[])
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[d:]:
        ax.axis('off')
    fig.suptitle('marginals: true (filled) vs flow (line)', fontsize=11)
    fig.savefig(os.path.join(OUT, 'marginals.png'), dpi=150)
    print(f'  wrote {OUT}/marginals.png')


def plot_summary(problem, flow, hist, n=50000):
    real = problem.sample(n, torch.Generator().manual_seed(1))
    fake = flow.sample(n, torch.Generator().manual_seed(2))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), constrained_layout=True)
    plot_training(hist, axes[0], axes[1])

    err = torch.corrcoef(fake.T) - torch.corrcoef(real.T)
    im = axes[2].imshow(err, cmap='RdBu_r', vmin=-.15, vmax=.15)
    axes[2].set(title=f'correlation error (max {err.abs().max():.3f})')
    fig.colorbar(im, ax=axes[2], fraction=.046)

    w = torch.full((problem.n_assets,), 1 / problem.n_assets)
    q = torch.linspace(.001, .999, 200)
    axes[3].plot(q, torch.quantile(real @ w, q), lw=1.8, label='true')
    axes[3].plot(q, torch.quantile(fake @ w, q), lw=1.4, ls='--', label='flow')
    axes[3].set(xlabel='quantile', ylabel='portfolio return',
                title='equal-weight return quantiles')
    axes[3].legend(fontsize=8)
    fig.savefig(os.path.join(OUT, 'training.png'), dpi=150)
    print(f'  wrote {OUT}/training.png')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    os.makedirs(OUT, exist_ok=True)
    problem = Portfolio()
    flow, hist = train(problem)
    i = hist['epoch'].index(min(hist['epoch'], key=lambda e: abs(e - hist['stopped_at'])))
    print(f'  kept epoch {hist["stopped_at"]}: held out {hist["val"][i]:.3f}   '
          f'KL {hist["kl"][i]:.4f}   (last epoch would give KL {hist["kl"][-1]:.4f})')
    plot_summary(problem, flow, hist)
    plot_marginals(problem, flow)
    os.makedirs(MODELS, exist_ok=True)
    Generator(flow).save(os.path.join(MODELS, 'generator.pt'))
    torch.save(problem.sample(200000, torch.Generator().manual_seed(7)),
               os.path.join(MODELS, 'data.pt'))
    print(f'  saved generator and nominal sample to {MODELS}/')
