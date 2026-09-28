"""Fit a class-conditional flow to the two-moons inputs and validate it.

    python3 train_generator.py

One flow per class, so p(x, y) = p(y) p(x | y) with p(y) = 1/2. An adversary then moves
the latent of the relevant flow, which moves x and leaves the label alone -- the semantics
adversarial training assumes.

Recorded during training: the training and held-out negative log-likelihood, and
KL(P || p_theta) against the quadrature density. The last is the one that matters, and it
is the one that reveals overfitting: the training loss falls throughout while the fitted
density drifts away from the truth.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dro.flows import Flow, load_flow                             # noqa: E402
from problem import TwoMoons                                      # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
CLASSES = (1.0, -1.0)


def train(problem, n_train=4000, epochs=1500, batch=512, lr=3e-3, seed=0, every=10,
          patience=15):
    """A flow per class, early-stopped on held-out likelihood."""
    g = torch.Generator().manual_seed(seed)
    x, y = problem.sample(n_train, g)
    xv, yv = problem.sample(n_train // 4, g)
    xk, yk = problem.sample(40000, torch.Generator().manual_seed(999))
    lp_true = problem.log_prob(xk, yk)

    flows, hist = {}, {'epoch': [], 'train': [], 'val': [], 'kl': []}
    for s in CLASSES:
        flows[s] = Flow(2, n_layers=6, hidden=64, seed=seed)
    opts = {s: torch.optim.Adam(flows[s].parameters(), lr=lr) for s in CLASSES}
    scheds = {s: torch.optim.lr_scheduler.CosineAnnealingLR(opts[s], T_max=epochs)
              for s in CLASSES}
    best = (float('inf'), 0, None)

    for ep in range(epochs):
        tr = 0.0
        for s in CLASSES:
            xs = x[y == s]
            idx = torch.randperm(xs.shape[0], generator=g)[:batch]
            loss = -flows[s].log_prob(xs[idx]).mean()
            opts[s].zero_grad()
            loss.backward()
            opts[s].step()
            scheds[s].step()
            tr += float(loss) / len(CLASSES)
        if ep % every == 0 or ep == epochs - 1:
            with torch.no_grad():
                val = sum(float(-flows[s].log_prob(xv[yv == s]).mean())
                          for s in CLASSES) / len(CLASSES)
                lp_hat = torch.where(yk > 0, flows[1.0].log_prob(xk),
                                     flows[-1.0].log_prob(xk))
                hist['epoch'].append(ep)
                hist['train'].append(tr)
                hist['val'].append(val)
                hist['kl'].append(float((lp_true - lp_hat).mean()))
            if val < best[0] - 1e-4:
                best = (val, ep, {s: {k: v.detach().clone()
                                      for k, v in flows[s].state_dict().items()}
                                  for s in CLASSES})
            elif ep - best[1] >= patience * every:
                break
    for s in CLASSES:
        flows[s].load_state_dict(best[2][s])
        flows[s].eval()
    hist['stopped_at'] = best[1]
    return flows, hist


class Generator:
    """The two class-conditional flows as one map of a standard normal.

    Three latent coordinates: the first two are the input, the sign of the third picks the
    class. Keeping the class inside the latent -- rather than drawing it separately -- is
    what lets a single low-discrepancy sequence drive the whole generator, and it gives the
    (latent_dim, model(z)) interface that the samplers elsewhere expect.
    """

    latent_dim = 3

    def __init__(self, flows):
        self.flows = flows

    def __call__(self, z):
        y = torch.where(z[:, 2] > 0, 1.0, -1.0)
        x = torch.empty(z.shape[0], 2)
        for s in CLASSES:
            x[y == s], _ = self.flows[s](z[y == s, :2])
        return torch.cat([x, y.unsqueeze(1)], 1)

    def save(self, path):
        torch.save({k: v for s, f in self.flows.items() for k, v in
                    ((str(s), f.state_dict()), (str(s) + '_config', f.config()))}, path)

    @staticmethod
    def load(path):
        ck = torch.load(path, weights_only=False)
        return Generator({s: load_flow(ck, str(s)) for s in CLASSES})


def sample_flows(flows, n, seed=2):
    g = torch.Generator().manual_seed(seed)
    y = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    x = torch.empty(n, 2)
    for s in CLASSES:
        m = y == s
        x[m] = flows[s].sample(int(m.sum()), g)
    return x, y


def plot(problem, flows, hist, n=20000):
    xr, yr = problem.sample(n, torch.Generator().manual_seed(1))
    xf, yf = sample_flows(flows, n)
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.2), constrained_layout=True)

    axes[0].plot(hist['epoch'], hist['train'], lw=1.4, label='train')
    axes[0].plot(hist['epoch'], hist['val'], lw=1.4, ls='--', label='held out')
    axes[0].axvline(hist['stopped_at'], color='.4', lw=1, ls=':')
    axes[0].set(xlabel='epoch', ylabel='negative log-likelihood', title='training loss')
    axes[0].legend(fontsize=8)

    i = hist['epoch'].index(min(hist['epoch'], key=lambda e: abs(e - hist['stopped_at'])))
    axes[1].plot(hist['epoch'], hist['kl'], lw=1.8, color='C3')
    axes[1].axvline(hist['stopped_at'], color='.4', lw=1, ls=':')
    axes[1].set(xlabel='epoch', ylabel=r'$\mathrm{KL}(P\,\|\,p_\theta)$', yscale='log',
                title=f'KL to the true model (kept {hist["kl"][i]:.4f})')

    for ax, (xx, yy), lab in [(axes[2], (xr, yr), 'true'), (axes[3], (xf, yf), 'flow')]:
        for s, c in zip(CLASSES, ('C0', 'C1')):
            m = yy == s
            ax.scatter(xx[m, 0][:4000], xx[m, 1][:4000], s=3, alpha=.3, c=c)
        ax.set(title=f'{lab} samples', xlim=(-2, 3), ylim=(-1.2, 1.7))

    for j, name in enumerate(('$x_1$', '$x_2$')):     # marginals, per coordinate
        bins = torch.linspace(float(xr[:, j].min()), float(xr[:, j].max()), 70)
        axes[4].hist(xr[:, j], bins=bins, density=True, histtype='stepfilled',
                     alpha=.35, color=f'C{j}', label=f'{name} true')
        axes[4].hist(xf[:, j], bins=bins, density=True, histtype='step', lw=1.5,
                     color=f'C{j}', ls='--', label=f'{name} flow')
    axes[4].set(title='marginals', yticks=[])
    axes[4].legend(fontsize=7)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'generator.png'), dpi=150)
    print(f'  wrote {OUT}/generator.png')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    problem = TwoMoons()
    flows, hist = train(problem)
    i = hist['epoch'].index(min(hist['epoch'], key=lambda e: abs(e - hist['stopped_at'])))
    print(f'  kept epoch {hist["stopped_at"]}: held out {hist["val"][i]:.3f}   '
          f'KL {hist["kl"][i]:.4f}   (last epoch would give {hist["kl"][-1]:.4f})')
    plot(problem, flows, hist)
    os.makedirs(MODELS, exist_ok=True)
    Generator(flows).save(os.path.join(MODELS, 'generator.pt'))
    torch.save(problem.sample(200000, torch.Generator().manual_seed(7)),
               os.path.join(MODELS, 'data.pt'))
    print(f'  saved generator and nominal sample to {MODELS}/')
