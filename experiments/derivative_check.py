"""Which parts of the gradient and Hessian at u = 0 are determined, and does the rest matter?

From the arrays run_shield.py saved (gradient and Hessian of the hits at u = 0, on the 1e9
nominal reference muons) of the best-trained models:

  1. the gradient component by component: consensus (median over models), spread, and how
     far the models agree once the k largest components are kept
  2. the Hessian: agreement of the top-k eigenvectors and eigenvalues
  3. what matters for robustness: each model's worst design in the +-10% box (quadratic
     model, trust region in u), its rise predicted by every other model

    python experiments/derivative_check.py        # -> outputs/figures/experiments/derivatives_*.png, outputs/results/experiments/*.txt

Run in the first analysis (43 parameters, muons on axis).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plots import COLOR, FIGS, LABEL
import run_shield

from analysis.core import trust_region_max
from paths import EXPERIMENTS
from shield import MuonShield

D = dict(run_shield.DEFAULTS)


def models():
    """[(name, kind, g, H)] of the models trained on the most data, and the existing one."""
    rows = run_shield.load_results()
    out = []
    for r in rows:
        c = r['config']
        top = c.get('budget') == max(run_shield.BUDGETS) and all(c.get(k) == v for k, v in D.items() if k not in ('kind', 'seed', 'budget'))
        if c['kind'] == 'existing' or (top and c['kind'] in ('deeponet', 'taylor', 'taylor:8', 'taylor-sub:4')):
            a = np.load(os.path.join(run_shield.OUT, 'arrays', f'{r["key"]}.npz'))
            out.append((f'{c["kind"]} s{c["seed"]}' if c['kind'] != 'existing' else 'existing', c['kind'], a['g'], a['H']))
    return sorted(out, key=lambda m: (m[1] == 'existing', m[0]))


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def top_vecs(H, k):
    ev, Q = np.linalg.eigh(H)
    i = np.argsort(-np.abs(ev))[:k]
    return ev[i], Q[:, i]


if __name__ == '__main__':
    M = models()
    new = [m for m in M if m[1] != 'existing']
    labels = MuonShield().labels
    G = np.array([m[2] for m in new])
    g_c = np.median(G, 0)
    order = np.argsort(-np.abs(g_c))
    lines = [f'{len(new)} models at 1.8e10 muons (+ the existing one); gradient of the hits at u = 0 '
             '(1e9 nominal muons), in hits per unit u (u = 1: +10%)', '']

    # 1. components
    lines.append('largest components of the consensus (median) gradient:  median, [min, max] over the new models, existing')
    for i in order[:15]:
        lines.append(f'  {labels[i]:16s} {g_c[i]:+8.1f}  [{G[:, i].min():+8.1f}, {G[:, i].max():+8.1f}]   {M[-1][2][i]:+8.1f}')
    share = np.cumsum(g_c[order] ** 2) / (g_c ** 2).sum()
    lines.append(f'share of |g|^2 in the top 3 / 5 / 10 components: {share[2]:.2f} / {share[4]:.2f} / {share[9]:.2f}')
    ks = [1, 2, 3, 5, 8, 10, 15, 20, 43]
    lines.append('\ncos with the consensus, keeping the k largest consensus components:')
    lines.append('  ' + f'{"k":>16s} ' + ' '.join(f'{k:>6d}' for k in ks))
    cos_k = {}
    for name, kind, g, _ in M:
        cos_k[name] = [cos(g[order[:k]], g_c[order[:k]]) if k > 1 else float(np.sign(g[order[0]] * g_c[order[0]])) for k in ks]
        lines.append(f'  {name:>16s} ' + ' '.join(f'{v:6.3f}' for v in cos_k[name]))

    # 2. Hessian
    lines.append('\nHessian: top eigenvalues (by |value|), hits per unit u^2')
    for name, kind, g, H in M:
        ev, _ = top_vecs(H, 5)
        lines.append(f'  {name:>16s} ' + ' '.join(f'{v:+8.1f}' for v in ev))
    _, Qc = top_vecs(np.median([m[3] for m in new], 0), 5)
    lines.append('|cos| of each model\'s top-k eigenvectors with the consensus Hessian\'s (k = 1..5, matched in order):')
    for name, kind, g, H in M:
        _, Q = top_vecs(H, 5)
        lines.append(f'  {name:>16s} ' + ' '.join(f'{abs(Q[:, j] @ Qc[:, j]):6.3f}' for j in range(5)))

    # 3. robustness: worst design in the box (|u|_2 <= 1 ball and the walk to the box face)
    lines.append('\nworst design of each model (trust region, |u| <= 1) and its rise [hits] as predicted by every model (quadratic):')
    V = {}
    for name, kind, g, H in M:
        V[name] = trust_region_max(g, H, 1.0)
    names = [m[0] for m in M]
    lines.append('  ' + f'{"design of / model":>18s} ' + ' '.join(f'{n[:10]:>10s}' for n in names))
    R = np.zeros((len(M), len(M)))
    for i, (ni, _, _, _) in enumerate(M):
        for j, (nj, _, g, H) in enumerate(M):
            v = V[ni]
            R[i, j] = g @ v + 0.5 * v @ H @ v
        lines.append(f'  {ni:>18s} ' + ' '.join(f'{x:10.0f}' for x in R[i]))
    eff = R / np.diag(R)[None, :]
    lines.append('as a fraction of each model\'s own worst rise (column-normalized); off-diagonal median '
                 f'{np.median(eff[~np.eye(len(M), dtype=bool)]):.3f}, min {eff[~np.eye(len(M), dtype=bool)].min():.3f}')
    lines.append('cos between the worst directions:')
    for i, ni in enumerate(names):
        lines.append(f'  {ni:>16s} ' + ' '.join(f'{cos(V[ni], V[nj]):6.2f}' for nj in names))
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(EXPERIMENTS, exist_ok=True)
    with open(os.path.join(EXPERIMENTS, 'derivatives_check.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    np.savez(os.path.join(EXPERIMENTS, 'derivatives_check.npz'), names=names, V=np.array([V[n] for n in names]), R=R,
             g=np.array([m[2] for m in M]), H=np.array([m[3] for m in M]), order=order)

    # figure: components, and the k-truncated agreement
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), constrained_layout=True, gridspec_kw=dict(width_ratios=(2.2, 1)))
    x = np.arange(20)
    for n, (name, kind, g, _) in enumerate(M):
        ax[0].plot(x + (n - len(M) / 2) * 0.05, g[order[:20]], 'o', ms=4, color=COLOR[kind], mfc='none' if kind == 'existing' else COLOR[kind],
                   label=LABEL[kind] if name.endswith('s0') or kind == 'existing' else None)
        ax[1].plot(ks, cos_k[name], '-o', ms=3, lw=1.5, color=COLOR[kind])
    ax[0].axhline(0, color='0.7', lw=0.8)
    ax[0].set_xticks(x, [labels[i] for i in order[:20]], rotation=60, fontsize=7)
    ax[0].set(ylabel='dH/du_i at u = 0 [hits per +10%]', title='gradient, largest 20 components (consensus order)')
    ax[0].legend(fontsize=7)
    ax[1].set(xscale='log', xlabel='k largest components kept', ylabel='cos with the consensus',
              title='agreement of the leading components', ylim=(0, 1.02))
    fig.savefig(os.path.join(FIGS, 'derivatives_components.png'), dpi=150)
    print('saved derivatives_components.png, derivatives_check.txt')
