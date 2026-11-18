"""The hit probability per (p_z, p_t) bin of the muons, at the nominal design and one tolerance
unit along some of robustness.py's directions: which muons get through, and which a direction
lets through more.

On the surrogate, integrated over the muon quadrature grid (no sampling noise): per bin, p(hit)
= expected hits / muons. Writes <run>/momentum_grid.npz and, to the run's figure folder,
momentum_grid/ with one figure per panel (for slides) and one with all of them:
    phit_nominal, phit_<dir>      p(hit) per bin (log colour, the same scale for all)
    ratio_<dir>                   p(hit) at the direction / at the nominal design
    momentum_grid                 all of them

    python -m analysis.momentum_grid --dirs worst,act1
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.core import cached_hits, feature_cache, setup
import cli
from paths import RUN, figures, save_figure

parser = cli.add_common(cli.parser(__doc__))
parser.add_argument('--run', default=RUN, help='robustness.py --out_dir (for its directions)')
parser.add_argument('--dirs', default='worst,act1', help='directions of the run, walked to --t')
parser.add_argument('--t', type=float, default=1.0, help='step along each direction (1 = one tolerance unit)')
parser.add_argument('--pz_edges', default='0,20,30,45,75,125,175,225,300,400', help='GeV')
parser.add_argument('--pt_edges', default='0,0.25,0.5,0.75,1,1.5,2,3,13', help='GeV')
parser.add_argument('--cache_grid', default='201x64', help='muon points (muons.py)')
parser.add_argument('--n_ref', type=float, default=1e9, help='muons the hits are quoted per')
args = parser.parse_args()

PZ = torch.tensor([float(s) for s in args.pz_edges.split(',')])
PT = torch.tensor([float(s) for s in args.pt_edges.split(',')])
NZ, NT = len(PZ) - 1, len(PT) - 1


def bin_of(x):
    """(p_z, p_t) bin of raw muons (n, 7); muons outside the edges in the nearest edge bin."""
    pz, pt = x[:, 2], torch.hypot(x[:, 0], x[:, 1])
    iz = (torch.bucketize(pz, PZ.to(x.device)) - 1).clamp(0, NZ - 1)
    it = (torch.bucketize(pt, PT.to(x.device)) - 1).clamp(0, NT - 1)
    return iz * NT + it


shield, model = setup(args)
r = np.load(os.path.join(args.run, 'robustness.npz'))
names = [str(s) for s in r['dir_names']]
DIRS = args.dirs.split(',')
U = np.stack([np.zeros(model.dim)] + [np.clip(args.t * r['dir_vecs'][names.index(d)], -1, 1) for d in DIRS])
cache = feature_cache(model, shield.muons.points(args.cache_grid, args.n_ref), bin_of)
G = NZ * NT
hits = cached_hits(model, cache, G, U).reshape(len(U), NZ, NT)            # expected hits per bin
mu = torch.zeros(G, dtype=torch.float64, device=cache.w.device).index_add_(0, cache.g, cache.w.double())
mu = mu.cpu().numpy().reshape(NZ, NT)                                      # muons per bin
with np.errstate(invalid='ignore', divide='ignore'):
    P = hits / mu
out = os.path.join(args.run, 'momentum_grid.npz')
np.savez(out, pz_edges=PZ.numpy(), pt_edges=PT.numpy(), names=np.array(['nominal'] + DIRS), U=U, hits=hits,
         muons=mu, p_hit=P, t=args.t)
print(f'saved {out}')
H = hits.sum((1, 2))
for n, h in zip(['nominal'] + DIRS, H):
    print(f'{n:10s} Y/Y_0 = {h / H[0]:.3f}')

# =============================================================================
# figures
# =============================================================================
plt.rcParams.update({'font.size': 12})
FIG = os.path.join(figures(args.run), 'momentum_grid')
zl = [f'{a:g}' for a in PZ.tolist()]
tl = [f'{a:g}' for a in PT.tolist()]
filled = mu > 1e-6 * mu.sum()                                              # bins the flow has muons in
pos = P[:, filled][P[:, filled] > 0]
pnorm = matplotlib.colors.LogNorm(10 ** np.floor(np.log10(pos.min())), 10 ** np.ceil(np.log10(pos.max())))
R = P / P[0]
rmax = np.nanmax(np.abs(np.log10(R[1:, filled])))
rnorm = matplotlib.colors.LogNorm(10 ** -rmax, 10 ** rmax)


def fmt_p(v):
    m, e = f'{v:.1e}'.split('e')
    return f'{m}e{int(e)}'


def panel(ax, M, norm, cmap, txt, title):
    """One grid: p_z bins along x, p_t bins along y, each bin one cell (equal size, edges labelled)."""
    M = np.where(filled, M, np.nan)
    im = ax.pcolormesh(np.arange(NZ + 1), np.arange(NT + 1), np.ma.masked_invalid(M.T), norm=norm, cmap=cmap,
                       edgecolors='w', linewidth=1.5)
    for (i, j), v in np.ndenumerate(M):
        if np.isfinite(v) and v > 0:
            c = cmap(norm(v))
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            ax.text(i + 0.5, j + 0.5, txt(v), ha='center', va='center', fontsize=8.5, color='k' if lum > 0.5 else 'w')
    for (i, j) in zip(*np.nonzero(~filled)):
        ax.add_patch(matplotlib.patches.Rectangle((i, j), 1, 1, facecolor='0.95', edgecolor='w', lw=1.5))
    ax.set_xticks(np.arange(NZ + 1), zl)
    ax.set_yticks(np.arange(NT + 1), tl)
    ax.set(xlabel='$p_z$ [GeV]', ylabel='$p_t$ [GeV]', title=title)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def save(fig, name):
    save_figure(fig, FIG, name, dpi=200)
    fig.savefig(os.path.join(FIG, f'{name}.pdf'))
    plt.close(fig)
    print(f'saved {FIG}/{name}.png/.pdf')


LABEL = {'nominal': 'nominal design'}
TT = f'{args.t:g} tolerance unit' if args.t != 1 else 'one tolerance unit'
panels = [('phit_nominal', P[0], pnorm, plt.get_cmap('viridis'), fmt_p,
           f'$p$(hit) per muon, nominal design ($Y_0$ = {H[0]:.0f} per {args.n_ref:.0e})', 'p(hit)')]
for k, d in enumerate(DIRS, 1):
    panels.append((f'phit_{d}', P[k], pnorm, plt.get_cmap('viridis'), fmt_p,
                   f'$p$(hit) per muon, {d} at {TT} ($Y/Y_0$ = {H[k] / H[0]:.2f})', 'p(hit)'))
for k, d in enumerate(DIRS, 1):
    panels.append((f'ratio_{d}', R[k], rnorm, plt.get_cmap('RdBu_r'), lambda v: f'{v:.2f}' if v < 9.95 else f'{v:.0f}',
                   f'{d} at {TT} / nominal: $p$(hit) ratio ($Y/Y_0$ = {H[k] / H[0]:.2f})', 'ratio to nominal'))
os.makedirs(FIG, exist_ok=True)
for name, M, norm, cmap, txt, title, clab in panels:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    fig.colorbar(panel(ax, M, norm, cmap, txt, title), ax=ax, label=clab)
    save(fig, name)

nr = 1 + (len(DIRS) > 0)
fig, axes = plt.subplots(nr, 1 + len(DIRS), figsize=(8 * (1 + len(DIRS)), 4.8 * nr), constrained_layout=True,
                         squeeze=False)
for ax, (name, M, norm, cmap, txt, title, clab) in zip(axes[0], panels[:1 + len(DIRS)]):
    im = panel(ax, M, norm, cmap, txt, title)
fig.colorbar(im, ax=axes[0], label='p(hit)')
axes[1, 0].axis('off')
axes[1, 0].text(0.5, 0.5, 'ratio to nominal:\nred = more hits, blue = fewer\n\nnumbers per bin; grey: no muons',
                ha='center', va='center', fontsize=13, transform=axes[1, 0].transAxes)
for ax, (name, M, norm, cmap, txt, title, clab) in zip(axes[1, 1:], panels[1 + len(DIRS):]):
    im = panel(ax, M, norm, cmap, txt, title)
fig.colorbar(im, ax=axes[1], label='ratio to nominal')
save(fig, 'momentum_grid')
