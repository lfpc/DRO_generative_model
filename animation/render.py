"""Render an animation of animation/trajectories.py's npz. Top left: the shield from above (x-z)
with the trajectories of context muons (sampled per p_z bin, faint) and of a fixed fraction of the
hits (opaque, a dot where they reach the plane); each p_z bin has its colour, and a line's width
grows with log p_z. Bottom left: the probability of a hit per (p_z, p_t) cell, from the surrogate
and from the simulation (hits / muons of the cell; grey: no hit). Right: the hits on the last
sensitive plane and Y along the walk. The designs play forward and back (--loop). The video is VP9
(WebM): this machine's ffmpeg has no working H.264, and MPEG-4 part 2 shows as a green screen in
many players.

    python -m animation.render outputs/results/animations/param_M7_dXOut.npz
    python -m animation.render <npz> --gif --fps 3
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

import cli
from paths import FIGURES

parser = cli.parser(__doc__)
parser.add_argument('npz', help='animation/trajectories.py output')
parser.add_argument('--fps', type=float, default=4)
parser.add_argument('--loop', action='store_true', default=True, help='play forward then back')
parser.add_argument('--no_loop', dest='loop', action='store_false')
parser.add_argument('--gif', action='store_true', help='a GIF instead of a WebM')
parser.add_argument('--cmap', default='viridis', help='sequential colour map of the p_z bins')
parser.add_argument('--map_cmap', default='magma_r', help='colour map of the hit probability')
parser.add_argument('--lw', type=float, nargs=2, default=[0.25, 1.2], help='line width at the lowest and highest p_z')
parser.add_argument('--alpha', type=float, nargs=2, default=[0.25, 0.9], help='opacity of the context and of the hit trajectories')
parser.add_argument('--p_min', type=float, default=1e-8, help='lowest hit probability of the maps\' colour scale')
parser.add_argument('--xmax', type=float, default=5, help='half-width of the top view (m)')
parser.add_argument('--zmax', type=float, default=93, help='end of the top view (m)')
parser.add_argument('--dpi', type=int, default=120)
parser.add_argument('--out', default=os.path.join(FIGURES, 'animations'))
args = parser.parse_args()

r = np.load(args.npz, allow_pickle=True)
s, Y, edges = r['s'], r['Y'], r['edges']
F, K = len(s), len(edges) - 1
nom = int(np.argmin(np.abs(s)))
planes = r['planes']                                     # (position, dx, dy) of each sensitive plane
BIN_COL = plt.get_cmap(args.cmap)(np.linspace(0, 0.85, K))   # no pale yellow on white
lo_p, hi_p = np.log(edges[0]), np.log(edges[-2] if np.isinf(edges[-1]) else edges[-1])


def width(p):
    return args.lw[0] + (args.lw[1] - args.lw[0]) * np.clip((np.log(np.maximum(p, 1e-3)) - lo_p) / (hi_p - lo_p), 0, 1)


def bin_of(p):
    return np.clip(np.searchsorted(edges, p, side='right') - 1, 0, K - 1)


srt = np.argsort(r['ctx_pz'], kind='stable')             # the hard muons drawn on top
ctx_traj, ctx_pz, ctx_hit = r['ctx_traj'][:, srt], r['ctx_pz'][srt], r['ctx_hit'][:, srt]
IRON = '0.55'
PZ, PT = r['map_pz'], r['map_pt']
sur_p = r['sur_p']
with np.errstate(invalid='ignore', divide='ignore'):
    sim_p = r['sim_hits'] / np.where(r['sim_n'] > 0, r['sim_n'], np.nan)[None]
sim_p = np.where(sim_p > 0, sim_p, np.nan)               # no hit: grey
MNORM = LogNorm(args.p_min, np.nanmax([np.nanmax(sur_p), np.nanmax(sim_p)]))


def bin_label(k):
    a, b = edges[k], edges[k + 1]
    return f'{a:g}+ GeV' if np.isinf(b) else f'{a:g}-{b:g} GeV'


def outlines(blocks, a):
    """Projected outlines of the iron blocks on (z, axis a)."""
    out = []
    for b in blocks:
        pts = b[:, [2, a]]
        try:
            out.append(pts[ConvexHull(pts).vertices])
        except Exception:                               # a flat block
            out.append(pts)
    return out


fig = plt.figure(figsize=(15, 8.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1], height_ratios=[1.1, 1], wspace=0.4, hspace=0.32,
                      left=0.05, right=0.88, top=0.9, bottom=0.08)
ax_top = fig.add_subplot(gs[0, :2])
ax_sur, ax_sim = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
ax_plane, ax_y = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])

ax_top.set_xlim(-3, args.zmax)
ax_top.set_ylim(-args.xmax, args.xmax)
ax_top.set_ylabel('x [m]')
ax_top.set_xlabel('z [m]', labelpad=1)
for z, dx, dy in planes:
    ax_top.plot([z, z], [-dx / 2, dx / 2], color='k', lw=2.5, solid_capstyle='butt', zorder=4)
ax_top.set_title(f'top view: context muons (faint) and {100 * float(r["show"]):g}% of the hits (opaque)',
                 fontsize=10, loc='left', pad=14)
ax_top.text(planes[-1][0], args.xmax * 0.92, 'sensitive\nplanes', ha='center', va='top', fontsize=8)
iron = ax_top.add_collection(PolyCollection([], facecolor=IRON, edgecolor='0.35', lw=0.4, zorder=1))
ctx_lines = ax_top.add_collection(LineCollection([], zorder=2))
hit_lines = ax_top.add_collection(LineCollection([], zorder=3))
ends = ax_top.scatter([], [], s=14, zorder=5, edgecolors='k', linewidths=0.3)

meshes = []
for ax, lab in ((ax_sur, 'surrogate'), (ax_sim, 'simulation')):
    ax.set_facecolor('0.9')
    meshes.append(ax.pcolormesh(PZ, PT, np.full((len(PT) - 1, len(PZ) - 1), np.nan), cmap=args.map_cmap, norm=MNORM))
    ax.set_xscale('log')
    ax.set_xlabel('$p_z$ [GeV]')
    ax.set_xticks([20, 50, 100, 200, 400])
    ax.set_xticklabels(['20', '50', '100', '200', '400'])
    ax.minorticks_off()
    ax.set_title(f'P(hit) per cell: {lab}', fontsize=10, loc='left')
ax_sur.set_ylabel('$p_t$ [GeV]')
cb = fig.colorbar(meshes[1], cax=ax_sim.inset_axes([1.04, 0, 0.05, 1]))
cb.set_label('probability of a hit')

z_last, dx, dy = planes[-1]
ax_plane.add_patch(plt.Rectangle((-dx / 2, -dy / 2), dx, dy, fill=False, color='k', lw=1.2))
ax_plane.set_xlim(-dx / 2 - 0.3, dx / 2 + 0.3)
ax_plane.set_ylim(-dy / 2 - 0.3, dy / 2 + 0.3)
ax_plane.set_aspect('equal')
ax_plane.set_xlabel('x [m]')
ax_plane.set_ylabel('y [m]')
pts = ax_plane.scatter([], [], s=7, alpha=0.85, lw=0)
n_per = np.bincount(bin_of(ctx_pz), minlength=K)
handles = [Line2D([], [], color=BIN_COL[k], lw=width(np.sqrt(edges[k] * (edges[k + 1] if np.isfinite(edges[k + 1]) else 2 * edges[k]))),
                  label=f'{bin_label(k)} ({n_per[k]})') for k in range(K)]
handles.append(Line2D([], [], ls='', marker='o', mfc='0.7', mec='k', mew=0.3, ms=5, label='hits the plane'))
ax_plane.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.03, 1.02), fontsize=8, frameon=False,
                title='$p_z$ at production\n(context muons)', title_fontsize=8)

ax_y.axvline(0, color='red', ls='--', lw=1.2, label='nominal')
ax_y.plot(s, Y, color='0.75', lw=1, zorder=1)
ax_y.scatter(s, Y, s=10, color='0.6', zorder=2)
cur, = ax_y.plot([], [], 'o', color='k', ms=8, zorder=3)
ax_y.set_xlabel(str(r['xlabel']))
ax_y.set_ylabel(r'$Y$ per $10^9$ muons')
ax_y.yaxis.set_label_position('right')
ax_y.yaxis.tick_right()
ax_y.set_ylim(0, Y.max() * 1.15)
ax_y.legend(fontsize=8, frameon=False, loc='upper center')
title = fig.suptitle('', fontsize=13)
labels = [ax_top.text(0, args.xmax * 1.02, f'M{m}', ha='center', va='bottom', fontsize=8) for m in range(8)]


def segments(traj):
    return [t[np.isfinite(t[:, 0])][:, [2, 0]] for t in traj]


def draw(f):
    iron.set_verts(outlines(r['blocks'][f], 0))
    ctx_lines.set_segments(segments(ctx_traj[f]))
    ctx_lines.set_color(BIN_COL[bin_of(ctx_pz)])
    ctx_lines.set_linewidth(width(ctx_pz))
    ctx_lines.set_alpha(args.alpha[0])
    hp = np.asarray(r['hit_pz'][f])
    o = np.argsort(hp, kind='stable')
    hseg = segments(np.asarray(r['hit_traj'][f])[o]) if len(hp) else []
    hp = hp[o]
    hit_lines.set_segments(hseg)
    hit_lines.set_color(BIN_COL[bin_of(hp)] if len(hp) else 'k')
    hit_lines.set_linewidth(width(hp) if len(hp) else 1)
    hit_lines.set_alpha(args.alpha[1])
    last = np.array([sg[-1] for sg in hseg]) if hseg else np.empty((0, 2))
    ends.set_offsets(last)
    ends.set_facecolor(BIN_COL[bin_of(hp)] if len(hp) else 'k')
    meshes[0].set_array(sur_p[f].T.ravel())
    meshes[1].set_array(sim_p[f].T.ravel())
    if 'mag_z' in r:
        for m, t in enumerate(labels):
            t.set_x(r['mag_z'][f, m].mean())
    xy = np.asarray(r['hits_xy'][f]).reshape(-1, 4)
    pts.set_offsets(xy[:, :2])
    pts.set_facecolor(BIN_COL[bin_of(xy[:, 2])])
    ax_plane.set_title(f'hits at z = {z_last:g} m: {len(xy)} of {r["n_hits"]:.0e} muons', fontsize=10)
    cur.set_data([s[f]], [Y[f]])
    title.set_text(f'{r["xlabel"]}: {s[f]:+.1f}     Y = {Y[f]:.0f} per $10^9$ muons')


order = list(range(F)) + (list(range(F - 2, 0, -1)) if args.loop else [])
os.makedirs(args.out, exist_ok=True)
out = os.path.join(args.out, os.path.splitext(os.path.basename(args.npz))[0] + ('.gif' if args.gif else '.webm'))
if os.path.exists('/usr/bin/ffmpeg'):                   # the conda one has no libvpx
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
writer = PillowWriter(fps=args.fps) if args.gif else FFMpegWriter(
    fps=args.fps, codec='libvpx-vp9', extra_args=['-crf', '28', '-b:v', '0', '-pix_fmt', 'yuv420p', '-row-mt', '1'])
with writer.saving(fig, out, dpi=args.dpi):
    for f in order:
        draw(f)
        writer.grab_frame()
draw(nom)
fig.savefig(out.rsplit('.', 1)[0] + '_nominal.png', dpi=args.dpi)
print(f'saved {out} ({len(order)} frames) and the nominal frame as PNG')
