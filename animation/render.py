"""Render an animation of animation/trajectories.py's npz: the shield from above (x-z) and from
the side (y-z) with the muons' trajectories, the hits on the sensitive plane, and Y along the
walk. Trajectories and hits are coloured by the muon's momentum at production (log scale);
muons that hit in the frame are drawn bold. The designs play forward and back (--loop). The
video is VP9 (WebM): this machine's ffmpeg has no working H.264, and MPEG-4 part 2 shows as a
green screen in many players.

    python -m animation.render outputs/results/animations/param_M7_dXOut.npz
    python -m animation.render <npz> --gif --fps 3

Needs hits_xy with the momentum (x, y, |p|): trajectories.py from 2026-11-24 on.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm
from scipy.spatial import ConvexHull

import cli
from paths import FIGURES

parser = cli.parser(__doc__)
parser.add_argument('npz', help='animation/trajectories.py output')
parser.add_argument('--fps', type=float, default=4)
parser.add_argument('--loop', action='store_true', default=True, help='play forward then back')
parser.add_argument('--no_loop', dest='loop', action='store_false')
parser.add_argument('--gif', action='store_true', help='a GIF instead of a WebM')
parser.add_argument('--p_range', type=float, nargs=2, default=[10, 400], help='momentum colour scale (GeV)')
parser.add_argument('--cmap', default='viridis')
parser.add_argument('--n_lost', type=int, default=200, help='drawn muons that do not hit (the first ones)')
parser.add_argument('--n_hit', type=int, default=40, help='drawn muons that hit the nominal design')
parser.add_argument('--xmax', type=float, default=5, help='half-width of the top view (m)')
parser.add_argument('--ymax', type=float, default=4, help='half-height of the side view (m)')
parser.add_argument('--zmax', type=float, default=93, help='end of the views (m)')
parser.add_argument('--dpi', type=int, default=120)
parser.add_argument('--out', default=os.path.join(FIGURES, 'animations'))
args = parser.parse_args()

r = np.load(args.npz, allow_pickle=True)
s, traj, hitf, mom, Y = r['s'], r['traj'], r['hit'], r['p'], r['Y']
F = len(s)
nom = int(np.argmin(np.abs(s)))
cand = r['candidate']
pick = np.r_[np.where(~cand)[0][:args.n_lost], np.where(cand & hitf[nom])[0][:args.n_hit]]
traj, hitf, mom = traj[:, pick], hitf[:, pick], mom[pick]
planes = r['planes']                                     # (position, dx, dy) of each sensitive plane
NORM = LogNorm(*args.p_range)
CMAP = matplotlib.colors.ListedColormap(plt.get_cmap(args.cmap)(np.linspace(0, 0.85, 256)))   # no pale yellow on white
COL = CMAP(NORM(mom))                                    # per drawn muon
IRON = '0.55'


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


fig = plt.figure(figsize=(14, 7.2))
gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], height_ratios=[1.35, 1], wspace=0.18, hspace=0.32)
ax_top, ax_side = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
ax_plane, ax_y = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

for ax, a, lim, lab in ((ax_top, 0, args.xmax, 'x [m]'), (ax_side, 1, args.ymax, 'y [m]')):
    ax.set_xlim(-3, args.zmax)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel(lab)
    for z, dx, dy in planes:
        h = (dx if a == 0 else dy) / 2
        ax.plot([z, z], [-h, h], color='k', lw=2.5, solid_capstyle='butt', zorder=4)
ax_top.set_title('top view', fontsize=10, loc='left', pad=14)
ax_side.set_title('side view', fontsize=10, loc='left')
ax_side.set_xlabel('z [m]')
ax_top.text(planes[-1][0], args.xmax * 0.92, 'sensitive\nplanes', ha='center', va='top', fontsize=8)
iron = [ax.add_collection(PolyCollection([], facecolor=IRON, edgecolor='0.35', lw=0.4, zorder=1)) for ax in (ax_top, ax_side)]
lines = [[ax.add_collection(LineCollection([], zorder=z)) for z in (2, 3)] for ax in (ax_top, ax_side)]
ends = [ax.scatter([], [], s=14, zorder=5) for ax in (ax_top, ax_side)]
ax_top.plot([], [], color='0.3', lw=0.5, alpha=0.5, label='lost')
ax_top.plot([], [], color='0.3', lw=1.8, label='hits the plane')
ax_top.legend(loc='lower right', fontsize=8, ncol=2, frameon=True)

z_last, dx, dy = planes[-1]
ax_plane.add_patch(plt.Rectangle((-dx / 2, -dy / 2), dx, dy, fill=False, color='k', lw=1.2))
ax_plane.set_xlim(-dx / 2 - 0.3, dx / 2 + 0.3)
ax_plane.set_ylim(-dy / 2 - 0.3, dy / 2 + 0.3)
ax_plane.set_aspect('equal')
ax_plane.set_xlabel('x [m]')
ax_plane.set_ylabel('y [m]')
pts = ax_plane.scatter([], [], s=7, c=[], cmap=CMAP, norm=NORM, alpha=0.8, lw=0)
cb = fig.colorbar(plt.cm.ScalarMappable(NORM, CMAP), ax=ax_plane, fraction=0.06, pad=0.04)
cb.set_ticks([t for t in (10, 20, 50, 100, 200, 400, 1000) if NORM.vmin <= t <= NORM.vmax])
cb.set_ticklabels([f'{t:g}' for t in cb.get_ticks()])
cb.minorticks_off()
cb.set_label(r'$|p|$ at production [GeV]' '\n(lines and points)')

ax_y.axvline(0, color='red', ls='--', lw=1.2, label='nominal')
ax_y.plot(s, Y, color='0.75', lw=1, zorder=1)
ax_y.scatter(s, Y, s=10, color='0.6', zorder=2)
cur, = ax_y.plot([], [], 'o', color='k', ms=8, zorder=3)
ax_y.set_xlabel(str(r['xlabel']))
ax_y.set_ylabel(r'$Y$ per $10^9$ muons')
ax_y.set_ylim(0, Y.max() * 1.15)
ax_y.legend(fontsize=8, frameon=False, loc='upper center')
title = fig.suptitle('', fontsize=13)
labels = [ax_top.text(0, args.xmax * 1.02, f'M{m}', ha='center', va='bottom', fontsize=8) for m in range(8)]


def draw(f):
    hit = hitf[f]
    for k, (ax, a) in enumerate(((ax_top, 0), (ax_side, 1))):
        iron[k].set_verts(outlines(r['blocks'][f], a))
        segs = [t[np.isfinite(t[:, 0])][:, [2, a]] for t in traj[f]]
        for j, sel in enumerate((~hit, hit)):
            idx = np.where(sel)[0]
            lines[k][j].set_segments([segs[i] for i in idx])
            lines[k][j].set_color(COL[idx])
            lines[k][j].set_linewidth(0.5 if j == 0 else 1.8)
            lines[k][j].set_alpha(0.45 if j == 0 else 1.0)
        last = np.array([sg[-1] if len(sg) else (np.nan, np.nan) for sg in segs])
        ends[k].set_offsets(last[hit] if hit.any() else np.empty((0, 2)))
        ends[k].set_color(COL[hit])
    if 'mag_z' in r:
        for m, t in enumerate(labels):
            t.set_x(r['mag_z'][f, m].mean())
    xy = np.asarray(r['hits_xy'][f]).reshape(-1, 3)
    pts.set_offsets(xy[:, :2])
    pts.set_array(xy[:, 2])
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
