"""Render an animation of animation/trajectories.py's npz. Top left: the shield from above (x-z)
with the trajectories of context muons (sampled per p_z bin, faint) and of tracked muons, the same
in every frame, that hit in some design of the walk (opaque where they hit, with a dot on the
plane, half-transparent where they miss); each p_z bin has its colour, and a line's width
grows with log p_z. Bottom left: the surrogate's probability of a hit per (p_z, p_t) cell, and its
expected hits per cell with the simulation's hits on top (one point each; at ~1e-6 hits per muon,
1e8 simulated muons are too few for a per-cell probability). Right: the hits on the last
sensitive plane and Y along the walk. The designs play forward and back (--loop). The video is VP9
(WebM): this machine's ffmpeg has no working H.264, and MPEG-4 part 2 shows as a green screen in
many players.

    python -m animation.render outputs/results/animations/param_M7_dXOut.npz
    python -m animation.render <npz> --gif --fps 3

Also writes <name>.html: every design as a frame, chosen by hand with a slider (or the arrow
keys), self-contained (the frames inline as WebP), to open in any browser.
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
from shield import pretty_text

parser = cli.parser(__doc__)
parser.add_argument('npz', help='animation/trajectories.py output')
parser.add_argument('--fps', type=float, default=4)
parser.add_argument('--loop', action='store_true', default=True, help='play forward then back')
parser.add_argument('--no_loop', dest='loop', action='store_false')
parser.add_argument('--gif', action='store_true', help='a GIF instead of a WebM')
parser.add_argument('--no_html', dest='html', action='store_false',
                    help='no HTML page with a slider over the designs (one self-contained file, WebP frames)')
parser.add_argument('--html_quality', type=int, default=88, help='WebP quality of the HTML frames')
parser.add_argument('--cmap', default='viridis', help='sequential colour map of the p_z bins')
parser.add_argument('--map_cmap', default='Blues_r', help='sequential colour map of the two maps, light = high (its palest 15%% cut)')
parser.add_argument('--lw', type=float, nargs=2, default=[0.25, 1.2], help='line width at the lowest and highest p_z')
parser.add_argument('--alpha', type=float, nargs=3, default=[0.25, 0.9, 0.4], help='opacity of the context muons, of the tracked ones that hit and that miss')
parser.add_argument('--p_min', type=float, default=1e-8, help='lowest hit probability of its colour scale')
parser.add_argument('--h_decades', type=float, default=3, help='decades of the expected hits\' colour scale')
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


tpz = r['track_pz']
srt = np.argsort(r['ctx_pz'], kind='stable')             # the hard muons drawn on top
ctx_traj, ctx_pz, ctx_hit = r['ctx_traj'][:, srt], r['ctx_pz'][srt], r['ctx_hit'][:, srt]
IRON = '0.55'
PZ, PT = r['map_pz'], r['map_pt']
sur_p = np.where(r['sur_p'] > 0, r['sur_p'], np.nan)
sur_h = r['sur_hits'] * r['n_hits'] / 1e9                # expected hits per cell among the simulated muons
sur_h = np.where(sur_h > 0, sur_h, np.nan)
PNORM = LogNorm(args.p_min, np.nanmax(sur_p))
HNORM = LogNorm(np.nanmax(sur_h) / 10 ** args.h_decades, np.nanmax(sur_h))
_c = plt.get_cmap(args.map_cmap)
MCMAP = matplotlib.colors.LinearSegmentedColormap.from_list('map', _c(np.linspace(0, 0.85, 256)))
MCMAP.set_under(_c(0))
DOT = '#e8590c'                                          # the simulated hits: orange on the blues


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
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1], height_ratios=[1.1, 1], wspace=0.45, hspace=0.32,
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
ax_top.set_title(f'top view: context muons (faint) and {len(r["track_pz"])} muons that hit in some design '
                 '(opaque where they hit)',
                 fontsize=10, loc='left', pad=14)
ax_top.text(planes[-1][0], args.xmax * 0.92, 'sensitive\nplanes', ha='center', va='top', fontsize=8)
iron = ax_top.add_collection(PolyCollection([], facecolor=IRON, edgecolor='0.35', lw=0.4, zorder=1))
ctx_lines = ax_top.add_collection(LineCollection([], zorder=2))
hit_lines = ax_top.add_collection(LineCollection([], zorder=3))
ends = ax_top.scatter([], [], s=14, zorder=5, edgecolors='k', linewidths=0.3)

meshes = []
for ax, norm, lab in ((ax_sur, PNORM, 'surrogate: probability of a hit'),
                      (ax_sim, HNORM, 'expected hits (surrogate) and simulated hits')):
    ax.set_facecolor('white')
    meshes.append(ax.pcolormesh(PZ, PT, np.full((len(PT) - 1, len(PZ) - 1), np.nan), cmap=MCMAP, norm=norm,
                                rasterized=True))
    ax.set_xscale('log')
    ax.set_xlim(PZ[0], PZ[-1])
    ax.set_ylim(PT[0], PT[-1])
    ax.set_xlabel('$p_z$ [GeV]')
    ax.set_xticks([20, 50, 100, 200])
    ax.set_xticklabels(['20', '50', '100', '200'])
    ax.minorticks_off()
    ax.set_title(lab, fontsize=10, loc='left')
ax_sur.set_ylabel('$p_t$ [GeV]')
for m, ax, lab in ((meshes[0], ax_sur, 'P(hit) per muon'), (meshes[1], ax_sim, f'hits per cell (of {r["n_hits"]:.0e})')):
    cb = fig.colorbar(m, cax=ax.inset_axes([1.03, 0, 0.04, 1]), extend='min')
    cb.set_label(lab, fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=7)
sim_pts = ax_sim.scatter([], [], s=7, color=DOT, edgecolors='k', linewidths=0.3, zorder=3)
ax_sim.legend(handles=[Line2D([], [], ls='', marker='o', mfc=DOT, mec='k', ms=4, label='a simulated hit')],
              loc='upper right', fontsize=7, frameon=True, framealpha=0.8)

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
ax_y.set_xlabel(pretty_text(r['xlabel']))
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
    th = r['track_hit'][f]
    o = np.lexsort((tpz, th))                            # the hitting ones on top, the hard ones above
    hseg = segments(r['track_traj'][f][o])
    col = BIN_COL[bin_of(tpz[o])].copy()
    col[:, 3] = np.where(th[o], args.alpha[1], args.alpha[2])
    hit_lines.set_segments(hseg)
    hit_lines.set_color(col)
    hit_lines.set_linewidth(width(tpz[o]))
    hh = np.where(th[o])[0]
    ends.set_offsets(np.array([hseg[i][-1] for i in hh]).reshape(-1, 2))
    ends.set_facecolor(BIN_COL[bin_of(tpz[o][hh])] if len(hh) else 'k')
    meshes[0].set_array(sur_p[f].T.ravel())
    meshes[1].set_array(sur_h[f].T.ravel())
    hm = np.asarray(r['hits_xy'][f]).reshape(-1, 4)
    sim_pts.set_offsets(hm[:, 2:4])
    if 'mag_z' in r:
        for m, t in enumerate(labels):
            t.set_x(r['mag_z'][f, m].mean())
    xy = np.asarray(r['hits_xy'][f]).reshape(-1, 4)
    pts.set_offsets(xy[:, :2])
    pts.set_facecolor(BIN_COL[bin_of(xy[:, 2])])
    ax_plane.set_title(f'hits at z = {z_last:g} m: {len(xy)} of {r["n_hits"]:.0e} muons', fontsize=10)
    cur.set_data([s[f]], [Y[f]])
    title.set_text(f'{pretty_text(r["xlabel"])}: {s[f]:+.1f}     Y = {Y[f]:.0f} per $10^9$ muons')


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


def html_page(path):
    """One page: the frames inline (WebP), a slider over the designs, arrow keys, play/pause."""
    import base64
    import io
    import json

    from PIL import Image
    frames = []
    for f in range(F):
        draw(f)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=args.dpi)
        im, out = Image.open(io.BytesIO(buf.getvalue())).convert('RGB'), io.BytesIO()
        im.save(out, format='webp', quality=args.html_quality, method=6)
        frames.append('data:image/webp;base64,' + base64.b64encode(out.getvalue()).decode())
    labels = [f'{v:+.2f}' if 'dir' in os.path.basename(path) else f'{v:+.1f}%' for v in s]
    name = os.path.splitext(os.path.basename(path))[0]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name}</title>
<style>
  body {{ margin: 0; font: 15px system-ui, sans-serif; background: #fff; color: #222; }}
  .bar {{ position: sticky; top: 0; background: #fff; display: flex; align-items: center; gap: 14px;
          padding: 10px 16px; border-bottom: 1px solid #ddd; flex-wrap: wrap; }}
  input[type=range] {{ flex: 1; min-width: 200px; }}
  button {{ font: inherit; padding: 4px 12px; }}
  .val {{ font-variant-numeric: tabular-nums; min-width: 16ch; }}
  img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; }}
  .hint {{ color: #777; font-size: 13px; }}
</style></head><body>
<div class="bar">
  <b>{pretty_text(r['xlabel'], 'html')}</b>
  <button id="prev" title="previous (left arrow)">&#9664;</button>
  <input id="s" type="range" min="0" max="{F - 1}" step="1" value="{nom}">
  <button id="next" title="next (right arrow)">&#9654;</button>
  <span class="val" id="v"></span>
  <button id="play">play</button>
  <span class="hint">arrow keys step, Home = nominal</span>
</div>
<img id="im" alt="frame">
<script>
const F = {json.dumps(frames)};
const L = {json.dumps(labels)};
const Y = {json.dumps([round(float(y)) for y in Y])};
const s = document.getElementById('s'), im = document.getElementById('im'), v = document.getElementById('v');
function show(i) {{ i = Math.max(0, Math.min(F.length - 1, i)); s.value = i; im.src = F[i];
  v.textContent = L[i] + '   Y = ' + Y[i] + ' per 1e9'; }}
s.oninput = () => show(+s.value);
document.getElementById('prev').onclick = () => show(+s.value - 1);
document.getElementById('next').onclick = () => show(+s.value + 1);
let timer = null, dir = 1;
document.getElementById('play').onclick = (e) => {{
  if (timer) {{ clearInterval(timer); timer = null; e.target.textContent = 'play'; return; }}
  e.target.textContent = 'pause';
  timer = setInterval(() => {{ let i = +s.value + dir; if (i < 0 || i >= F.length) {{ dir = -dir; i = +s.value + dir; }} show(i); }}, {int(1000 / args.fps)});
}};
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') show(+s.value - 1);
  else if (e.key === 'ArrowRight') show(+s.value + 1);
  else if (e.key === 'Home') show({nom});
  else return;
  e.preventDefault();
}});
show({nom});
</script></body></html>
"""
    with open(path, 'w') as fh:
        fh.write(page)
    print(f'saved {path} ({F} designs, {os.path.getsize(path) / 1e6:.1f} MB)')


if args.html:
    html_page(out.rsplit('.', 1)[0] + '.html')
