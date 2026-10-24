"""The true worst case for a given tolerance, and a tolerance allocation that meets a target.

worst_cases.py scales the worst *direction* found at one tolerance unit down to a maximum change
of X%; the worst design in the +-X% box can be worse. Here, on the surrogate (integrated over the
muon flow on the cached grid):

  box worst@X%   projected-gradient ascent of the hits over the box |phi_i/phi_0,i - 1| <= X%,
                 from several starts (sign of the gradient, the scaled worst direction, random
                 corners), batched
  allocation     greedy: start with every parameter at +-10%; repeatedly find the box's worst
                 design u*, and tighten the --k_step parameters that contribute most to it
                 (first-order share of log H: u*_i d log H / du_i) by --shrink, down to --floor,
                 until the worst case is below the smallest --targets. The first box meeting each
                 target is that target's allocation; its worst design is simulated to check it.

Errors on the surrogate: the members' standard error at the found designs. --sim simulates the
found designs on the sampled muons of robustness.py (seed 0), split over --n_workers GPUs; the
report includes whichever simulations exist. The search is saved and reused unless --redo.

    python tolerance_boxes.py                                       # search, report (surrogate)
    CUDA_VISIBLE_DEVICES=0 python tolerance_boxes.py --sim --worker 0 --n_workers 2 &
    CUDA_VISIBLE_DEVICES=3 python tolerance_boxes.py --sim --worker 1 --n_workers 2 &
    python tolerance_boxes.py                                       # report with the simulations

Writes <run>/tolerance_boxes.npz, tolerance_sim_<worker>.npz and tolerance_boxes_report.txt;
plots.py draws tolerance_boxes.png and tolerance_allocation.png.
"""
import argparse
import glob
import os

import numpy as np
import torch

from analysis import (ArgFormatter, Report, cached_hits, feature_cache, grid_spec, muon_blocks, one_group,
                      seed_se, setup, simulate)
from lcso.surrogate import load_members

parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument('--run', default='outputs/robustness', help='robustness.py --out_dir (for its worst direction)')
parser.add_argument('--model', default='outputs/lcso.pt')
parser.add_argument('--cache_grid', default='201x64', help='quadrature grid: latent points per side x azimuths')
parser.add_argument('--n_ref', type=float, default=1e9, help='hits are quoted per this many muons')
parser.add_argument('--pcts', default='1,2,3,5,10', help='box half-widths X (%%)')
parser.add_argument('--restarts', type=int, default=16, help='starts per box search')
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.05, help='Adam step, in units of the box half-width')
parser.add_argument('--targets', default='1.25,1.5,2,3', help='allocation targets for the worst H/H_0')
parser.add_argument('--k_step', type=int, default=2, help='parameters tightened per allocation step')
parser.add_argument('--shrink', type=float, default=0.8, help='factor per tightening')
parser.add_argument('--floor', type=float, default=0.5, help='tightest tolerance allowed (%%)')
parser.add_argument('--max_iter', type=int, default=400)
parser.add_argument('--redo', action='store_true', help='search again even if tolerance_boxes.npz exists')
parser.add_argument('--sim', action='store_true', help='simulate this worker\'s share of the designs and exit')
parser.add_argument('--n_muons', type=int, default=int(1e9), help='sampled muons for the simulator')
parser.add_argument('--seed', type=int, default=0, help='0 = the muons of robustness.py and worst_cases.py')
parser.add_argument('--worker', type=int, default=0)
parser.add_argument('--n_workers', type=int, default=1)
parser.add_argument('--block', type=int, default=int(2e7), help='sampled muons held in memory')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

OUT = os.path.join(args.run, 'tolerance_boxes.npz')
shield, model = setup(args.model, args.device)
D, labels = model.dim, shield.labels

if args.sim:
    t = np.load(OUT)
    jobs = list(range(len(t['U'])))[args.worker::args.n_workers]
    U = np.concatenate([np.zeros((1, D)), t['U'][jobs]])          # row 0: the nominal design, per worker
    print(f'worker {args.worker}: {len(jobs)} designs + the nominal over {args.n_muons:,} muons')
    hits = simulate(model, shield, muon_blocks(shield, args.n_muons, args.block, args.seed), U, one_group, 1)[:, 0]
    out = os.path.join(args.run, f'tolerance_sim_{args.worker}.npz')
    np.savez(out, idx=np.array(jobs), hits=hits[1:], nominal=hits[0], n_muons=args.n_muons)
    print(f'saved {out}')
    raise SystemExit

class BoxSearch:
    """The surrogate's hits over the cached quadrature grid, differentiable in the design: the
    worst design in a box (box_worst) and what each parameter contributes to it (shares)."""

    def __init__(self, model, cache):
        self.model, self.cache = model, cache
        self.T, self.W = cache.t.float(), cache.w
        self.H0 = float(cached_hits(model, cache, 1, np.zeros((1, model.dim)))[0, 0])

    def log_hits(self, U):
        """log H(u)/H_0 of a batch of designs (B, D), differentiable."""
        b = self.model.branch_net(U)                               # (B, P)
        h = sum((torch.sigmoid(self.T[c:c + 2 ** 20] @ b.T + self.model.bias).double()
                 * self.W[c:c + 2 ** 20, None]).sum(0) for c in range(0, len(self.T), 2 ** 20))
        return torch.log(h / self.H0)

    def box_worst(self, e, starts, warm=None, seed=0):
        """Worst design in the box |u_i| <= e_i (numpy (D,)): (u*, log H/H_0). Ascent on z in
        [-1, 1]^D, u = e z (so the step is relative to each half-width), from the given starting
        directions, the previous solution (warm) and random corners, all batched."""
        D = len(e)
        g = torch.Generator().manual_seed(seed)
        Z = [torch.from_numpy(z0).float() for z0 in starts]
        if warm is not None:
            Z.append(torch.from_numpy(np.clip(warm / e, -1, 1)).float())
        Z += [torch.randint(0, 2, (D,), generator=g).float() * 2 - 1 for _ in range(args.restarts - len(Z))]
        z = torch.stack(Z).to(args.device).requires_grad_(True)
        ev = torch.as_tensor(e, dtype=torch.float32, device=args.device)
        opt = torch.optim.Adam([z], lr=args.lr)
        best, u_best = -np.inf, None
        for _ in range(args.steps + 1):
            opt.zero_grad()
            lh = self.log_hits(z * ev)
            i = int(lh.argmax())
            if float(lh[i]) > best:
                best, u_best = float(lh[i]), (z[i] * ev).detach().cpu().numpy()
            (-lh.sum()).backward()
            opt.step()
            with torch.no_grad():
                z.clamp_(-1, 1)
        return u_best, best

    def shares(self, u):
        """First-order share of each parameter in log H(u)/H_0: u_i d log H / du_i."""
        x = torch.as_tensor(u, dtype=torch.float32, device=args.device)[None].requires_grad_(True)
        self.log_hits(x)[0].backward()
        return (x.grad[0].cpu().numpy() * u).astype(float)


search = BoxSearch(model, feature_cache(model, shield.grid(*grid_spec(args.cache_grid), args.n_ref), one_group))
H0 = search.H0
r = np.load(os.path.join(args.run, 'robustness.npz'))
GRAD = r['grad']
V = r['dir_vecs'][[str(s) for s in r['dir_names']].index('worst')]
STARTS = [np.sign(GRAD), V / np.abs(V).max()]               # the sign of the gradient, the scaled worst direction
PCTS = [float(x) for x in args.pcts.split(',')]
TARGETS = sorted(float(x) for x in args.targets.split(','))

if os.path.exists(OUT) and not args.redo:
    t = np.load(OUT)
    U, names, path_e, path_h = t['U'], [str(s) for s in t['names']], t['path_e'], t['path_h']
    print(f'reusing the search in {OUT} (--redo to search again)')
else:
    names, U = [], []
    for x in PCTS:
        u, lh = search.box_worst(np.full(D, x / 10), STARTS)
        names.append(f'box{x:g}%')
        U.append(u)
        print(f'box +-{x:g}%: worst H/H_0 = {np.exp(lh):.4g}, {int((np.abs(u) > 0.99 * x / 10).sum())}/{D} on a face')
    # greedy allocation
    e, u, todo = np.ones(D), None, list(TARGETS)[::-1]                  # largest target first
    path_e, path_h = [], []
    for it in range(args.max_iter):
        u, lh = search.box_worst(e, STARTS, warm=u, seed=it)
        path_e.append(e.copy())
        path_h.append(lh)
        while todo and np.exp(lh) <= todo[0]:
            names.append(f'alloc{todo.pop(0):g}')
            U.append(u)
            print(f'  target met: {names[-1]} after {it} steps, worst {np.exp(lh):.3f}')
        if not todo:
            break
        s = np.where(e > args.floor / 10 + 1e-9, search.shares(u), -np.inf)  # only parameters that can still shrink
        if not np.isfinite(s).any():
            print('every parameter at the floor; stopping')
            break
        for k in np.argsort(-s)[:args.k_step]:
            if np.isfinite(s[k]) and s[k] > 0:
                e[k] = max(e[k] * args.shrink, args.floor / 10)
        if it % 10 == 0:
            print(f'step {it}: worst H/H_0 = {np.exp(lh):.4g}; {int((e < 1).sum())} parameters tightened, '
                  f'tightest {10 * e.min():.2f}%')
    U, path_e, path_h = np.stack(U), np.stack(path_e), np.array(path_h)
    np.savez(OUT, names=np.array(names), U=U, path_e=path_e, path_h=path_h, pcts=np.array(PCTS),
             targets=np.array(TARGETS), labels=np.array(labels), H_0=H0, cache_grid=args.cache_grid)
    print(f'saved {OUT}')

# =============================================================================
# The found designs: ensemble, members' standard error, simulator (whatever exists)
# =============================================================================
pred = cached_hits(model, search.cache, 1, U)[:, 0] / H0
SH = np.stack([search.shares(u) for u in U])                   # what each parameter contributes
del search                                                     # free the grid before the members'
torch.cuda.empty_cache()
mem = []
for m in load_members(args.model, shield.phi_0, args.device):
    c = feature_cache(m, shield.grid(*grid_spec(args.cache_grid), args.n_ref), one_group)
    mem.append(cached_hits(m, c, 1, U)[:, 0] / cached_hits(m, c, 1, np.zeros((1, D)))[0, 0])
    del c
pred_se = seed_se(mem) if mem else np.zeros(len(U))
sim, sim0 = np.full(len(U), np.nan), {}
for f in sorted(glob.glob(os.path.join(args.run, 'tolerance_sim_*.npz'))):
    s = np.load(f)
    sim[s['idx']] = s['hits']
    sim0[f] = float(s['nominal'])
S0 = np.mean(list(sim0.values())) if sim0 else np.nan

# worst@X% (the scaled direction) from worst_cases.npz, for comparison
wc = {}
if os.path.exists(os.path.join(args.run, 'worst_cases.npz')):
    w = np.load(os.path.join(args.run, 'worst_cases.npz'))
    wn = [str(s) for s in w['names']]
    ws = w['sim_bin'].sum(1)
    wp = w['pred_bin'].sum(1)
    for x in PCTS:
        if f'worst@{x:g}%' in wn:
            i = wn.index(f'worst@{x:g}%')
            wc[x] = (wp[i] / wp[0], ws[i] / ws[0] if w['simulated'] else np.nan)

say = Report()
say(f'model {args.model}; surrogate integrated on the {args.cache_grid} grid, H_0 = {H0:.1f} per {args.n_ref:.0e}; '
    'errors: members\' standard error (surrogate), Poisson (simulator)')
if sim0:
    say(f'simulator: {args.n_muons:,} muons (seed {args.seed}); nominal ' + ', '.join(f'{v:.0f}' for v in sim0.values())
        + ' hits over the workers')


def sim_txt(i):
    if np.isnan(sim[i]):
        return 'not simulated'
    rt = sim[i] / S0
    return f'{rt:.3f} +- {rt * np.sqrt(1 / max(sim[i], 1) + 1 / S0):.3f} ({sim[i]:.0f} hits)'


say('\n' + '=' * 90 + '\nTHE WORST DESIGN IN EACH BOX  (every parameter within +-X%)\n' + '=' * 90)
say(f'{"box":>6s} {"surrogate H/H_0":>18s} {"simulator H/H_0":>30s} {"on a face":>9s}   {"scaled worst direction (surr. / sim.)":s}')
for x in PCTS:
    i = names.index(f'box{x:g}%')
    face = int((np.abs(U[i]) > 0.99 * x / 10).sum())
    ref = f'{wc[x][0]:.3f} / {wc[x][1]:.3f}' if x in wc else '-'
    say(f'+-{x:g}%'.rjust(6) + f' {pred[i]:9.4g} +- {pred_se[i]:<6.3g} {sim_txt(i):>30s} {face:6d}/{D}   {ref}')
    say('        largest shares of log H: ' + ', '.join(f'{labels[k]} {10 * U[i][k]:+.1f}% ({SH[i][k]:+.2f})'
                                                  for k in np.argsort(-SH[i])[:6])
        + f'; cos to the worst direction {U[i] @ V / np.linalg.norm(U[i]) / np.linalg.norm(V):+.2f}')

say('\n' + '=' * 90 + '\nTOLERANCE ALLOCATION  (greedy: tighten the largest contributors to the box worst case)\n' + '=' * 90)
say(f'start: every parameter at +-10%; per step the {args.k_step} largest contributors x{args.shrink:g}, '
    f'floor {args.floor:g}%; {len(path_h)} steps')
for name, u in zip(names, U):
    if not name.startswith('alloc'):
        continue
    i = names.index(name)
    step = int(np.argmax(np.exp(path_h) <= float(name[5:])))
    e = path_e[step]
    say(f'\nworst H/H_0 <= {name[5:]}: surrogate {pred[i]:.3f} +- {pred_se[i]:.3f}, simulator {sim_txt(i)}')
    say(f'  {int((e < 1).sum())} parameters tighter than +-10%, the rest at +-10%:')
    for k in np.argsort(e):
        if e[k] < 1:
            say(f'    {labels[k]:20s} +-{10 * e[k]:5.2f}%   ({abs(shield.phi_0[k].item()) * e[k] / 10:.3g} {shield.units[k]})')
say.save(os.path.join(args.run, 'tolerance_boxes_report.txt'))
np.savez(OUT.replace('.npz', '_eval.npz'), names=np.array(names), pred=pred, pred_se=pred_se, shares=SH, sim=sim, sim0=S0,
         wc_pcts=np.array(list(wc)), wc=np.array(list(wc.values())).reshape(-1, 2))
print(f'saved {args.run}/tolerance_boxes_report.txt')
