"""Simulate the one-at-a-time scans of robustness.py: the most sensitive parameters alone at a few
points of the tolerance, on the same sampled muons as its walks, to check the surrogate's scans.

Writes <run>/scan_sim_<worker>.npz (the parameters, the points t, the simulated hits and the
worker's own simulated nominal); plots.py adds them to robustness_scans.png. Workers split the
designs, so several GPUs can share the work:

    CUDA_VISIBLE_DEVICES=2 python scan_sim.py --worker 0 --n_workers 2 &
    CUDA_VISIBLE_DEVICES=3 python scan_sim.py --worker 1 --n_workers 2 &
"""
import argparse
import os

import numpy as np
import torch

from analysis import ArgFormatter, muon_blocks, one_group, setup, simulate

parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument('--run', default='outputs/robustness', help='robustness.py --out_dir')
parser.add_argument('--model', default='outputs/lcso.pt')
parser.add_argument('--t', default='-1,-0.5,0.5,1', help='points of each scan (u_i; +-1 = +-10%%)')
parser.add_argument('--n_muons', type=int, default=int(1e9))
parser.add_argument('--seed', type=int, default=0, help='0 = the muons of robustness.py\'s walks')
parser.add_argument('--block', type=int, default=int(2e7), help='sampled muons held in memory')
parser.add_argument('--worker', type=int, default=0)
parser.add_argument('--n_workers', type=int, default=1)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

r = np.load(os.path.join(args.run, 'robustness.npz'))
shield, model = setup(args.model, args.device)
D, ts = model.dim, [float(v) for v in args.t.split(',')]
jobs = [(int(k), t) for k in r['scan_params'] for t in ts][args.worker::args.n_workers]
U = np.zeros((len(jobs) + 1, D))                      # row 0: the nominal design
for i, (k, t) in enumerate(jobs):
    U[i + 1, k] = t
print(f'worker {args.worker}: {len(jobs)} scan points + the nominal over {args.n_muons:,} muons')
blocks = muon_blocks(shield, args.n_muons, args.block, args.seed)
hits = simulate(model, shield, blocks, U, one_group, 1)[:, 0]
out = os.path.join(args.run, f'scan_sim_{args.worker}.npz')
np.savez(out, params=np.array([k for k, _ in jobs]), t=np.array([t for _, t in jobs]), hits=hits[1:],
         nominal=hits[0], n_muons=args.n_muons)
print(f'saved {out}')
