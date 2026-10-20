"""Train the production LCSO surrogate (DeepONet) the way the surrogate study found best.

  1. a pool of simulated designs (surrogate_study/shield_pool.py): designs spread over the
     +-10% box, each with its own n flow muons at a raised momentum temperature (more of the
     high-momentum muons that make the hits), every hit and a fraction of the misses kept
  2. one fit on all of the pool's designs at once, with the case-control logit offset; early
     stopping on held-out designs, the learning rate dropping 3x on each plateau
     (surrogate_study/training.py)
  3. saved as surrogate.LCSONet, scored on the study's reference designs if they exist

    python surrogate_study/shield_pool.py --name T2.5_m1e6 --temperature 2.5 --n_muons 1000000 --n_designs 10000
    python train_surrogate.py --pool T2.5_m1e6 --name lcso_T2.5
    python train_surrogate.py --pool T1.5_d1+T2_m1e6+T3_m1e6 --name lcso      # several pools at once
    python train_surrogate.py --stack outputs/lcso_s0.pt outputs/lcso_s1.pt --name lcso   # an ensemble
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surrogate_study'))
import run_shield  # noqa: E402
from surrogates import make  # noqa: E402
from training import fit  # noqa: E402

from shield import MuonShield  # noqa: E402
from surrogate import LCSONet, load_surrogate, save_surrogate, stack  # noqa: E402


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument('--pool', help="pool (outputs/surrogate_study/shield_pool/<pool>), or several: 'A+B'")
parser.add_argument('--stack', nargs='+', default=None, help='instead of training: average these checkpoints (one LCSONet)')
parser.add_argument('--name', default='lcso', help='checkpoint: outputs/<name>.pt')
parser.add_argument('--n', type=int, default=None, help='muons per design to use (default: all of them)')
parser.add_argument('--keep_neg', type=float, default=None, help='thin the stored misses to this fraction (default: as stored)')
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--p', type=int, default=16, help='branch/trunk feature dimension')
parser.add_argument('--batch', type=int, default=16384)
parser.add_argument('--max_steps', type=int, default=200000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--score', action='store_true', help='score on the study references (pins ~22 GB of host memory)')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
path = os.path.join(run_shield.ROOT, 'outputs', f'{args.name}.pt')


def score():
    refs = {t: run_shield.Reference(t) for t in run_shield.REFS}
    model = run_shield.Production(load_surrogate(path, MuonShield().phi_0, args.device))
    print(json.dumps(run_shield.score(model, refs, {}), indent=1))


if args.stack:
    models = [load_surrogate(f, MuonShield().phi_0) for f in args.stack]
    save_surrogate(stack(models), path, stacked=args.stack,
                   trained_on=[torch.load(f, weights_only=False).get('trained_on') for f in args.stack])
    print(f'saved {path}: the mean logit of {len(models)} models')
    if args.score:
        score()
    raise SystemExit

pool = run_shield.load_pool(args.pool)
c = dict(pool=args.pool, seed=args.seed, n=args.n, keep_neg=args.keep_neg or pool['keep_neg'],
         budget=(args.n or 0) * len(pool['ids']))
train, val = run_shield.train_val(c, args.device)
print(f'{args.pool}: {train.U.shape[0]} + {val.U.shape[0]} validation designs, {len(train.y):,} training rows, '
      f'{int(train.y.sum()):,} hits (logit offset {train.offset:.2f})', flush=True)

torch.manual_seed(args.seed)
net = make('deeponet', 43, 4, p=args.p, hidden=args.hidden).to(args.device)
info = fit(net, train, val, 'fixed', max_steps=args.max_steps, batch=args.batch, evals_per_epoch=4, seed=args.seed,
           log=print)
print(f'trained: {info["steps"]} steps, validation BCE {info["val_bce"]:.6g}')

# the study's DeepONet has the production network's layout: copy it over
model = LCSONet(MuonShield().phi_0, hidden=args.hidden, p=args.p)
missing, unexpected = model.load_state_dict(net.state_dict(), strict=False)
assert set(missing) == {'lower_bound', 'upper_bound'} and not unexpected, (missing, unexpected)
save_surrogate(model, path, trained_on=dict(c, steps=info['steps'], val_bce=info['val_bce']))
print(f'saved {path}')
if args.score:
    score()
