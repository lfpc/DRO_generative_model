"""Train an LCSO surrogate (a DeepONet) on simulated pools, or stack several trained seeds into one.

  1. the pools (lcso/simulate_pools.py): designs spread over the +-10% box, each with its own
     muons at a raised momentum temperature (more of the muons that make the hits), every hit
     and a fraction of the misses kept
  2. one fit on all of the pools' designs at once, with the case-control logit offset; early
     stopping on held-out designs, the learning rate dropping 3x on each plateau (lcso/training.py)
  3. saved for the analysis as an LCSONet (scored by validation/validate.py)

--stack averages the logits of several checkpoints into one model (an ensemble). The production
surrogates are four seeds stacked; a seed whose validation loss is more than 10% above the median
of the others is replaced by a new seed (run_all.sh).

    python -m lcso.train --pool T2_m1e6+T1.5_d1 --hidden 128 --p 32 --seed 0 --name lcso_s0
    python -m lcso.train --stack outputs/models/lcso_s{0,1,2,3}.pt --name lcso
"""
import fcntl
import os

import torch

import cli
from paths import MODELS, checkpoint
from shield import MuonShield

from . import data
from .models import make
from .surrogate import LCSONet, load_surrogate, save_surrogate, stack
from .training import fit

parser = cli.parser(__doc__)
parser.add_argument('--pool', help="pool (paths.POOLS/<pool>), or several: 'A+B'")
parser.add_argument('--stack', nargs='+', default=None, help='instead of training: average these checkpoints')
parser.add_argument('--name', default='lcso', help='checkpoint: paths.MODELS/<name>.pt')
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--p', type=int, default=16, help='branch/trunk feature dimension')
parser.add_argument('--batch', type=int, default=16384)
parser.add_argument('--max_steps', type=int, default=200000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
path = checkpoint(args.name)
phi_0 = MuonShield().phi_0

if args.stack:
    models = [load_surrogate(f, phi_0) for f in args.stack]
    save_surrogate(stack(models), path, stacked=args.stack,
                   trained_on=[torch.load(f, weights_only=False).get('trained_on') for f in args.stack])
    print(f'saved {path}: the mean logit of {len(models)} models')
else:
    # the pools take 20-30 GB of host memory while loading: one process at a time, then free them
    with open(os.path.join(MODELS, '.pool_load.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        train, val = data.train_val(args.pool, args.seed, args.device)
        data.clear_cache()
    off = train.offset if torch.is_tensor(train.offset) else torch.tensor([train.offset])
    print(f'{args.pool}: {train.U.shape[0]} + {val.U.shape[0]} validation designs, {len(train.y):,} training rows, '
          f'{int(train.y.sum()):,} hits (logit offset {float(off.min()):.2f}-{float(off.max()):.2f})', flush=True)

    torch.manual_seed(args.seed)
    net = make('deeponet', len(phi_0), 7, p=args.p, hidden=args.hidden).to(args.device)
    info = fit(net, train, val, max_steps=args.max_steps, batch=args.batch, evals_per_epoch=4, seed=args.seed, log=print)
    print(f'trained: {info["steps"]} steps, validation BCE {info["val_bce"]:.6g}')
    model = LCSONet(phi_0, hidden=args.hidden, p=args.p)          # lcso.models.DeepONet has its layout
    missing, unexpected = model.load_state_dict(net.state_dict(), strict=False)
    assert set(missing) == {'lower_bound', 'upper_bound'} and not unexpected, (missing, unexpected)
    save_surrogate(model, path, trained_on=dict(pool=args.pool, seed=args.seed, steps=info['steps'], val_bce=info['val_bce']))
    print(f'saved {path}')
