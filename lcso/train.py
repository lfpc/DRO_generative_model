"""Train an LCSO surrogate on simulated pools, or stack several trained seeds into one.

  1. the pools (lcso/simulate_pools.py): designs spread over the +-10% box, each with its own
     muons at a raised momentum temperature (more of the muons that make the hits), every hit
     and a fraction of the misses kept
  2. one fit on all of the pools' designs at once, with the case-control logit offset; early
     stopping on held-out designs, the learning rate dropping 3x on each plateau (lcso/training.py)
  3. saved for the analysis as an LCSONet (a DeepONet) or a ModelSurrogate (any other --kind of
     lcso/models.py, e.g. the Taylor surrogate 'taylor-res4:8'); --score scores it on the
     reference designs (validation/validate.py)

--stack averages the logits of several checkpoints of one kind into one model (an ensemble). The
production surrogates are four seeds stacked; a seed whose validation loss is more than 10% above
the median of the others is replaced by a new seed.

    P=T2_m1e6+T2_d2_m1e6+T1.5_d1+T1.5_adapt
    python -m lcso.train --pool $P --hidden 128 --p 32 --seed 0 --name lcso_s0
    python -m lcso.train --pool $P --hidden 128 --p 32 --p_res 32 --kind taylor-res4:8 --near 0.3 --seed 0 --name taylor_s0
    python -m lcso.train --stack outputs/models/lcso_s{0,1,2,3}.pt --name lcso
"""
import fcntl
import json
import os

import torch

import cli
from paths import MODELS, checkpoint
from shield import MuonShield

from . import data
from .models import Ensemble, make
from .surrogate import LCSONet, ModelSurrogate, load_surrogate, save_surrogate, stack
from .training import fit

parser = cli.parser(__doc__)
parser.add_argument('--pool', help="pool (paths.POOLS/<pool>), or several: 'A+B'")
parser.add_argument('--stack', nargs='+', default=None, help='instead of training: average these checkpoints')
parser.add_argument('--name', default='lcso', help='checkpoint: paths.MODELS/<name>.pt')
parser.add_argument('--kind', default='deeponet', help="architecture (lcso/models.py), e.g. 'taylor-res4:8'")
parser.add_argument('--n', type=int, default=None, help='muons per design to use (default: all of them)')
parser.add_argument('--keep_neg', type=float, default=None, help='thin the stored misses to this fraction (default: as stored)')
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--p', type=int, default=16, help='branch/trunk feature dimension (DeepONet)')
parser.add_argument('--p_res', type=int, default=16, help='features of the gated residual (taylor-*-res)')
parser.add_argument('--near', type=float, default=0.0,
                    help='taylor-*-res: the Taylor part learns from each row with weight exp(-sqrt(dist4(u)) / near^2), '
                         'the residual from all (0: plain loss)')
parser.add_argument('--batch', type=int, default=16384)
parser.add_argument('--max_steps', type=int, default=200000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--score', action='store_true', help='score on the study references (pins ~22 GB of host memory)')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
path = checkpoint(args.name)
phi_0 = MuonShield().phi_0


def score():
    from validation import validate
    refs = {t: validate.Reference(t) for t in validate.REFS}
    model = validate.as_study(load_surrogate(path, phi_0, args.device))
    print(json.dumps(validate.score(model, refs, {}), indent=1))


if args.stack:
    models = [load_surrogate(f, phi_0) for f in args.stack]
    info = dict(stacked=args.stack, trained_on=[torch.load(f, weights_only=False).get('trained_on') for f in args.stack])
    if all(isinstance(m, LCSONet) for m in models):
        save_surrogate(stack(models), path, **info)
    else:
        if len({m.hp.get('kind') for m in models}) > 1:
            raise SystemExit('--stack: the checkpoints are of different kinds')
        save_surrogate(ModelSurrogate(Ensemble([m.net for m in models]), phi_0, models[0].delta, models[0].hp), path, **info)
    print(f'saved {path}: the mean logit of {len(models)} models')
    if args.score:
        score()
    raise SystemExit

# the pools take 20-30 GB of host memory while loading: one process at a time, then free them
with open(os.path.join(MODELS, '.pool_load.lock'), 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    pool = data.load_pool(args.pool, len(phi_0))
    c = dict(pool=args.pool, seed=args.seed, n=args.n, keep_neg=args.keep_neg or pool['keep_neg'],
             budget=(args.n or 0) * len(pool['ids']))
    train, val = data.train_val(c, len(phi_0), args.device)
    del pool
    data.clear_cache()
print(f'{args.pool}: {train.U.shape[0]} + {val.U.shape[0]} validation designs, {len(train.y):,} training rows, '
      f'{int(train.y.sum()):,} hits (logit offset {train.offset:.2f})', flush=True)

torch.manual_seed(args.seed)
net = make(args.kind, len(phi_0), 7, p=args.p, hidden=args.hidden, p_res=args.p_res).to(args.device)
net.near = args.near
info = fit(net, train, val, 'fixed', max_steps=args.max_steps, batch=args.batch, evals_per_epoch=4, seed=args.seed,
           log=print)
print(f'trained {args.kind}: {info["steps"]} steps, validation BCE {info["val_bce"]:.6g}')
trained_on = dict(c, steps=info['steps'], val_bce=info['val_bce'], near=args.near)
if args.kind == 'deeponet':
    # lcso.models.DeepONet has LCSONet's layout: copy the weights over
    model = LCSONet(phi_0, hidden=args.hidden, p=args.p)
    missing, unexpected = model.load_state_dict(net.state_dict(), strict=False)
    assert set(missing) == {'lower_bound', 'upper_bound'} and not unexpected, (missing, unexpected)
else:
    model = ModelSurrogate(net.cpu(), phi_0, hp=dict(kind=args.kind, hidden=args.hidden, p=args.p, p_res=args.p_res, dx=7))
save_surrogate(model, path, trained_on=trained_on)
print(f'saved {path}')
if args.score:
    score()
