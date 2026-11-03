"""Do M4's inner yoke and M7's widths act together in the tolerance boxes' worst designs?

The worst designs of the +-2, 3, 5% boxes (analysis/tolerance_boxes.py) widen M4's inner yoke,
which alone changes nothing. Each is simulated as found, with M4:dXYokeIn set back to nominal,
with it reversed, and with M7's four widths (dXIn, dXOut, dXYokeIn, dXYokeOut) at nominal; the
surrogate predicts the same designs. The report's table of Sec. "M4's inner yoke acts through M7".

    python experiments/box_interaction.py --n_muons 3e8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.core import cached_hits, feature_cache, muon_blocks, one_group, simulate  # noqa: E402
from lcso.surrogate import load_surrogate  # noqa: E402
from paths import MODELS, RUN  # noqa: E402
from shield import MuonShield  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', default=os.path.join(MODELS, 'lcso.pt'))
parser.add_argument('--run', default=RUN, help='results directory with tolerance_boxes.npz')
parser.add_argument('--boxes', default='box2%,box3%,box5%')
parser.add_argument('--n_muons', type=float, default=3e8)
parser.add_argument('--seed', type=int, default=7, help='muon seed')
args = parser.parse_args()

b = np.load(os.path.join(args.run, 'tolerance_boxes.npz'), allow_pickle=True)
L = list(b['labels'])
k4 = L.index('M4:dXYokeIn')
k7 = [L.index(p) for p in ('M7:dXOut', 'M7:dXYokeOut', 'M7:dXYokeIn', 'M7:dXIn')]
names, U = ['nominal'], [np.zeros(len(L), np.float32)]
for n in args.boxes.split(','):
    u = b['U'][list(b['names']).index(n)]
    for tag, f in (('as found', 1), ("M4's inner yoke at nominal", 0), ("M4's inner yoke reversed", -1)):
        v = u.copy()
        v[k4] *= f
        names.append(f'{n} {tag}')
        U.append(v)
    v = u.copy()
    v[k7] = 0
    names.append(f"{n} M7's widths at nominal")
    U.append(v)
U = np.stack(U)

sh = MuonShield('flow')
sh.muons.load()
model = load_surrogate(args.model, sh.phi_0).cuda()
pred = cached_hits(model, feature_cache(model, sh.muons.points('201x64', 1e9), one_group), 1, U)[:, 0]
sim = simulate(model, sh, muon_blocks(sh, int(args.n_muons), int(1e7), seed=args.seed), U, one_group, 1)[:, 0]
print(f'{"design":40s} {"surrogate":>10s} {"simulator Y/Y_0":>18s}')
for n, p, s in zip(names, pred, sim):
    r = s / sim[0]
    print(f'{n:40s} {p / pred[0]:10.2f} {r:9.2f} +- {r * np.sqrt(1 / max(s, 1) + 1 / sim[0]):.2f}  ({s:.0f} hits)')
