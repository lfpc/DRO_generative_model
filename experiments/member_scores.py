"""The seed-to-seed scatter of the reference scores: every member of the given ensembles scored
alone on the reference designs (validation/validate.py's scores), next to
the ensembles' own scores from final_compare.py. Tells whether a difference between two
ensembles is beyond what retraining with another seed changes.

The references are loaded once (~22 GB of host memory), so everything runs in one process.

    python experiments/member_scores.py --ensembles lcso,lcsoA,lcsoAD     # -> figs/member_scores.{txt,json}

Run in the first analysis.
"""
import json
import os

import numpy as np
import torch

import cli
from lcso.surrogate import load_surrogate
from paths import EXPERIMENTS, OUTPUTS, ROOT
from shield import MuonShield
from validation.validate import REFS, Reference, as_study, ref_scores

FIGS = os.path.join(EXPERIMENTS, 'figs')
KEYS = ['h0_err_T1', 'pred_half_T1', 'pred_box_T1', 'h0_err_T1.5', 'pred_box_T1.5', 'oat_slope_err', 'oat_curv_err']

if __name__ == '__main__':
    parser = cli.parser(__doc__)
    parser.add_argument('--ensembles', default='lcso,lcsoA,lcsoAD', help='outputs/<name>.pt, stacked from <name>_s<k>.pt')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    shield = MuonShield()
    refs = {t: Reference(t) for t in REFS}
    cache_path = os.path.join(FIGS, 'member_scores.json')
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    lines = ['each member alone on the reference designs (columns _T1: nominal muons, _T1.5: hot check sample); '
             'mean +- sd over the members, and the ensemble (final_compare.py)', '',
             f'{"":16s}' + ''.join(f'{k:>16s}' for k in KEYS)]
    for ens in args.ensembles.split(','):
        ck = torch.load(os.path.join(OUTPUTS, f'{ens}.pt'), map_location='cpu', weights_only=False)
        rows = []
        for p in ck['stacked']:
            name = os.path.splitext(os.path.basename(p))[0]
            if name not in cache:
                model = load_surrogate(os.path.join(ROOT, p), shield.phi_0, args.device)
                net = as_study(model)
                s = {}
                for tag, R in refs.items():
                    s.update(ref_scores(R, R.predict(net), tag))
                cache[name] = s
                with open(cache_path, 'w') as f:
                    json.dump(cache, f, indent=1)
                print(name, json.dumps({k: round(s[k], 4) for k in KEYS}), flush=True)
            rows.append([cache[name][k] for k in KEYS])
            lines.append(f'  {name:14s}' + ''.join(f'{v:16.4f}' for v in rows[-1]))
        rows = np.array(rows)
        lines.append(f'  {ens + " mean+-sd":14s}' + ''.join(f'{m:9.4f}+-{s:.3f}' for m, s in zip(rows.mean(0), rows.std(0, ddof=1))))
        ef = os.path.join(FIGS, f'final_compare_{ens}_scores.json')
        if os.path.exists(ef):
            e = json.load(open(ef))
            lines.append(f'  {ens + " ensemble":14s}' + ''.join(f'{e[k]:16.4f}' for k in KEYS))
        lines.append('')
    with open(os.path.join(FIGS, 'member_scores.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
