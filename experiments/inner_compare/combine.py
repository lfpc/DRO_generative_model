"""One figure over all five problems, from the JSON each run leaves behind."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_compare import COLOR, METHODS

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ['two_moons', 'portfolio_optimization', 'newsvendor', 'short_column', 'beam_shield']
d = {p: json.load(open(os.path.join(HERE, f'{p}.json'))) for p in ORDER
     if os.path.exists(os.path.join(HERE, f'{p}.json'))}

print(f'{"":26s}' + ''.join(f'{p[:13]:>15s}' for p in d))
print('\nnats below the real data (lower is better; the adversary should stay plausible)')
for m in METHODS:
    print(f'  {m:24s}' + ''.join(f'{v["base_plaus"] - v["rows"][m]["plaus"]:15.3f}'
                                 for v in d.values()))
print('\nsurrogate error at its own points, MINUS the error on nominal data')
print('(near zero = the surrogate is as good there as on real data; large = it was gamed)')
for m in METHODS:
    print(f'  {m:24s}' + ''.join(f'{v["rows"][m]["error"] - v["base_err"]:15.4f}'
                                 for v in d.values()))
print('\ntrue-loss evaluations')
for m in METHODS:
    print(f'  {m:24s}' + ''.join(f'{v["rows"][m]["evals"]:15,d}' for v in d.values()))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), constrained_layout=True)
xs, w = np.arange(len(d)), 0.2
for j, m in enumerate(METHODS):
    axes[0].bar(xs + (j - 1.5) * w,
                [max(v['base_plaus'] - v['rows'][m]['plaus'], 1e-3) for v in d.values()],
                width=w, color=COLOR[m], label=m)
    axes[1].bar(xs + (j - 1.5) * w,
                [abs(v['rows'][m]['error'] - v['base_err']) for v in d.values()],
                width=w, color=COLOR[m], label=m)
axes[0].set(yscale='log', ylabel='nats below the real data', xticks=xs,
            title='how implausible are the points it certifies against?')
axes[1].set(yscale='log', ylabel='|excess surrogate error|', xticks=xs,
            title='is the surrogate still valid where it went?')
for ax in axes:
    ax.set_xticklabels([p.replace('_optimization', '') for p in d], rotation=12, fontsize=8)
    ax.legend(fontsize=7)
fig.suptitle('every adversary calibrated to the same worst-case risk (+25%)', fontsize=10)
fig.savefig(os.path.join(HERE, 'figs', 'compare_all.png'), dpi=150)
print(f"\n  wrote {os.path.join(HERE, 'figs', 'compare_all.png')}")
