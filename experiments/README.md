# Experiments

The studies that chose and checked the method; none of them is part of the pipeline
(`run_all.sh`). Run from the repository root after `source env.sh`, e.g.
`python experiments/quadrature.py`. Results go to `outputs/results/experiments/`, figures to
`outputs/figures/experiments/`. Most ran in the first analysis (43 parameters, muons on axis); their
results are in `outputs/archive/round1_43p_onaxis/surrogate_study/`.

## Integrating over the muons (second analysis)

| script | question | answer |
|---|---|---|
| `quadrature.py` | sampling vs. the quadrature grid for H, its gradient and Hessian | the analysis grids are within 4e-4 (801x64) and 2.5e-3 (201x64) of a much finer grid in H; 1e9 sampled muons leave 0.7% |
| `quadrature_convergence.py` | the same, as running estimates of the reported quantities | the grid converges where Monte Carlo is still at the percent level |
| `importance.py` | would importance-sampling the simulator from a hotter flow help? | no: x1.0 at T = 1.5, the nominal hits sit in the latent's core |
| `proposal_check.py` | could a targeted training proposal beat the momentum temperature? | at most 1.74x effective muons over T = 2 (the bound q* ~ p^(2/3) pi^(1/3)); T = 1.5 gives 1.33x |

## Choosing the surrogate (first analysis)

Every surrogate models one muon's hit probability, `p(hit | u, x) = sigmoid(<B(u), T(x)> + bias)`,
with the architectures of `lcso/models.py` (DeepONet, Taylor of rank r, full quadratic,
response surface, MLP).

| script | does |
|---|---|
| `toys.py`, `run_toys.py`, `run_toys.sh` | four toy problems with an exact answer (D = 10-43, rare hits), every architecture and training setting |
| `compare_toys.py` | DeepONet vs. Taylor: convergence to the exact gradient and Hessian with the samples |
| `run_shield.py`, `run_shield.sh` | the shield: every architecture and training setting, scored on the reference designs (`validation/`) |
| `plots.py` | the figures and summaries of both (`--final_only`: the versions in the report's App. C) |
| `final_compare.py`, `final_compare.sh` | candidate production surrogates at the final setup |
| `member_scores.py` | every seed of an ensemble scored alone (the seed-to-seed scatter) |
| `derivative_check.py`, `worst_check.py` | which parts of the gradient and Hessian are determined; the worst directions of different surrogates, simulated |
| `flow_vs_model.py` | why a gradient changed between two versions: the flow or the surrogate |

What they found (the choices of `run_all.sh`):

- Conditioning on the muon is what makes it work: a response surface of the hit count alone needs
  ~10x more simulation for the same gradient.
- Many designs with few muons each (1e6 x 2000 beats 1e7 x 200), all trained at once, at a raised
  momentum temperature (T = 2; T = 1 has too few hits, T = 3 is no better), keeping all hits and
  0.4% of the misses.
- The DeepONet is the safest default; capacity matters at 1e8 rows (128 hidden, p = 32). An MLP
  on [u, x] predicts slightly better but has no separable structure (no cached trunk); the Taylor
  surrogates give the same robustness conclusions, 13-20% low along the harmful directions.
- Four seeds stacked; a seed more than 10% above the median validation loss is replaced (one bad
  seed made an ensemble look worse).
- Designs far from the nominal one help only when chosen adaptively (`lcso/adaptive_designs.py`).
- Directions (gradient, top Hessian eigenvector, worst direction) agree across seeds and
  architectures to cosine >= 0.95; magnitudes far from u = 0 do not, so they are always simulated.
