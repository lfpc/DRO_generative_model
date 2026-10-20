# Surrogate study

Does an LCSO surrogate give the right hit rate, gradient and Hessian, and which architecture
and training choices matter? Tested on toy problems with an exact answer, and on the muon
shield against the simulator.

Every surrogate models the hit probability of one sample (a muon) at one design,
`p(hit | u, x) = sigmoid(<B(u), T(x)> + bias)`, with u the design normalized to the
tolerance box (u_i = +-1: parameter i at its tolerance). The expected hit rate
`H(u) = mean_x p(u, x)` and its exact gradient and Hessian follow from one pass over the
samples (`surrogates.derivatives`). The architectures (`surrogates.py`):

| kind | branch B(u) | what it assumes |
|---|---|---|
| `deeponet` | MLP | nothing |
| `taylor` / `taylor:r` | `[1, u, (Q'u)^2/2]` | per-sample quadratic, curvature of rank r in a learned shared eigenbasis Q |
| `taylor-sub:r` | `[1, Q'u, (Q'u)^2/2]` | the design acts through r directions only (a low-dimensional manifold) |
| `taylor-res` | `taylor` + gated MLP | quadratic at u = 0 exactly, anything beyond it away from 0 |
| `quad` | `[1, u, u_i u_j]` | a full per-sample quadratic |
| `rsm` | as `quad`, constant trunk | a response surface of the hit rate, blind to x (baseline) |
| `fd` | none | central finite differences of the simulated hit rate (baseline) |

## Toy problems (`toys.py`, `run_toys.py`)

Binary outcomes with a known p(u, x), so H, its gradient and Hessian are exact:
`synthetic` (D = 43, rare hits, a 4-d active subspace), `short_column` (D = 10),
`beam_shield` (D = 16, effectively one direction), `two_moons` (D = 21, a flat optimum). Each job
simulates `budget` samples (budget / n designs x n samples, in the region delta around
u = 0), fits one surrogate (`training.py`) and scores it at u = 0 (`metrics.py`): prediction
error in the half box and the box, gradient and Hessian errors, and whether the surrogate's
worst design in |u| <= 1 is the true worst. The experiments vary one setting at a time:
budget, training scheme (fixed vs. stream), samples per design, region size, DeepONet
shape, Taylor rank.

## Muon shield (`shield_pool.py`, `shield_ref.py`, `run_shield.py`)

1. `shield_pool.py`: designs simulated once (1e6-1e7 flow muons each); every hit and 0.4% of the
   misses are stored. Pools: T = 1.5 at +-10% (2048 designs), +-5% and +-20% regions, T = 1, T = 2,
   and T = 2 and 3 with 10 000 designs x 1e6 muons.
2. `shield_ref.py`: reference designs (nominal x4, 16 in the half box, 16 in the box, every
   parameter alone at +-10%) simulated on one stored muon sample, 4e8 at T = 1.5 and 1e9 at
   T = 1. The surrogates predict on the same muons.
3. `run_shield.py`: fits on subsets of the pools (case-control: logit offset -log(keep_neg)),
   scores on the references, and saves the gradient and Hessian at u = 0. The existing
   `outputs/lcso_deeponet.pt` is scored too.

## Findings (runs of 2026-10-19/20; figures and tables in `outputs/surrogate_study/figs/`)

Toys (`toy_*.png`, `toys_summary.txt`; median of 3 seeds, errors relative, at u = 0):

- **Conditioning on the sample is what makes LCSO work.** The response surface fitted to the
  same data without x is far worse at small budgets (synthetic, 1e6 samples: gradient error
  9.7 vs 0.64 for the DeepONet, Hessian 76 vs 1.25) and only catches up near 1e7.
- **Every per-sample surrogate converges with the budget** (`toy_budget.png`). On the 43-d
  synthetic problem, at 1e7 samples: hit rate in the box ~10%, gradient ~0.21-0.25, Hessian
  ~0.45-0.5, and the surrogate's worst design reaches 95% of the true worst rise.
- **Low rank wins for the Hessian.** The Taylor surrogate on an r-dimensional subspace is the
  best Hessian on every problem (1e7: beam 0.13 vs 0.44 for the DeepONet, two moons 0.06 vs
  0.20), and lower ranks are better, even r = 1-2 (`toy_rank.png`). The full-rank Taylor, the
  full quadratic and the gated residual add parameters and lose accuracy.
- **The DeepONet is the safest default**: best or near-best hit-rate predictions on every
  problem and budget, and the most robust at small budgets.
- **Finite differences** beat every surrogate's gradient at high budget in low dimension
  (short column, D = 10, 1e7: 0.04 vs 0.08), and lose in 43 dimensions (0.63 vs 0.25). They give
  no Hessian.
- **Train on all designs at once, not a rolling buffer.** With the same data and steps, the
  rolling buffer is ~2x worse at mid budgets (`toy_scheme.png`).
- **Many designs with few samples each** beats few designs with many: 10-1000 samples per
  design are all fine; 1e4-1e5 (10-100 designs) fail (`toy_n.png`).
- **Region size**: training designs must cover the region where derivatives and worst cases
  are read. delta = 0.25 is poor; a region of 1-4x the tolerance box helps the gradient and
  Hessian at u = 0 (`toy_delta.png`).
- **DeepONet shape barely matters** (`toy_shape.png`): width and depth change the errors within
  the seed scatter, except on the 43-d problem, where deeper and wider improve the Hessian
  (1.9 -> 0.8). A larger latent p helps the Hessian there too. **The branch activation matters**:
  ReLU is worst for both the gradient and the Hessian; tanh and softplus are best
  (`toy_latent_act.png`).

Muon shield (`shield_*.png`, `shield_summary.txt`; median of 2 seeds; simulation-noise floors:
0.25% for T = 1.5 box predictions, 1.6% for one-at-a-time secants):

- At 1.8e10 simulated muons (1800 designs), the DeepONet predicts the T = 1.5 reference
  designs to 9% in the box, the one-at-a-time secants to 17% and the curvatures to 26%. The
  full-rank Taylor matches it there (9%, 12%, 28%) but is much worse below 3e9. The Taylor on a
  4-d subspace is the best predictor at the top budget (7.5%). The full quadratic and the
  response surface fail. All of these errors are far above the simulation noise, so the
  models, not the simulator, limit the accuracy.
- **The existing `lcso_deeponet.pt`** (box 19% at T = 1.5, secants 28%) matches a new DeepONet
  trained on ~3e9-1e10 muons. It is beaten by the new ones at the top budget.
- **At T = 1 every surrogate is worse** (box ~24-30%, nominal often 5-15% high): the T = 1
  hits come from a corner of momentum space that T = 1.5 training samples thinly.
- **Train hotter.** At equal budget (3e9), a T = 2 pool gives the best DeepONet on both
  references (T = 1.5 box 7.7%, secants 8%; T = 1 box 28%). T = 1 training is the worst
  (secants 100%): too few hits.
- More designs with fewer muons each is better here too (1e6 muons x 2000 designs: 8.9%,
  against 1e7 x 200: 18%). Thinning the misses below the stored 0.4% loses accuracy.
- **Hotter and more designs, combined (`hot`).** A T = 2 pool of 10 000 designs x 1e6 muons
  gives, at 9e9 muons, a T = 1.5 box error of 4%, secants 6%, curvatures 16%. T = 3 is no
  better than T = 2. T = 1 stays at ~25-30%: its hits come from 30-75 GeV muons (per
  momentum bin, the hits above 125 GeV are predicted to a few percent, those below 75 GeV
  are not), and every pool has few of them.
- **The final surrogate** (`outputs/lcso.pt`, `train_surrogate.py`): every pool at once
  (5e10 muons, 2e8 rows), a larger network (128 hidden, 32 features; capacity matters at this
  size: 64/16 gives T = 1 box 21%, 128/32 15%), four seeds averaged into one network. T = 1.5:
  box 3.0%, secants 3.4%, curvatures 12%; T = 1: nominal -7%, box 16%. Upweighting the
  low-momentum muons in the loss did not help.
- **Which parts of the derivatives are determined** (`derivative_check.py`,
  `derivatives_*`): across single models at 1.8e10 the ten largest gradient components (92% of
  its norm) agree to cos 0.95-0.99 and the top 2-3 Hessian eigenvectors to 0.9-0.99; the rest
  is small components and an overall +-30% scale. The four members of the final surrogate agree
  to cos 0.94-0.98 on the whole gradient and 0.96-0.99 on the top eigenvector.
- **Does it matter for robustness?** (`worst_check.py`, `worst_check.txt`) Simulated at T = 1:
  the worst direction of the new models raises the hits x5.2 at |u| = 1 and x41 at |u| = 2,
  against x2.65 and x10.6 for that of the earlier surrogate (the report's), and the quadratic
  model of either underestimates both (the response is near exponential). Better surrogates
  find worse, real, worst cases; magnitudes must be simulated.

- **Integrating instead of sampling** (`quadrature.py`, `quadrature_convergence.py`): with the
  muons on axis, the surrogate's hits are an integral over the flow's 2-d momentum latent, the
  azimuth and the charge. A grid (Gaussian weights in the latent) is converged to 1e-5 from ~1e6
  points; plain Monte Carlo is at 2.6% (H), 1.9% (gradient) after 1e8 muons, randomized QMC
  about 2x better. The robustness scripts now use the grid (`MuonShield.grid`).
- **Importance sampling the simulator** (`importance.py`): drawing the muons at a raised
  momentum temperature and reweighting to T = 1 (exact weights from the flow) gains nothing
  (x1.0 at T = 1.5, worse above): the T = 1 hits are 30-75 GeV muons in the core of the latent,
  of which a hotter sample holds fewer.

## Running

```bash
source env.sh                                            # from the repo root
GPUS="0 2 3" bash surrogate_study/run_shield_pool.sh       # pools, ~8 h on 3 GPUs
python surrogate_study/shield_ref.py --name T1.5 --stage sample --temperature 1.5 --n_muons 400000000 --oat
python surrogate_study/shield_ref.py --name T1 --stage sample --temperature 1 --n_muons 1000000000
python surrogate_study/shield_ref.py --name T1.5 --stage simulate --worker 0 --n_workers 1   # and T1
GPUS="0 2 3" PER_GPU=5 bash surrogate_study/run_toys.sh    # toys, ~3 h
GPUS="0 2 3" PER_GPU=2 bash surrogate_study/run_shield.sh  # shield fits, ~2 h
python surrogate_study/plots.py                            # outputs/surrogate_study/figs/
python surrogate_study/derivative_check.py                 # which derivative components agree
python surrogate_study/worst_check.py designs              # then simulate them (shield_ref.py --designs),
python surrogate_study/worst_check.py report               # and compare
python surrogate_study/quadrature.py                       # sampling vs. quadrature, errors at equal points
python surrogate_study/quadrature_convergence.py           # running Monte Carlo vs. the grid
python surrogate_study/importance.py                       # tempered importance sampling of the simulator
```

Everything is resumable: rerun a step and it continues where it stopped. Outputs go to
`outputs/surrogate_study/`.
