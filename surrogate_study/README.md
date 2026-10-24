# Surrogate study

Does an LCSO surrogate give the right hit rate, gradient and Hessian, and which architecture
and training choices matter? It is tested two ways: on toy problems with an exact answer, and on
the muon shield against the simulator. Figures and tables are in `outputs/surrogate_study/figs/`
(runs of 2026-10-19/24).

## Bottom line

- The production surrogate (`outputs/lcso.pt`) has the following setup:
  - a DeepONet, 128 hidden units and 32 features;
  - trained on all eight pools at once (6.2e10 muons, 2.8e8 rows), including a uniform and an
    adaptive pool of designs far from the nominal one;
  - four seeds averaged into one network (a seed more than 10% above the others' median
    validation loss is replaced).
- On the nominal muons it predicts the nominal design to −2%, the half box to 4% and the box to
  13%; on the hot check sample the box to 4%, the one-at-a-time slopes to 3.9% and the curvatures
  to 12%.
- **Directions are determined; magnitudes far from u = 0 are not.** The gradient, the top Hessian
  eigenvectors and the worst direction agree across seeds and architectures to cosine ≥ 0.95. The
  response away from u = 0 is close to exponential, so magnitudes there are always simulated.
- **No alternative tested is worth switching to:**
  - other architectures;
  - a targeted training proposal;
  - importance-sampled simulation.

## Findings → evidence

Training (toys: `toy_*.png`, `toys_summary.txt`, median of 3 seeds; shield: `shield_*.png`,
`shield_summary.txt`, median of 2 seeds):

| finding | evidence |
|---|---|
| Conditioning on the muon is what makes LCSO work. A response surface of H alone needs ~10x more data (synthetic, 1e6: gradient error 9.7 vs 0.64). | `toy_budget.png` |
| Use many designs with few muons each. Toys: 10–1000 samples per design are fine, 1e4–1e5 fail. Shield: 1e6 x 2000 gives 8.9%, 1e7 x 200 gives 18%. | `toy_n.png`, `shield_settings.png` |
| Train on all designs at once; a rolling buffer is ~2x worse. | `toy_scheme.png` |
| Training designs must cover the region where derivatives are read (1–4x the box). δ = 0.25 is poor. | `toy_delta.png` |
| Train hotter. At equal budget a T = 2 pool is best. T = 1 training has too few hits (secants 100%). T = 3 is no better than T = 2. | `shield_settings.png` |
| Capacity matters at 2e8 rows: nominal-muon box 21% at 64/16, 15% at 128/32. Toys: shape barely matters, but a ReLU branch is worst. | `toy_shape.png`, `toy_latent_act.png` |
| Keep the stored 0.4% of the misses; thinning further loses accuracy. | `shield_settings.png` |
| Upweighting low-momentum muons in the loss does not help. | (`lcso/train.py` runs) |
| The errors are far above the simulation noise (0.25% box, 1.6% secants): the models limit the accuracy, not the simulator. | `shield_budget.png` |
| Designs far from nominal: a uniform pool (1024 designs, no radial pull) alone does not help (nominal-muon box 16% → 19%); adding an adaptive pool (212 designs at the tolerance-box worst cases, their neighbourhoods and where the members disagree most) does: box 13%, half box 4.1%, nominal design −2%, members' scatter much smaller; secants 3.4% → 3.9%. This is the production `lcso.pt`. | `final_compare_AD_final.txt`, `member_scores.txt` (`lcso/adaptive_designs.py`) |
| Score single seeds, not only ensembles: one bad seed (validation loss 15% above its siblings) made the A+D ensemble look worse on the secants. Seeds more than 10% above the median validation loss are replaced. | `member_scores.txt` |
| (The six-pool surrogate is kept as `outputs/lcso_v1.pt`; the uniform-pool one as `lcsoA`.) | |
| The earlier `lcso_deeponet.pt` (box 19%, secants 28%) equals a new DeepONet trained on ~3e9–1e10 muons. | `shield_summary.txt`, `shield_oat.png` |

Architecture (`lcso/models.py`; production comparisons `final_compare_*.txt`):

| finding | evidence |
|---|---|
| On toys the DeepONet is the safest default. The low-rank Taylor gives the best Hessian. The full Taylor, full quadratic and gated residual lose accuracy. | `toy_budget.png`, `toy_rank.png` |
| Finite differences win only in low dimension (D = 10) and give no Hessian. | `toys_summary.txt` |
| At the final setup the reduced Taylors are worse (slopes 6–10%, curvatures 18–25%). | `final_compare.txt` |
| An MLP on [u, x] predicts best (nominal box 11%, slopes 2.6%, curvatures 9.9%). A DeepONet with p = 128 is in between. Both match the DeepONet's directions to cosine 0.99–0.997. They are not separable: no cached trunk, derivatives by autograd. **The DeepONet is kept.** | `final_compare_architectures.txt` |
| Taylor + gated residual: the gate (mean u²)² leaves a single parameter at its limit a pure quadratic (symmetric responses, wrong); mean(u⁴) opens it along an axis (`taylor-res4`). Weighting the Taylor part to the designs near u = 0 (`--near 0.3`) makes it as accurate as the DeepONet near the nominal design, worse far from it. | `final_compare_taylor4.txt`, the comparison report |
| The weighted Taylor surrogate (`tr4n30r`) gives the same robustness conclusions as the DeepONet (worst direction cos 0.99, same worst cases per tolerance up to ±3%, same tolerance priorities), 13–20% low along the harmful directions. Each surrogate predicts the other's selected designs better than its own. | `outputs/robustness_taylor/`, the comparison report |
| Toys, error against simulated samples up to 1e8: the DeepONet is best at few samples and levels off; the unweighted Taylors keep improving (better Hessians on 3 of 4 problems at 1e8); the reduced Taylor needs the fewest samples for a given Hessian; the weighting costs 2–12x samples. | `compare_toys.png`, `compare_toys.txt` |

What the derivatives mean:

| finding | evidence |
|---|---|
| Ten gradient components (92% of the norm) agree across models to cosine 0.95–0.99, and the top 2–3 Hessian eigenvectors to 0.9–0.99. The ensemble members agree to 0.94–0.98 on the gradient. | `derivatives_check.txt` |
| Better surrogates find worse, real worst cases. Simulated: x5.2 at \|u\| = 1 and x41 at \|u\| = 2, against x2.65 and x10.6 for the earlier surrogate's direction. The quadratic model underestimates both. | `worst_check.txt` |
| M2:Z_len's gradient shrank because of the surrogate, not the flow. The earlier surrogate overstated the 30–75 GeV response ~2x. The momentum bands cancel in the final one (g/H0 +0.23 to +0.29 earlier, ~0 now). | `flow_vs_model.txt` |

Integration and sampling:

| finding | evidence |
|---|---|
| Integrate the surrogate on a grid over the flow's latent, azimuth and charge (`MuonShield.grid`). It converges to 1e-5 from ~1e6 points. Monte Carlo is at 2.6% after 1e8 muons; randomized QMC is ~2x better. | `quadrature.txt`, `quadrature_convergence.png` |
| Importance-sampling the simulator from a hotter flow gains nothing (x1.0 at T = 1.5). The nominal hits sit in the latent's core. | `importance.txt` |
| A targeted training proposal gains at most 1.74x effective training muons over T = 2 (the bound q* ∝ p^(2/3) π^(1/3)). T = 1.5 gives 1.33x and the best Gaussian bump 1.48x. Temperature is close enough to optimal. | `proposal_check.txt` |

## Setup

Every surrogate models one muon's hit probability at one design:

    p(hit | u, x) = sigmoid(<B(u), T(x)> + bias)

Here u is the design normalized to the tolerance box (u_i = ±1 is ±10%). Then
H(u) = mean_x p and its exact gradient and Hessian come from one pass over the muons
(`lcso.models.derivatives`).

| kind | branch B(u) | assumes |
|---|---|---|
| `deeponet` | MLP | nothing |
| `taylor[:r]` | `[1, u, (Q'u)^2/2]` | a per-muon quadratic, curvature of rank r in a shared basis Q |
| `taylor-sub:r` | `[1, Q'u, (Q'u)^2/2]` | the design acts through r directions |
| `taylor-res`, `taylor-sub-res:r` | the above + an MLP gated by (mean u²)² | the Taylor at u = 0, anything away from it |
| `taylor-res4:r`, `taylor-sub-res4:r` | the same, gated by mean(u⁴) | also one-sided along a single axis; `--near`: the Taylor part fitted near u = 0 |
| `quad` | `[1, u, u_i u_j]` | a full per-muon quadratic |
| `rsm` | as `quad`, constant trunk | a response surface, blind to x (baseline) |
| `mlp[:depth]` | none: an MLP on [u, x] | nothing; not separable |
| `fd` | none | central finite differences (toy baseline) |

**Toys** (`toys.py`): binary outcomes with a known p(u, x).
- `synthetic`: D = 43, rare hits, a 4-d active subspace.
- `short_column`: D = 10.
- `beam_shield`: D = 16, effectively one direction.
- `two_moons`: D = 21, a flat optimum.

**Shield data:**
- **Pools** (`lcso/simulate_pools.py`, `lcso/data.py`): designs simulated once with fresh flow muons at a raised momentum
  temperature; every hit and 0.4% of the misses are stored. Training uses the case-control logit
  offset −log(keep_neg).
- **References** (`shield_ref.py`): fixed designs simulated on stored muons, on which the
  surrogates predict too. The designs are the nominal x4, 16 in the half box, 16 in the box, and
  every parameter alone at ±10%. There are two sets:
  - `T1`: 1e9 nominal muons;
  - `T1.5`: the **hot check sample**, 4e8 muons drawn like the T = 1.5 pools. It has many more
    hits, so the one-at-a-time secants and curvatures are measured on it.

## Files

The surrogate itself (architectures, training, pools) is in `lcso/`; this folder is the study.

| file | does | writes |
|---|---|---|
| `metrics.py` | toy scores, shared helpers | — |
| `toys.py`, `run_toys.py`, `compare_toys.py` | toy problems; every toy job; DeepONet vs. Taylor convergence with samples | `toys/results_*.jsonl`, `figs/compare_toys*` |
| `shield_ref.py` | simulate the reference designs | `shield_ref/` |
| `run_shield.py` | every shield fit, scored on the references; `score`, `as_study`, `Reference` (also used by `lcso/train.py --score`) | `shield/results_*.jsonl`, `shield/arrays/` |
| `plots.py` | every toy and shield figure, the summaries (`--final_only`: the report's versions) | `figs/toy_*`, `figs/shield_*` |
| `derivative_check.py`, `worst_check.py` | which derivative parts are determined; simulate the worst directions | `derivatives_*`, `worst_check*` |
| `quadrature.py`, `quadrature_convergence.py`, `importance.py` | grid vs. sampling; tempered importance sampling | `quadrature*`, `importance.txt` |
| `final_compare.sh`, `final_compare.py` | candidate production surrogates at the final setup, against `outputs/lcso.pt` | `final_compare*` |
| `member_scores.py` | every seed of an ensemble scored alone (the seed-to-seed scatter) | `member_scores.*` |
| `ref_figures.py` | the reference-design figures of any surrogates, side by side | `refs_*` |
| `proposal_check.py`, `flow_vs_model.py` | the training-proposal bound; flow vs. model behind a gradient change | `proposal_check.txt`, `flow_vs_model.txt` |

## Running

```bash
source env.sh                                            # from the repo root
GPUS="0 1" bash lcso/simulate_pools.sh                     # the training pools
python surrogate_study/shield_ref.py --name T1.5 --stage sample --temperature 1.5 --n_muons 400000000 --oat
python surrogate_study/shield_ref.py --name T1 --stage sample --temperature 1 --n_muons 1000000000
python surrogate_study/shield_ref.py --name T1.5 --stage simulate --worker 0 --n_workers 1   # and T1
GPUS="0 1" PER_GPU=5 bash surrogate_study/run_toys.sh      # toys
python surrogate_study/run_toys.py --experiments converge --seeds 2 --worker gpu0   # DeepONet vs. Taylor, 1e8
python surrogate_study/compare_toys.py
GPUS="0 1" PER_GPU=2 bash surrogate_study/run_shield.sh    # shield fits
python surrogate_study/plots.py                            # figures and summaries
python surrogate_study/derivative_check.py
python surrogate_study/worst_check.py designs              # then shield_ref.py --designs ... --tag worst
python surrogate_study/worst_check.py report
python surrogate_study/quadrature.py; python surrogate_study/quadrature_convergence.py
python surrogate_study/importance.py
CANDIDATES="taylor taylor-res4:8 --near 0.3" GPUS="0 1" bash surrogate_study/final_compare.sh
python surrogate_study/member_scores.py --ensembles lcso,taylor
python surrogate_study/ref_figures.py --models lcso,tr4n30r --labels DeepONet,Taylor
python surrogate_study/proposal_check.py --device cpu; python surrogate_study/flow_vs_model.py
```

Every step is resumable. Outputs go to `outputs/surrogate_study/`.
