# Robustness of the SHiP muon shield

How much do the muon hits change when the 43 free parameters of the optimised muon shield deviate
from their nominal values? A surrogate of the per-muon hit probability (LCSO) is trained once on
simulated designs, the robustness quantities are read off it (sensitivities, Hessian, active
subspace, worst directions, the worst design for a given tolerance, a tolerance allocation), and
every conclusion is checked with the simulator.

Three reports, built from the code here (`latexmk -pdf <file>.tex` in their folders):

| report | contents |
|---|---|
| `report/muon_shield_robustness.tex` | the analysis with the DeepONet surrogate (the main one) |
| `report/taylor/muon_shield_robustness_taylor.tex` | the same analysis with the Taylor surrogate (`python build.py` first: it reuses the main report's model-independent sections) |
| `report/comparison/surrogate_comparison.tex` | the two surrogates compared: convergence to the exact derivatives on toy problems, and the robustness conclusions on the shield |

`source env.sh` first: the `muonshield` conda env, and the simulator (`cuda_muons_ship`, from
MuonsAndMatter, needs a GPU) on the path. The muons come from the generative flow in
`~/projects/generator_muons`; set `PROJECTS_DIR` if your projects are not in `~/projects`. Every
script takes `--help`, which lists each flag with its default. Results go to `outputs/` (not
tracked).

## The pipeline

```bash
source env.sh
# 1. the training pools, simulated once (designs in the +-10% box, flow muons at a raised momentum temperature)
GPUS="0 1" bash lcso/simulate_pools.sh
# 2. the surrogate: four seeds on all pools, averaged into one network (outputs/lcso.pt)
P=T1.5_d1+T1.5_d0.5+T1.5_d2+T1_d1+T2_d1+T2_m1e6+T1.5_u1+T1.5_adapt
for s in 0 1 2 3; do python -m lcso.train --pool $P --hidden 128 --p 32 --seed $s --name lcso_s$s; done
python -m lcso.train --stack outputs/lcso_s{0,1,2,3}.pt --name lcso
#    the Taylor surrogate the same way: --kind taylor-res4:8 --near 0.3 --p_res 32
# 3. the robustness analysis, ~4 h of simulation on 2-3 GPUs (FAST=1: fewer muons and designs)
nohup bash run_analysis.sh > outputs/logs/run_analysis.log 2>&1 &
```

The adaptive pool `T1.5_adapt` is chosen from a first surrogate trained on the other pools and its
tolerance-box results (`python -m lcso.adaptive_designs`). `run_analysis.sh` runs, all writing to
`outputs/robustness/`:

| script | what it does | writes |
|---|---|---|
| `robustness.py` | sensitivities, Hessian, active subspace, worst directions, random designs, walks checked with the simulator | `robustness.npz`, `robustness_report.txt` |
| `worst_cases.py` | concrete designs (worst direction scaled to X%, its top-k parameters, single parameters, box worst case, random tolerances), simulated | `worst_cases.npz`, `worst_cases_report.txt` |
| `per_momenta.py` | the sensitivity split by p_z, p_t or charge | `<var>_bins.npz`, `<var>_bins_report.txt` |
| `tolerance_boxes.py` | the worst design within +-X% on every parameter (searched on the surrogate), a greedy tolerance allocation for worst-case targets, both simulated (`--sim`, split over GPUs) | `tolerance_boxes.npz`, `tolerance_sim_*.npz`, `tolerance_boxes_report.txt` |
| `scan_sim.py` | optional step `scans`: the one-at-a-time scans simulated at +-5% and +-10% | `scan_sim_<worker>.npz` |
| `plots.py` | every figure, from the saved results | `figs/*.png` |

The surrogate is integrated over the muon flow on a quadrature grid (`MuonShield.grid`: the flow's
2-d momentum latent x azimuth x charge, with Gaussian weights), not sampled: exact to ~1e-5 in
seconds, where 1e9 sampled muons leave 2-5% of noise. Errors on surrogate quantities are the
scatter of the ensemble's members; only the simulator uses sampled muons. `--no_sim` runs
`robustness.py` and `per_momenta.py` on the surrogate alone (minutes).

## Code

| path | contents |
|---|---|
| `lcso/` | the surrogate: architectures (`models.py`), the analysis' view of a trained one (`surrogate.py`: `LCSONet`, `ModelSurrogate`, `load_surrogate`, `stack`), training (`training.py`, `train.py`), the training pools (`data.py`, `simulate_pools.py`, `adaptive_designs.py`) |
| `shield.py` | `MuonShield`: the 43 free parameters, flow muons, the simulator call |
| `analysis.py` | shared machinery of the analysis scripts: derivatives from one pass over the grid, cached hits, simulator passes |
| `surrogate_study/` | how good the surrogates are and which choices matter: toy problems with an exact answer, the shield against simulated reference designs, the architecture and training comparisons (`surrogate_study/README.md`) |
| `report/` | the three reports |
| `experiments/` | the earlier DRO / LCSO study on toy problems and its muon-shield port (`experiments/README.md`) |

## What the studies say (details in `surrogate_study/README.md` and the comparison report)

- Per-muon surrogates are what make this work: a response surface of the hit count alone needs
  ~10x more simulation for the same gradient.
- Train on many designs with few muons each, all at once, at a raised momentum temperature, and
  on every pool available: the surrogate models p(hit | design, muon), so muons from any
  temperature are valid data. Designs far from the nominal one help only when chosen adaptively
  (where the conclusions rest and where the ensemble disagrees).
- Directions (gradient, top Hessian eigenvector, worst direction) are well determined; magnitudes
  away from the nominal design are not quadratic, so they are always simulated. A surrogate is
  least reliable on the designs it selects as worst.
- DeepONet vs. Taylor: both converge to the exact derivatives on the toys (the DeepONet faster at
  few samples, the Taylor surrogates more accurate at many) and give the same robustness
  conclusions on the shield; the DeepONet predicts magnitudes more accurately.
