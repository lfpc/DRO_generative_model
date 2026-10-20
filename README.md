# Robustness of the SHiP muon shield

Train a surrogate of the muon shield's hits (LCSO, DeepONet), then read robustness off it:
sensitivities, Hessian, active subspace and worst-case directions, each checked with the simulator.
The write-up is `report/muon_shield_robustness.tex`.

`source env.sh` first: the `muonshield` conda env, and the simulator (`cuda_muons_ship`, from
MuonsAndMatter, needs a GPU) on the path. The muons come from the generative flow in
`~/projects/generator_muons` (`src.flows.load_generator`); set `PROJECTS_DIR` if your projects
are not in `~/projects`.

## The pipeline

```bash
source env.sh
# 1. simulate training pools once (designs in the +-10% box, flow muons at a raised momentum temperature)
GPUS="0 1 2 3" bash surrogate_study/run_shield_pool.sh
# 2. the surrogate: four seeds on all pools, averaged into one network (outputs/lcso.pt)
P=T1.5_d1+T1.5_d0.5+T1.5_d2+T1_d1+T2_d1+T2_m1e6
for s in 0 1 2 3; do python train_surrogate.py --pool $P --hidden 128 --p 32 --seed $s --name lcso_s$s; done
python train_surrogate.py --stack outputs/lcso_s{0,1,2,3}.pt --name lcso --score
# 3. the robustness analysis, ~6 h of simulation (FAST=1: ~2 h)
nohup bash run_analysis.sh > outputs/logs/run_analysis.log 2>&1 &
```

`run_analysis.sh` runs, all writing to `outputs/robustness/`:

| script | what it does | writes |
|---|---|---|
| `robustness.py` | sensitivities, Hessian, active subspace, worst directions, random designs, walks checked with the simulator | `robustness.npz`, `robustness_report.txt` |
| `worst_cases.py` | concrete designs (worst direction scaled to X%, its top-k parameters, single parameters, box worst case, random tolerances), simulated | `worst_cases.npz`, `worst_cases_report.txt` |
| `per_momenta.py` | the sensitivity split by p_z, p_t or charge | `<var>_bins.npz`, `<var>_bins_report.txt` |
| `plots.py` | every figure, from the saved results | `figs/*.png` |

The surrogate is integrated over the muon flow on a quadrature grid (`MuonShield.grid`: the
flow's 2-d momentum latent x azimuth x charge, with Gaussian weights), not sampled: exact to
~1e-5 in seconds, where 1e9 sampled muons left 2-5% of noise. Errors on surrogate quantities are
the scatter of the ensemble's members; only the simulator uses sampled muons. `--no_sim` runs
`robustness.py` and `per_momenta.py` on the surrogate alone (minutes). Every script takes
`--help`, which lists each flag with its default.

## Code

| file | contents |
|---|---|
| `shield.py` | `MuonShield`: the 43 free parameters, flow muons, the simulator call |
| `surrogate.py` | `LCSONet` (DeepONet on (px, py, pz, charge)), `stack` (an ensemble as one network), `load_surrogate` |
| `train_surrogate.py` | fit `LCSONet` on simulated pools, or stack checkpoints |
| `analysis.py` | shared machinery: the frozen-trunk derivatives, cached hits, surrogate-vs-simulator passes |
| `surrogate_study/` | how good are the surrogates, and which choices matter: toy problems with an exact answer and the muon shield against the simulator (`surrogate_study/README.md`) |
| `experiments/` | the DRO / LCSO study on toy problems and its muon-shield port (`experiments/README.md`) |

## What the surrogate study says (details in `surrogate_study/README.md`)

- Per-muon surrogates are what make this work: a response surface of the hit count alone needs
  ~10x more simulation for the same gradient.
- Train on many designs with few muons each, all at once (not a rolling buffer), at a raised
  momentum temperature, and on every pool available: the surrogate models p(hit | design, muon),
  so muons from any temperature are valid data.
- With ~2e8 training rows, capacity matters (128 hidden, 32 features), and averaging seeds
  removes most of what is left of the scatter of the gradient.
- Directions (gradient, top Hessian eigenvector, worst direction) are well determined; magnitudes
  away from the nominal design are not quadratic, so they are always simulated.
