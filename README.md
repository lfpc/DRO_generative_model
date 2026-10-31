# Robustness of the SHiP muon shield

How much do the muon hits change when the 65 free parameters of the optimised muon shield (the
hadron absorber M0 and magnets M1-M7, with their heights and yoke heights) deviate from their
nominal values? A surrogate of the per-muon
hit probability, LCSO (`p(hit | design, muon)`, a DeepONet), is trained once on simulated
designs; the robustness quantities are read off it (sensitivities, Hessian, active subspace,
worst directions, the worst design for a given tolerance, a tolerance allocation); and every
conclusion is checked with the simulator.

The report is `report/muon_shield_robustness.tex` (`latexmk -pdf muon_shield_robustness.tex` in
`report/`). Its figures in `report/figs/` are PDFs that the plotting scripts write there directly
(`analysis.plots`, `validation.validate`, `experiments/quadrature*.py`, each with a quick redraw
from saved results); every figure also goes, as PNG, to `outputs/figures/`.

## Setup

`source env.sh` first: the `muonshield` conda env, this repository and the simulator
(`cuda_muons_ship`, from MuonsAndMatter; needs a GPU) on the path. The muons come from the
generative flow of `~/projects/generator_muons`. Locations are set in `paths.py` and can be
changed with environment variables (`RMS_OUTPUTS`, `PROJECTS_DIR`, `MUON_FLOW`). Every script
takes `--help`, which lists each flag with its default.

The simulator runs in the magnetic field map of each design, from snoopy's FEM (`--fields map`,
the default; `--fields uniform`: a uniform field per iron block). A map takes ~20 s on 8 CPU cores,
in a separate process, and is computed in the background while the GPU simulates the previous
designs. The analysis keeps the maps in `outputs/data/field_maps/` (delete them when snoopy
changes); pools and references delete each map after use.

## Running

```bash
nohup bash run_all.sh > outputs/logs/run_all.log 2>&1 &       # everything, on GPUs 0 and 1 (~12.5 h)
nohup bash run_analysis.sh > outputs/logs/run_analysis.log 2>&1 &   # the analysis of an existing surrogate (~5 h)
SIM=0 bash run_analysis.sh                                     # the same on the surrogate alone (minutes)
```

`run_all.sh` (resumable: every step skips what is done):

1. simulates the training pools (`lcso/simulate_pools.py`): many designs around the nominal one
   (scrambled Sobol, pulled towards it), each with its own muons at a raised momentum temperature;
   every hit and 0.4% of the misses are kept;
2. simulates the reference designs of the validation on stored muons (`validation/references.py`);
3. trains a first four-seed ensemble (`lcso/train.py`), finds its tolerance-box worst cases and
   simulates an adaptive pool there (`lcso/adaptive_designs.py`);
4. trains the final ensemble on all pools: `outputs/models/lcso.pt`;
5. runs the analysis (`run_analysis.sh`) and the validation (`validation/validate.py`).

The settings (muon source, pools, architecture, simulated muons) are at the top of `run_all.sh`.

## Code

| path | contents |
|---|---|
| `shield.py` | the design problem: the 65 free parameters, the tolerance box (`u_i = +-1` is +-10%), the field maps, the simulator call |
| `muons.py` | muon sources: the generative flow (sampled, or as a quadrature grid), or stored muons (`--muons file:<muons.npy>`); another generator plugs in as a class with `sample()` |
| `paths.py`, `cli.py` | where things are; the shared command-line options (`--model`, `--muons`, `--device`) |
| `lcso/` | the surrogate: architectures (`models.py`), the trained model as the analysis sees it (`surrogate.py`), training (`training.py`, `train.py`), the pools (`data.py`, `simulate_pools.py`, `adaptive_designs.py`) |
| `analysis/` | the robustness analysis (`python -m analysis.<script>`): `robustness`, `worst_cases`, `bins`, `tolerance_boxes`, `scan_sim`, `plots`; shared machinery in `core.py` |
| `validation/` | reference designs simulated on stored muons, and the surrogate scored on them (`validate.py`); the surrogate scored on every design the analyses simulated at the physical flux (`flux_check.py`) |
| `experiments/` | the studies that chose and checked the method (quadrature vs. sampling, architectures, training settings, derivative convergence) |
| `report/` | the report |
| `DRO/` | the earlier latent-space DRO / LCSO project (toy problems, paper), independent of the above |

The surrogate is integrated over the muons on a quadrature grid (the flow's momentum latent x
azimuth x charge, one production point per node drawn with a fixed seed), not sampled: within
~1e-3 of a much finer grid in seconds, where 1e9 sampled muons leave ~1% of noise
(`experiments/quadrature.py`). Only the simulator uses sampled muons. Errors on surrogate
quantities are the scatter of the ensemble's members.

## Results

Under `outputs/` (not tracked; `paths.py`):

    models/     the surrogates: lcso.pt and its seeds, lcso_first.pt (the first ensemble)
    data/       simulations: pools/ (training), references/ (validation), adaptive_designs.npy,
                field_maps/ (one folder per design)
    results/    numbers and text reports: robustness/, robustness_first/, validation/, experiments/
    figures/    every figure: robustness/, validation/, experiments/, muons/
    logs/       the launchers' logs

The products of the 50-parameter analysis (the one in the report) carry the suffix `_p50`
(`models/lcso_p50.pt`, `results/robustness_p50/`, ...); its pools are `data/pools/r2_*`, those of
the first analysis (43 parameters) `r1_*`. Both were simulated with M4's exit at nominal (M4 tapered
when its entrance moved); from the 65-parameter analysis on M4 is rectangular, so they are not
trained on, and the references' hits of that geometry are kept as `shard_p50_*.pt`.
