# Robustness of the SHiP muon shield

Train a surrogate of the muon shield's hits (LCSO), then read robustness off it: sensitivities,
Hessian, active subspace and worst-case directions, each checked with the simulator.

The simulator (`cuda_muons_ship`) needs a GPU. The muons come from the generative flow in
`~/projects/generator_muons` (`--muons gen`, the default) or from the real data file (`--muons data`).
Set `PROJECTS_DIR` if your projects are not in `~/projects`. Everything writes to `outputs/`.

## The pipeline

```bash
# 1. train the surrogate
python lcso_resample.py --model_type deeponet      # -> outputs/lcso_resample_deeponet.pt
python lcso.py                                     # alternative: fixed designs -> outputs/lcso_model.pt
python test_lcso.py --model outputs/lcso_model.pt  # predicted vs. true hits

# 2. robustness of the nominal design
python robustness.py --model outputs/lcso_resample_deeponet.pt               # main analysis
python robustness_per_momenta.py --model outputs/lcso_resample_deeponet.pt   # split by muon kinematics
```

`--no_sim` skips the simulator, so step 2 can be run with the surrogate alone.
`--model_type taylor` (per-muon quadratic) works with every script as well.

## Other checks

| script | what it does |
|---|---|
| `grad_test.py` | walk the descent direction, surrogate vs. simulator |
| `active_subspace_test.py` | active subspace of log hits, walk its top directions |
| `optimization_test.py` | one steepest-descent / trust-region Newton step, checked with the simulator |
| `zero_order_robustness.py` | 1-D scan of one parameter with the simulator |
| `gp.py` | Gaussian-process baseline on total hits |

Every script takes `--help`, which lists each flag with its default.

## Code

| file | contents |
|---|---|
| `problems.py` | `ShipMuonShieldCuda`: the shield parameters, muon sampling, the simulator call |
| `models.py` | the LCSO surrogates (DeepONet, Taylor) and `load_surrogate` |
| `trainer.py` | `train_hits_classifier` |
| `muons_io.py` | muon blocks and the surrogate-vs-simulator sweep shared by the test scripts |

## What works for the robustness analysis

From the toy-problem study (`experiments/optb/robustcheck.py`), which compares each surrogate's
directions with ones found by simulation:

- The **active subspace** and the **gradient** directions are reliable. They find at least
  0.95 of the worst-case rise in almost every case.
- **Hessian eigen-directions** are only useful when the curvature is large compared with the
  tolerance. At a flat optimum or a small tolerance they are no better than random.
- **Projected-gradient ascent** and **corner** search on the surrogate find the worst design
  in the box reliably. Always confirm those few designs with the simulator.
- Non-smooth responses (sharp ridges at an optimum) defeat every smooth surrogate.

`experiments/` holds the DRO / LCSO paper study on toy problems and its muon-shield port;
see `experiments/README.md`.
