# Experiments

Latent-space DRO and LCSO, on toy problems and on the muon shield. Everything runs on the
CPU, except the muon shield's simulator, which needs a GPU.

## Problems

Each folder holds the same scripts, run from inside the folder:

| script | what it does |
|---|---|
| `problem.py` | the loss, the true distribution, the design box |
| `train_generator.py` | fit (or load) the generative flow -> `models/generator.pt` |
| `run_inner_cost.py` | inner loop at a fixed design: cost of each adversary; fits `models/surrogate.pt` |
| `run_inner_samples.py` | worst-case samples of each adversary |
| `run_inner.py` | worst-case risk vs. radius (portfolio, two moons, muon shield) |
| `train_lcso.py` | optimise the design through an LCSO surrogate |

Problems: `short_column`, `beam_shield`, `newsvendor`, `portfolio_optimization`, `two_moons`,
`mnist` (standalone), and **`muon_shield`**, the real target. For the muon shield, see
`muon_shield/README.md`: `bash muon_shield/run_all.sh` runs everything.

## Cross-problem studies

```bash
python3 outer/run_outer.py --problem short_column --surrogate taylor   # DRO outer loop
python3 inner_compare/run_compare.py --problem short_column            # adversaries at equal risk
python3 inner_compare/combine.py
python3 integration/quasi-MC.py --problem two_moons                    # MC vs QMC vs sparse grids
python3 integration/quantify.py --problem two_moons
```

## LCSO surrogates (`optb/`)

```bash
python3 optb/compare_lcso.py --problem short_column   # every surrogate x {resample, nominal} x seeds
python3 optb/gradcheck.py --problem short_column      # do the gradient and Hessian converge?
python3 optb/robustcheck.py --problem short_column    # are the robustness conclusions right?
```

Add `--plot_only` to redraw from the saved JSON. Surrogate kinds: `taylor` (rank-8 quadratic),
`taylor-full` (full-rank quadratic re-centred on the incumbent), `taylor-res[:lambda[:keep|reset|fold]]`
(the same, plus a residual net beyond second order), `mlp`/`deeponet` (branch-trunk).

## Libraries

| file | contents |
|---|---|
| `dro/flows.py` | the generator: a RealNVP-style flow |
| `dro/transforms.py` | latent transforms and their KL to the base |
| `dro/surrogate.py` | branch-trunk and Taylor surrogates |
| `dro/method.py` | `trust_region_max`, `optimize_design` |
| `dro/baselines.py` | `kl_weights`, `erm` |
| `optb/lcso.py` | LCSO: `lcso`, `Budget`, `_fit`, `_make_surrogate` |
| `optb/samplers.py` | Sobol / uniform design draws |
