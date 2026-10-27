# Muon shield

The SHiP muon shield of `../../../../MuonShieldChallenge`, simulated with `cuda_muons` (needs an
NVIDIA GPU and `faster_muons_torch`, see `MuonShieldChallenge/README.md`). The muon generator is
the pretrained flow of `../../../../generator_muons`.

The loss is `N_mu × (hit + constraint penalty)` per muon, so every risk reads as the expected
number of hits (N_mu = 2.7e10 weighted muons). The simulator has no gradient: the Wasserstein
and latent adversaries ascend a surrogate, and every risk they report is re-simulated.

## Run everything

```bash
bash DRO/experiments/muon_shield/run_all.sh
STAGES="inner compare" bash DRO/experiments/muon_shield/run_all.sh        # only some stages
MUON_SHIELD_CONFIG=hard CUDA_VISIBLE_DEVICES=1 bash DRO/experiments/muon_shield/run_all.sh
```

Logs go to `logs/<step>.log`, a one-line summary per step to `logs/STATUS`. Sample sizes are
variables at the top of `run_all.sh` (`N=1e8` for every estimate).

## The steps, in order

| stage | command | output |
|---|---|---|
| generator | `python train_generator.py` (CPU) | `models/generator.pt`, `models/data.pt`, `figs/generator.png` |
| check | `python problem.py` | hit rate and constraints of the reference design |
| inner | `python run_inner_cost.py` | `models/design.pt`, `models/surrogate.pt`, `figs/inner_cost.png` |
| | `python run_inner_samples.py` | `figs/inner_samples.png` |
| | `python run_inner.py` | `figs/inner.png`, `figs/inner.json` |
| lcso | `python train_lcso.py` | `figs/lcso.png`, `figs/lcso_{resample,nominal}.txt` (designs) |
| compare | `../inner_compare/run_compare.py --problem muon_shield` | `../inner_compare/` |
| outer | `../outer/run_outer.py --problem muon_shield` | `../outer/figs/outer_muon_shield.png` |
| integration | `../integration/quasi-MC.py`, `quantify.py --problem muon_shield` | `../integration/figs/` |

`run_inner_cost.py` must run before everything after it: it fixes the design and the surrogate
the later studies load. Every script takes `--help`.

## Settings

At the top of `problem.py`, read by every script including the shared ones:

- `CONFIG`: `easy` (30 parameters) or `hard` (72); env `MUON_SHIELD_CONFIG`. Changing it changes
  the design dimension, so rerun from `run_inner_cost.py`.
- `BACKEND`: `cuda`, or `toy` for a crude CPU transport that only smoke-tests the pipeline
  (its numbers mean nothing); env `MUON_SHIELD_BACKEND`.
- `NOMINAL`: `gmm` (the challenge's `MuonsGenerator`, exact density) or `pool` (real muons
  resampled by weight; run `python build_pool.py` first, it reads ~74 GB once).
- `CHUNK` (1e7 muons), `N_MU`, and the paths to `MuonShieldChallenge` and `generator_muons`.

CPU smoke test:

```bash
MUON_SHIELD_BACKEND=toy python problem.py --n_samples 1e5
```

## Representation

A muon is `(x, y)`: six standardised features (logit |p|, log-radial slopes px/pz and py/pz,
production x and y, logit of the Beta variable behind z) and the charge as a label. Every point
of R^6 is a valid muon. `problem.log_prob` is the `MuonsGenerator` density in these features:
the truth under `gmm`, an independent judge under `pool`. The flow (`train_generator.PretrainedFlow`)
maps an 8-D normal to these features: six flow latents, which the latent adversary moves, plus
the global azimuth and the sign of dphi.
