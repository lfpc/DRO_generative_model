# Experiments

Studies that chose and checked how the analysis integrates over the muons; none of them is part
of the pipeline (`run_all.sh`). Run from the repository root after `source env.sh`, e.g.
`python experiments/quadrature.py`. Results go to `outputs/results/experiments/`, figures to
`outputs/figures/experiments/`.

| script | question | answer |
|---|---|---|
| `quadrature.py` | sampling vs. the quadrature grid for Y, its gradient and Hessian | the analysis grids are within 4e-4 (801x64) and 2.5e-3 (201x64) of a much finer grid in Y; 1e9 sampled muons leave 0.7% |
| `quadrature_convergence.py` | the same, as running estimates of the reported quantities | the grid converges where Monte Carlo is still at the percent level |
| `importance.py` | would importance-sampling the simulator from a hotter flow help? | no: x1.0 at T = 1.5, the nominal hits sit in the latent's core |
| `proposal_check.py` | could a targeted training proposal beat the momentum temperature? | at most 1.74x effective muons over T = 2 (the bound q* ~ p^(2/3) pi^(1/3)); T = 1.5 gives 1.33x |

The choice of the surrogate itself (architectures, training settings, toy problems with an exact
answer) was made in the first analysis; its scripts and data are gone, and the report's appendix
keeps the conclusions. `lcso/models.py` still has every architecture (DeepONet, Taylor, quadratic,
MLP), which MuonShieldChallenge imports.
