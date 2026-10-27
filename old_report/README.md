# The first analysis (superseded)

The first robustness analysis (2026-10-19 to 24): 43 free parameters (the magnet heights fixed)
and every muon on axis at z = -1 m. The second analysis (`report/`) adds the heights and the
muons' production points and replaces it; its conclusions are the same.

| report | contents |
|---|---|
| `muon_shield_robustness.pdf` | the analysis with the DeepONet surrogate |
| `taylor/muon_shield_robustness_taylor.pdf` | the same analysis with the Taylor surrogate (`taylor/build.py` assembles it from the main report's model-independent sections) |
| `comparison/surrogate_comparison.pdf` | the two surrogates compared: convergence to the exact derivatives on toy problems, and the robustness conclusions on the shield |

The code that made them is commit `39adfff`; its results, models and pools are in
`outputs/archive/round1_43p_onaxis/` (not tracked). The figures (`figs/`) are not tracked either:
they are copied from there.
