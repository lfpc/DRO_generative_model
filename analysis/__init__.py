"""The robustness analysis of a trained surrogate (python -m analysis.<script>; run_analysis.sh runs all):

  core.py             shared machinery: derivatives from one pass over the muon points, cached hits,
                      simulator passes, the trust-region worst direction
  robustness.py       sensitivities, Hessian, active subspace, worst directions, random designs, walks
  worst_cases.py      concrete worst-case designs, simulated
  bins.py             the sensitivity split by p_z, p_t or charge
  tolerance_boxes.py  the worst design within +-X% on every parameter, a tolerance allocation
  scan_sim.py         the one-at-a-time scans, simulated
  plots.py            every figure, from the saved results
"""
