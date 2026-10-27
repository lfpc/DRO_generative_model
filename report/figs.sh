#!/bin/bash
# Collect the report's figures into report/figs/ from the results, then build the report:
#     bash report/figs.sh && (cd report && latexmk -pdf muon_shield_robustness.tex)
set -euo pipefail
cd "$(dirname "$0")"
O=${RMS_OUTPUTS:-../outputs}
mkdir -p figs
cp $O/robustness/figs/*.png figs/                                        # run_analysis.sh
cp $O/validation/calibration.png figs/validation_calibration.png         # validation/validate.py
cp $O/validation/oat.png figs/validation_oat.png
cp $O/experiments/figs/quadrature.png $O/experiments/figs/quadrature_convergence.png figs/   # experiments/quadrature*.py
R1=$O/archive/round1_43p_onaxis/report/figs                              # App. C: the first analysis' study
cp $R1/study_toy_budget.png figs/round1_toy_budget.png
cp $R1/study_shield_budget_final.png figs/round1_shield_budget.png
echo "figures in $(pwd)/figs"
