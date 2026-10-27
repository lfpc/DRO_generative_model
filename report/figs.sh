#!/bin/bash
# Collect the report's figures into report/figs/ (tracked) from the results, then build the report:
#     bash report/figs.sh && (cd report && latexmk -pdf muon_shield_robustness.tex)
set -euo pipefail
cd "$(dirname "$0")"
O=${RMS_OUTPUTS:-../outputs}
mkdir -p figs
cp $O/figures/robustness/*.png figs/                                      # run_analysis.sh
for f in hits_T1 hits_T1.5 calibration_T1 calibration_T1.5; do cp $O/figures/validation/$f.png figs/validation_$f.png; done   # validation/validate.py
cp $O/figures/validation/oat.png figs/validation_oat.png
cp $O/figures/experiments/quadrature.png $O/figures/experiments/quadrature_convergence.png figs/   # experiments/quadrature*.py
R1=$O/archive/round1_43p_onaxis/report/figs                              # App. C: the first analysis' study
cp $R1/study_toy_budget.png figs/round1_toy_budget.png
cp $R1/study_shield_budget_final.png figs/round1_shield_budget.png
for f in figs/*.png; do                                                  # only what the report shows
  grep -q "{$(basename $f)}" muon_shield_robustness.tex || rm $f
done
echo "$(ls figs | wc -l) figures in $(pwd)/figs"
