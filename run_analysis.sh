#!/bin/bash
# The complete robustness analysis of a trained surrogate (no training). Needs the GPU.
# Everything goes to $RUN: the .npz results, the *_report.txt files and figs/.
#
#     nohup bash run_analysis.sh > outputs/logs/run_analysis.log 2>&1 &
#     FAST=1 nohup bash run_analysis.sh > outputs/logs/run_analysis.log 2>&1 &     # < 1 h
#     STEPS="bins plots" bash run_analysis.sh                                       # some steps only
#
# Steps: robustness (derivatives, directions, simulated walks; ~3 h for 1e9 simulated muons),
# worst_cases (concrete designs, simulated), bins (sensitivity split by p_z, p_t and charge,
# surrogate only, a minute each), plots. Optional: scans (the one-at-a-time scans simulated,
# scan_sim.py, ~2 h; add it with STEPS="scans plots" after a robustness run). The surrogate is integrated over the muon flow on
# quadrature grids (no sampling); only the simulator uses N sampled muons.
# FAST=1: 3e8 simulated muons and fewer concrete designs.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
MODEL=${MODEL:-outputs/lcso.pt}
RUN=${RUN:-outputs/robustness}
N=${N:-1000000000}                 # simulated muons (~800 hits at the nominal design)
STEPS=${STEPS:-"robustness worst_cases bins plots"}
if [[ "${FAST:-0}" == 1 ]]; then
  N=300000000
  WC="--pcts 1,2,5,10,-10 --top_k 1,3,10 --n_single 5 --random 2:4,5:4,10:4"
else
  WC=""
fi
mkdir -p "$RUN"
has() { [[ " $STEPS " == *" $1 "* ]]; }

if has robustness; then
  python robustness.py --model "$MODEL" --n_muons_sim "$N" --out_dir "$RUN"
fi
if has worst_cases; then
  python worst_cases.py --run "$RUN" --model "$MODEL" --n_muons "$N" $WC
fi
if has bins; then
  BINS="--model $MODEL --no_sim --out_dir $RUN"
  python per_momenta.py $BINS --bin_var pz --edges 10,20,30,45,75,125,175,225,300
  python per_momenta.py $BINS --bin_var pt --edges 0.25,0.5,0.75,1,1.5
  python per_momenta.py $BINS --bin_var charge
fi
if has scans; then
  python scan_sim.py --run "$RUN" --model "$MODEL" --n_muons "$N"
fi
if has plots; then
  python plots.py "$RUN"
fi
{
  echo "date:   $(date)"
  echo "host:   $(hostname)"
  echo "commit: $(git rev-parse --short HEAD) $(git diff --quiet || echo '(uncommitted changes)')"
  echo "model:  $MODEL ($(stat -c %y "$MODEL"))"
  echo "steps:  $STEPS; $N simulated muons; surrogate on quadrature grids; FAST=${FAST:-0}"
} > "$RUN"/MANIFEST.txt
echo "done: $RUN"
