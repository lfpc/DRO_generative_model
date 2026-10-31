#!/bin/bash
# The robustness analysis of a trained surrogate (no training), on one or two GPUs. Everything goes
# to $RUN (the .npz results and *_report.txt files) and outputs/figures/<basename of $RUN>/. A step is skipped when its result
# exists (delete it to redo the step).
#
#     nohup bash run_analysis.sh > outputs/logs/run_analysis.log 2>&1 &
#     MODEL=outputs/models/other.pt RUN=outputs/results/robustness_other GPUS="0" bash run_analysis.sh
#     SIM=0 bash run_analysis.sh       # the surrogate alone, no simulation (minutes)
#     FAST=1 bash run_analysis.sh      # 3e8 simulated muons and fewer concrete designs (< 1 h)
#
# Two lanes, one per GPU (one after the other on one GPU, or when both are the same GPU: a
# simulation takes most of its memory):
#   A  robustness (derivatives, directions, simulated walks; ~3 h for 1e9 muons), worst_cases
#      (concrete designs, simulated), scans (the one-at-a-time scans simulated, shared with B)
#   B  bins (the sensitivity split by p_z, p_t and charge; surrogate only), tolerance_boxes (the box
#      worst cases and the allocation, searched on the surrogate, then simulated), scans
# then the tolerance report and every figure (analysis/plots.py).
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
MODEL=${MODEL:-outputs/models/lcso.pt}
RUN=${RUN:-outputs/results/robustness}
MUONS=${MUONS:-flow}
N=${N:-1000000000}                   # simulated muons (~1200 hits at the nominal design)
GPUS=(${GPUS:-0 1})
W=${#GPUS[@]}
WC=""
if [[ "${FAST:-0}" == 1 ]]; then
  N=300000000
  WC="--pcts 1,2,5,10,-10 --top_k 1,3,10 --n_single 5 --random 2:4,5:4,10:4"
fi
NOSIM=""
[[ "${SIM:-1}" == 0 ]] && NOSIM="--no_sim"
M="--model $MODEL --muons $MUONS"
mkdir -p "$RUN"
py() { echo "$(date '+%H:%M') $*"; python -W ignore -m "$@"; }

lane_a() {
  export CUDA_VISIBLE_DEVICES=${GPUS[0]}
  [ -f $RUN/robustness.npz ] || py analysis.robustness $M --n_muons_sim $N --out_dir $RUN $NOSIM
  [ -f $RUN/worst_cases.npz ] || py analysis.worst_cases $M --run $RUN --n_muons $N $WC $NOSIM
  [ -n "$NOSIM" ] || [ -f $RUN/scan_sim_0.npz ] || py analysis.scan_sim $M --run $RUN --n_muons $N --worker 0 --n_workers $W
}
lane_b() {
  export CUDA_VISIBLE_DEVICES=${GPUS[$((W - 1))]}
  B="$M --no_sim --out_dir $RUN"
  [ -f $RUN/pz_bins.npz ] || py analysis.bins $B --bin_var pz --edges 10,20,30,45,75,125,175,225,300
  [ -f $RUN/pt_bins.npz ] || py analysis.bins $B --bin_var pt --edges 0.25,0.5,0.75,1,1.5
  [ -f $RUN/charge_bins.npz ] || py analysis.bins $B --bin_var charge
  until [ -f $RUN/robustness.npz ]; do sleep 120; done           # the box search starts from its worst direction
  [ -f $RUN/tolerance_boxes.npz ] || py analysis.tolerance_boxes $M --run $RUN
  [ -n "$NOSIM" ] || [ -f $RUN/tolerance_sim_0.npz ] || py analysis.tolerance_boxes $M --run $RUN --sim --n_muons $N
  [ -n "$NOSIM" ] || [ $W == 1 ] || [ -f $RUN/scan_sim_1.npz ] || py analysis.scan_sim $M --run $RUN --n_muons $N --worker 1 --n_workers 2
}
if [ $W == 1 ] || [ "${GPUS[0]}" == "${GPUS[$((W - 1))]}" ]; then
  lane_a && lane_b
else
  lane_a & A=$!; lane_b & B=$!
  wait $A; wait $B
fi
CUDA_VISIBLE_DEVICES=${GPUS[0]} py analysis.tolerance_boxes $M --run $RUN         # the report, with the simulations
py analysis.plots $RUN
{
  echo "date:   $(date)"
  echo "host:   $(hostname)"
  echo "commit: $(git rev-parse --short HEAD) $(git diff --quiet || echo '(uncommitted changes)')"
  echo "model:  $MODEL ($(stat -c %y "$MODEL"))"
  echo "muons:  $MUONS; $N simulated (SIM=${SIM:-1}, FAST=${FAST:-0})"
} > "$RUN"/MANIFEST.txt
echo "done: $RUN"
