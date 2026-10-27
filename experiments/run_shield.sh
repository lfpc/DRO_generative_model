#!/bin/bash
# Every muon-shield fit (run_shield.py), PER_GPU workers on each GPU. Resumable: rerun to finish.
# Each worker holds both references in pinned host memory (~22 GB).
#     GPUS="0 1" PER_GPU=2 nohup bash experiments/run_shield.sh > outputs/experiments/logs/shield.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
GPUS=(${GPUS:-0 1})
PER_GPU=${PER_GPU:-2}
rm -rf outputs/experiments/shield/claims
mkdir -p outputs/experiments/logs
k=0
for gpu in "${GPUS[@]}"; do
  for _ in $(seq "$PER_GPU"); do
    CUDA_VISIBLE_DEVICES=$gpu python -W ignore experiments/run_shield.py --worker g$k ${SKIP_POOLS:+--skip_pools $SKIP_POOLS} \
      > outputs/experiments/logs/shield_$k.log 2>&1 &
    k=$((k + 1))
  done
done
wait
echo "all shield jobs done"
