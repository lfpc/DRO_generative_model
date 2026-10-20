#!/bin/bash
# Every muon-shield fit (run_shield.py), PER_GPU workers on each GPU. Resumable: rerun to finish.
# Each worker holds both references in pinned host memory (~22 GB).
#     GPUS="0 2 3" PER_GPU=2 nohup bash surrogate_study/run_shield.sh > outputs/surrogate_study/logs/shield.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
GPUS=(${GPUS:-0 1 2 3})
PER_GPU=${PER_GPU:-2}
rm -rf outputs/surrogate_study/shield/claims
mkdir -p outputs/surrogate_study/logs
k=0
for gpu in "${GPUS[@]}"; do
  for _ in $(seq "$PER_GPU"); do
    CUDA_VISIBLE_DEVICES=$gpu python -W ignore surrogate_study/run_shield.py --worker g$k ${SKIP_POOLS:+--skip_pools $SKIP_POOLS} \
      > outputs/surrogate_study/logs/shield_$k.log 2>&1 &
    k=$((k + 1))
  done
done
wait
echo "all shield jobs done"
