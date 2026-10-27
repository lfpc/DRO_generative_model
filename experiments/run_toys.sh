#!/bin/bash
# Every toy job (run_toys.py): PER_GPU workers on each GPU plus CPUS workers on the CPU (4 threads
# each), all pulling from one queue. Resumable: rerun to finish (a launch clears stale claims).
#     GPUS="0 1" PER_GPU=5 CPUS=24 nohup bash experiments/run_toys.sh > outputs/experiments/logs/toys.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
GPUS=(${GPUS:-0 1})
PER_GPU=${PER_GPU:-4}
rm -rf outputs/experiments/toys/claims
mkdir -p outputs/experiments/logs
k=0
for gpu in "${GPUS[@]}"; do
  for _ in $(seq "$PER_GPU"); do
    CUDA_VISIBLE_DEVICES=$gpu python -W ignore experiments/run_toys.py --worker g$k \
      > outputs/experiments/logs/toys_$k.log 2>&1 &
    k=$((k + 1))
  done
done
for k in $(seq 0 $(( ${CPUS:-0} - 1 ))); do
  CUDA_VISIBLE_DEVICES= python -W ignore experiments/run_toys.py --worker c$k --device cpu --threads 4 \
    > outputs/experiments/logs/toys_c$k.log 2>&1 &
done
wait
echo "all toy jobs done"
