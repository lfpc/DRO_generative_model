#!/bin/bash
# Simulate every muon-shield pool, one worker per GPU (~8 h on 3 GPUs). Resumable.
#     GPUS="0 2 3" nohup bash surrogate_study/run_shield_pool.sh > outputs/surrogate_study/logs/pool.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
GPUS=(${GPUS:-0 1 2 3})
W=${#GPUS[@]}
# name, momentum temperature, delta, designs, seed, muons per design; override with POOLS="..." (one per line)
DEFAULT_POOLS="T1.5_d1 1.5 1 2048 0 10000000
T1.5_d0.5 1.5 0.5 512 1 10000000
T1.5_d2 1.5 2 512 2 10000000
T1_d1 1.0 1 512 3 10000000
T2_d1 2.0 1 512 4 10000000
T2_m1e6 2.0 1 10000 5 1000000
T3_m1e6 3.0 1 10000 6 1000000"
mapfile -t POOLS <<< "${POOLS:-$DEFAULT_POOLS}"
mkdir -p outputs/surrogate_study/logs
for k in "${!GPUS[@]}"; do
  (
    for p in "${POOLS[@]}"; do
      read -r name T delta n seed mu <<< "$p"
      # a crash (the simulator can fail on extreme designs) must not stop the next pools
      CUDA_VISIBLE_DEVICES=${GPUS[$k]} python surrogate_study/shield_pool.py --name "$name" --n_muons "$mu" \
        --temperature "$T" --delta "$delta" --n_designs "$n" --seed "$seed" --worker "$k" --n_workers "$W" \
        2>&1 | grep --line-buffered -E "^design|^done|^resuming|Error|error" > "outputs/surrogate_study/logs/pool_${name}_${k}.log" || true
    done
  ) &
done
wait
echo "all pools done"
