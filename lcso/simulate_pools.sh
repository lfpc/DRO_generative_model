#!/bin/bash
# Simulate every training pool of the surrogate, one worker per GPU. Resumable: a worker that
# crashes (the simulator can fail on extreme designs) is restarted, and a design that crashes it
# twice is skipped (lcso/simulate_pools.py).
#     GPUS="0 1" nohup bash lcso/simulate_pools.sh > outputs/surrogate_study/logs/pool.log 2>&1 &
# The adaptive pool needs outputs/surrogate_study/adaptive_D.npy (python -m lcso.adaptive_designs),
# which needs a first surrogate trained on the other pools and its tolerance-box run.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
GPUS=(${GPUS:-0 1 2 3})
W=${#GPUS[@]}
# name, momentum temperature, delta, designs, seed, muons per design, extra options;
# override with POOLS="..." (one per line)
DEFAULT_POOLS="T1.5_d1 1.5 1 2048 0 10000000
T1.5_d0.5 1.5 0.5 512 1 10000000
T1.5_d2 1.5 2 512 2 10000000
T1_d1 1.0 1 512 3 10000000
T2_d1 2.0 1 512 4 10000000
T2_m1e6 2.0 1 10000 5 1000000
T1.5_u1 1.5 1 1024 7 10000000 --no_radial --n_nominal 0
T1.5_adapt 1.5 1 0 8 10000000 --designs outputs/surrogate_study/adaptive_D.npy --n_nominal 0"
mapfile -t POOLS <<< "${POOLS:-$DEFAULT_POOLS}"
mkdir -p outputs/surrogate_study/logs
for k in "${!GPUS[@]}"; do
  (
    for p in "${POOLS[@]}"; do
      read -r name T delta n seed mu extra <<< "$p"
      log="outputs/surrogate_study/logs/pool_${name}_${k}.log"
      for try in $(seq 1 100); do
        CUDA_VISIBLE_DEVICES=${GPUS[$k]} python -m lcso.simulate_pools --name "$name" --n_muons "$mu" \
          --temperature "$T" --delta "$delta" --n_designs "$n" --seed "$seed" --worker "$k" --n_workers "$W" $extra \
          2>&1 | grep --line-buffered -E "^design|^done|^resuming|^skipping|Error|error" >> "$log" || true
        grep -q "^done" "$log" && break
      done
    done
  ) &
done
wait
echo "all pools done"
