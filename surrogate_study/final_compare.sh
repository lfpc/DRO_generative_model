#!/bin/bash
# Candidate production surrogates at the final setup: every candidate trained on all pools with
# SEEDS seeds (lcso/train.py), stacked into one ensemble each, then compared with the
# DeepONet ensemble outputs/lcso.pt (final_compare.py). Jobs are spread over GPUS, one per GPU
# at a time; a trained seed is not retrained. CANDIDATES: one "name kind [extra args]" per line.
#     GPUS="1 2 3" nohup bash surrogate_study/final_compare.sh > outputs/surrogate_study/logs/final_compare.log 2>&1 &
#     CANDIDATES="dpo128 deeponet --p 128
#     mlp mlp:3 --hidden 256" GPUS="1 2 3" bash surrogate_study/final_compare.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
GPUS=(${GPUS:-0 1 2 3})
SEEDS=${SEEDS:-4}
POOL=${POOL:-T1.5_d1+T1.5_d0.5+T1.5_d2+T1_d1+T2_d1+T2_m1e6}
COMMON="--pool $POOL --hidden 128 --p 32 --p_res 32"
DEFAULT="tsub4 taylor-sub:4
tsub8 taylor-sub:8
tsubres4 taylor-sub-res:4
tsubres8 taylor-sub-res:8
taylor8 taylor:8"
mapfile -t CANDIDATES <<< "${CANDIDATES:-$DEFAULT}"
mkdir -p outputs/surrogate_study/logs

JOBS=()
for c in "${CANDIDATES[@]}"; do
  read -r name kind extra <<< "$c"
  for s in $(seq 0 $((SEEDS - 1))); do
    [[ -f outputs/${name}_s$s.pt ]] || JOBS+=("$name $kind $s $extra")
  done
done
for k in "${!GPUS[@]}"; do
  (
    for ((j = k; j < ${#JOBS[@]}; j += ${#GPUS[@]})); do
      read -r name kind s extra <<< "${JOBS[$j]}"
      CUDA_VISIBLE_DEVICES=${GPUS[$k]} python -W ignore -m lcso.train $COMMON $extra --kind "$kind" --seed "$s" \
        --name "${name}_s$s" > "outputs/surrogate_study/logs/train_${name}_s$s.log" 2>&1
    done
  ) &
done
wait

names=lcso
for c in "${CANDIDATES[@]}"; do
  read -r name kind extra <<< "$c"
  python -W ignore -m lcso.train --stack $(for s in $(seq 0 $((SEEDS - 1))); do echo outputs/${name}_s$s.pt; done) \
    --name "$name" --device cpu
  names="$names,$name"
done
CUDA_VISIBLE_DEVICES=${GPUS[0]} python -W ignore surrogate_study/final_compare.py --candidates "$names"
echo "done"
