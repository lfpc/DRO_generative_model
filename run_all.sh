#!/bin/bash
# The whole analysis, end to end on two GPUs: training pools, reference designs, a first surrogate
# and its adaptive pool, the final surrogate, the robustness analysis, the validation. Resumable:
# every step skips what is done.
#
#     nohup bash run_all.sh > outputs/logs/run_all.log 2>&1 &
#
# Everything is set in the block below: the muon source (MUONS: the flow, another flow
# checkpoint, or stored muons), the pools, the surrogate, the number of simulated muons. The run
# of 2026-10-26 (50 parameters, the flow's muons with their production points) took 12.5 h; with
# 65 parameters the references only add the new designs (step 2: --stage extend, ~5 h):
#   1  pools                                                         GPUs 0, 1   2.9 h
#   2  references                                                    GPU 1       4.3 h
#      first ensemble -> its tolerance boxes -> adaptive pool        GPU 0       1.3 h
#   3  final ensemble (4 seeds; a seed > 10% above the median validation loss is replaced)  1 h
#   4  the analysis (run_analysis.sh's steps, split over both GPUs)                       5.1 h
#   5  the validation on the references                                                   minutes
set -uo pipefail
cd "$(dirname "$0")"
source env.sh

# ---- settings
GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}
export MUONS=${MUONS:-flow}                 # muon source: flow | flow:<checkpoint> | file:<muons.npy>
N=1000000000                                # simulated muons of the analysis
ARCH="--hidden 128 --p 32"                  # the surrogate (python -m lcso.train --help)
# pools: name, momentum temperature, region half-width (u), designs, seed, muons per design (the
# seed also draws the muons: the 50-parameter pools had 10-13)
POOLS_A="T2_m1e6 2.0 1 24000 20 1000000
T2_d2_m1e6 2.0 2 4000 21 1000000
T1.5_d1 1.5 1 1024 22 10000000"
ADAPT="T1.5_adapt 1.5 1 0 23 10000000"      # simulated at the first ensemble's worst cases
EXTRA=r2_T2_m1e6+r2_T2_d2_m1e6+r2_T1.5_d1+r2_T1.5_adapt   # the 50-parameter pools, the rest nominal; empty: none
# (the first analysis' r1_* would also load, but with them a training run holds ~75 GB of host memory)

L=outputs/logs/run_all
mkdir -p $L outputs/logs/pools
step() { echo "$(date '+%F %H:%M') $*"; }
names() { echo "$1" | awk '{print $1}' | paste -sd+; }
TRAIN_POOLS=$(names "$POOLS_A")+$(names "$ADAPT")${EXTRA:++$EXTRA}

pool() {  # GPU worker n_workers name T delta n seed muons [extra]: restarted after a simulator crash
  local gpu=$1 w=$2 nw=$3 name=$4 T=$5 delta=$6 n=$7 seed=$8 mu=$9; shift 9
  local log=outputs/logs/pools/${name}_$w.log
  for try in $(seq 1 100); do
    grep -q "^done" $log 2>/dev/null && return 0
    CUDA_VISIBLE_DEVICES=$gpu python -W ignore -m lcso.simulate_pools --muons $MUONS --name $name --temperature $T \
      --delta $delta --n_designs $n --seed $seed --n_muons $mu --worker $w --n_workers $nw "$@" 2>&1 \
      | grep --line-buffered -E "^design|^done|^resuming|^skipping|Error|error" >> $log
  done
  grep -q "^done" $log
}
train() {  # name seed gpu
  [ -f outputs/models/$1.pt ] && return 0
  CUDA_VISIBLE_DEVICES=$3 python -W ignore -m lcso.train --pool $POOLS $ARCH --seed $2 --name $1 > $L/train_$1.log 2>&1
}
ensemble() {  # name gpu: seeds 0-3 two at a time, a bad seed replaced (seeds 4, 5, ...), stacked into outputs/models/<name>.pt
  local name=$1 gpu=$2 a b
  train ${name}_s0 0 $gpu & a=$!; train ${name}_s1 1 $gpu & b=$!; wait $a $b
  train ${name}_s2 2 $gpu & a=$!; train ${name}_s3 3 $gpu & b=$!; wait $a $b
  local seeds="0 1 2 3" next=4
  for round in 1 2; do
    local bad=$(python - $name $seeds <<'EOF'
import sys, numpy as np, torch
name, seeds = sys.argv[1], sys.argv[2:]
v = {s: torch.load(f'outputs/models/{name}_s{s}.pt', weights_only=False)['trained_on']['val_bce'] for s in seeds}
print(' '.join(s for s in seeds if v[s] > 1.1 * np.median([v[t] for t in seeds if t != s])))
EOF
)
    [ -z "$bad" ] && break
    for s in $bad; do
      step "  $name seed $s: validation loss > 10% above the others; replaced by seed $next"
      train ${name}_s$next $next $gpu
      seeds=$(echo $seeds | sed "s/\b$s\b/$next/"); next=$((next + 1))
    done
  done
  python -W ignore -m lcso.train --stack $(for s in $seeds; do echo outputs/models/${name}_s$s.pt; done) --name $name > $L/stack_$name.log 2>&1
  step "  $name: seeds $seeds"
}

# ---- 1. pools, split over both GPUs
step "1. pools"
pools() {  # GPU worker
  while read -r p; do pool $1 $2 2 $p || return 1; done <<< "$POOLS_A"
}
pools $GPU_A 0 & A=$!; pools $GPU_B 1 & B=$!
wait $A && wait $B && step "1. pools done" || { step "1. a pool failed"; exit 1; }

# ---- 2. references on GPU B | first ensemble and the adaptive pool on GPU A
step "2. references | first ensemble, adaptive pool"
(
  export CUDA_VISIBLE_DEVICES=$GPU_B
  R="python -W ignore -m validation.references --muons $MUONS"
  [ -f outputs/data/references/T1.5/meta.pt ] || $R --name T1.5 --stage sample --temperature 1.5 --n_muons 400000000 --oat > $L/ref_sample_T1.5.log 2>&1
  [ -f outputs/data/references/T1/meta.pt ] || $R --name T1 --stage sample --temperature 1 --n_muons 600000000 > $L/ref_sample_T1.log 2>&1
  for t in T1.5 T1; do        # references of fewer parameters: the same muons, new designs appended
    $R --name $t --stage extend --n_random 8 >> $L/ref_extend_$t.log 2>&1
  done
  for t in T1.5 T1; do
    for try in 1 2 3; do $R --name $t --stage simulate >> $L/ref_sim_$t.log 2>&1 && break; done
  done
  step "2. references done"
) &
REFS=$!
(
  POOLS=$(names "$POOLS_A")${EXTRA:++$EXTRA} ensemble lcso_first $GPU_A
  F=outputs/results/robustness_first
  M="--model outputs/models/lcso_first.pt --muons $MUONS"
  export CUDA_VISIBLE_DEVICES=$GPU_A
  [ -f $F/robustness.npz ] || python -W ignore -m analysis.robustness $M --no_sim --out_dir $F > $L/first_robustness.log 2>&1
  [ -f $F/tolerance_boxes.npz ] || python -W ignore -m analysis.tolerance_boxes --run $F $M > $L/first_tolerance.log 2>&1
  ls $F/tolerance_sim_0.npz >/dev/null 2>&1 \
    || python -W ignore -m analysis.tolerance_boxes --run $F $M --sim --n_muons 300000000 > $L/first_tolerance_sim.log 2>&1
  [ -f outputs/data/adaptive_designs.npy ] || python -W ignore -m lcso.adaptive_designs --run $F $M > $L/adaptive_designs.log 2>&1
  pool $GPU_A 0 1 $ADAPT --designs outputs/data/adaptive_designs.npy --n_nominal 0
) || { step "2. first ensemble / adaptive pool failed"; exit 1; }
step "2. adaptive pool done"

# ---- 3. the final ensemble (GPU A; GPU B may still be on the references)
step "3. final ensemble"
POOLS=$TRAIN_POOLS ensemble lcso $GPU_A || { step "3. failed"; exit 1; }

# ---- 4. the analysis
step "4. analysis"
wait $REFS
GPUS="$GPU_A $GPU_B" N=$N bash run_analysis.sh > $L/analysis.log 2>&1 || { step "4. failed"; exit 1; }
step "4. analysis done"

# ---- 5. the validation
step "5. validation"
CUDA_VISIBLE_DEVICES=$GPU_A python -W ignore -m validation.validate --muons $MUONS > $L/validate.log 2>&1
step "all done"
