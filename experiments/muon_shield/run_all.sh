#!/bin/bash
# Every muon shield study, in order. Needs the GPU and faster_muons_torch (MuonShieldChallenge/README.md).
#
#     bash experiments/muon_shield/run_all.sh
#     STAGES="inner compare" bash experiments/muon_shield/run_all.sh     # only some stages
#     MUON_SHIELD_CONFIG=hard CUDA_VISIBLE_DEVICES=1 bash experiments/muon_shield/run_all.sh
#
# Logs: muon_shield/logs/<step>.log, and one line per step (exit code, time) in logs/STATUS.
# Estimates use 1e8 muons; only training (surrogate fits, adversary in the outer loop) undersamples.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP=$(dirname "$HERE")
LOGS=$HERE/logs
mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1

N=${N:-100000000}                          # nominal muons of every estimate
N_ADV=${N_ADV:-10000000}                   # outer loop: adversary sample while training
PER_DESIGN=${PER_DESIGN:-1000000}          # outer loop: muons per design in each fit
FIT_DESIGNS=${FIT_DESIGNS:-60}             # local surrogate fits: designs
FIT_PER_DESIGN=${FIT_PER_DESIGN:-1000000}  # local surrogate fits: muons per design
LCSO_BUDGET=${LCSO_BUDGET:-24000000000}    # ~20 rounds of 120 designs x 1e7
ROUNDS=${ROUNDS:-25}
QMC_SIZES=${QMC_SIZES:-"1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864"}
STAGES=${STAGES:-"generator check inner lcso compare outer integration"}   # add "pool" for NOMINAL='pool'

step() {  # name dir cmd...
  local name=$1 dir=$2; shift 2
  local t0=$(date +%s)
  echo ">>> $name"
  (cd "$dir" && "$@") > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s" | tee -a "$LOGS/STATUS"
  return $rc
}
has() { [[ " $STAGES " == *" $1 "* ]]; }

has pool && step build_pool "$HERE" python build_pool.py
has generator && step train_generator "$HERE" python train_generator.py
has check && step check "$HERE" python problem.py --n_samples "$N"
if has inner; then   # run_inner_cost.py writes models/design.pt and models/surrogate.pt for all later steps
  step run_inner_cost "$HERE" python run_inner_cost.py --n "$N" \
       --n_designs "$FIT_DESIGNS" --per_design "$FIT_PER_DESIGN" || exit 1
  step run_inner_samples "$HERE" python run_inner_samples.py --n "$N"
  step run_inner "$HERE" python run_inner.py --n "$N"
fi
has lcso && step train_lcso "$HERE" python train_lcso.py --budget "$LCSO_BUDGET" \
  --n_samples 100000000 --resample_per_design 10000000 --n_eval "$N"
has compare && step compare "$EXP/inner_compare" python run_compare.py --problem muon_shield --n "$N" \
  && step compare.combine "$EXP/inner_compare" python combine.py
has outer && step outer "$EXP/outer" python run_outer.py --problem muon_shield --rounds "$ROUNDS" \
  --n_test "$N" --n_adv "$N_ADV" --per_design "$PER_DESIGN" \
  --local_fit "$FIT_DESIGNS" "$FIT_PER_DESIGN"
if has integration; then
  # shellcheck disable=SC2086
  step quasi-MC "$EXP/integration" python quasi-MC.py --problem muon_shield --sizes $QMC_SIZES
  # shellcheck disable=SC2086
  step quantify "$EXP/integration" python quantify.py --problem muon_shield --sizes $QMC_SIZES
  # shellcheck disable=SC2086
  step quantify.surrogate "$EXP/integration" python quantify.py --problem muon_shield \
       --loss surrogate --sizes $QMC_SIZES
fi
echo "done; see $LOGS/STATUS"
