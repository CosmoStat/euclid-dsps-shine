#!/bin/bash
# SOURCE_INFERENCE NEW_ROOT [CONFIG], or --resume EXISTING_ROOT.
set -Eeuo pipefail
command -v sbatch >/dev/null
REPO=$(pwd -P)
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLBACKEND=Agg
if [[ ${1:-} == --resume ]]; then
  export FENIKS_RR_ROOT=$(realpath "${2:?existing redesign root}")
  source "$FENIKS_RR_ROOT/INPUT.env"
else
  SOURCE=$(realpath "${1:?completed coherent inference root}")
  export FENIKS_RR_ROOT=$(realpath -m "${2:?new redesign root}")
  CONFIG=$(realpath "${3:-configs/experiments/feniks_reference_redesign.yaml}")
  test ! -e "$FENIKS_RR_ROOT"
  test ! -e "$FENIKS_RR_ROOT.code.tar"
  python -m scripts.feniks_reference_redesign init --source "$SOURCE" --root "$FENIKS_RR_ROOT" --config "$CONFIG"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_RR_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
  DIGEST=$(sha256sum "$FENIKS_RR_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_RR_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/reference-redesign-$DIGEST"
  mkdir -p "$FENIKS_RR_CODE"
  tar -xf "$FENIKS_RR_ROOT.code.tar" -C "$FENIKS_RR_CODE"
  [[ -e "$FENIKS_RR_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_RR_CODE/Data"
  printf '%s\n' "$DIGEST" > "$FENIKS_RR_ROOT/CODE_SHA256"
  printf 'export FENIKS_RR_CODE=%q\n' "$FENIKS_RR_CODE" > "$FENIKS_RR_ROOT/INPUT.env"
fi
export FENIKS_RR_CODE
ALL_JOBS=""
if [[ -f "$FENIKS_RR_ROOT/JOBS.env" ]]; then
  source "$FENIKS_RR_ROOT/JOBS.env"
  # Query live user jobs, not expired IDs; fail closed if SLURM cannot answer.
  LIVE=$(squeue -h -u "${USER:?}" -o '%i')
  ACTIVE=$(printf '%s\n' "$LIVE" | awk -v ids="$ALL_JOBS" 'BEGIN {n=split(ids,a,","); for(i=1;i<=n;i++) wanted[a[i]]=1} {base=$1; sub(/_.*/,"",base); if(base in wanted) print $1}')
  [[ -z "$ACTIVE" ]] || { echo "Jobs still active: $ACTIVE" >&2; exit 1; }
fi
cd "$FENIKS_RR_CODE"
python -m scripts.feniks_reference_redesign schedule --root "$FENIKS_RR_ROOT" > "$FENIKS_RR_ROOT/SCHEDULE.env"
source "$FENIKS_RR_ROOT/SCHEDULE.env"
if [[ $NEED_REPORT == 0 && $NEED_PREPARE == 0 && $NEED_REPLAY == 0 && -z $MISSING_CELLS ]]; then
  echo "Qualification already complete: $FENIKS_RR_ROOT"; exit 0
fi
CURRENT_JOBS=""
ATTEMPT=$(date +%Y%m%d_%H%M%S)
COMMON=(--parsable --nodes=1 --ntasks=1 --cpus-per-task="$CPU_THREADS" --hint=nomultithread)
CPU=(--account="${FENIKS_CPU_ACCOUNT:-jrx@cpu}" --partition="${FENIKS_CPU_PARTITION:-cpu_p1}")
GPU=(--account="${FENIKS_GPU_ACCOUNT:-jrx@h100}" --partition="${FENIKS_GPU_PARTITION:-gpu_p6}" --constraint=h100 --gres=gpu:1)
submit() {
  local mode=$1 minutes=$2 gpu=$3
  shift 3
  local -a resources=("${CPU[@]}")
  [[ $gpu == 0 ]] || resources=("${GPU[@]}")
  local raw
  raw=$(sbatch "${COMMON[@]}" "${resources[@]}" --time="$minutes" \
    --job-name="feniks_ref_$mode" --export="ALL,FENIKS_RR_MODE=$mode,FENIKS_RR_GPU=$gpu" \
    --output="$FENIKS_RR_ROOT/logs/$mode-$ATTEMPT-%A_%a.out" \
    --error="$FENIKS_RR_ROOT/logs/$mode-$ATTEMPT-%A_%a.err" "$@" \
    "$FENIKS_RR_CODE/scripts/feniks_reference_redesign.slurm")
  SUBMITTED_ID=${raw%%;*}
  [[ $SUBMITTED_ID =~ ^[0-9]+$ ]]
  ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$SUBMITTED_ID"
  CURRENT_JOBS="${CURRENT_JOBS:+$CURRENT_JOBS,}$SUBMITTED_ID"
  printf 'export ALL_JOBS=%q\nexport CURRENT_JOBS=%q\n' "$ALL_JOBS" "$CURRENT_JOBS" > "$FENIKS_RR_ROOT/JOBS.env"
  printf 'export REDESIGN=%q\n' "$FENIKS_RR_ROOT" > "$(dirname "$FENIKS_RR_ROOT")/avi_reference_redesign_latest.env"
  echo "$mode=$SUBMITTED_ID"
}
DEP=()
if [[ $NEED_PREPARE == 1 ]]; then
  submit prepare "$PREPARE_MINUTES" 0
  DEP=(--dependency="afterok:$SUBMITTED_ID" --kill-on-invalid-dep=yes)
fi
if [[ -n $MISSING_CELLS ]]; then
  submit capacity "$CAPACITY_MINUTES" 0 "${DEP[@]}" --array="$MISSING_CELLS%$CAPACITY_CONCURRENCY"
fi
if [[ $NEED_REPLAY == 1 ]]; then
  submit replay "$REPLAY_MINUTES" 1 "${DEP[@]}"
fi
REPORT_DEP=()
[[ -z $CURRENT_JOBS ]] || REPORT_DEP=(--dependency="afterany:${CURRENT_JOBS//,/:}")
submit report "$REPORT_MINUTES" 0 "${REPORT_DEP[@]}"
echo "4 CPU capacity cells and one paired GPU replay in parallel after preparation."
echo "No training. H100 allocation ceiling: $REPLAY_MINUTES minutes; queue time is additional."
printf 'watch=bash scripts/watch_feniks_reference_redesign.sh %q\n' "$FENIKS_RR_ROOT"
