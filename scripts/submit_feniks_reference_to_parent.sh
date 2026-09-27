#!/bin/bash
# REDESIGN NEW_ROOT [CONFIG], --explore-cdf FAILED_RUN NEW_ROOT, or --resume ROOT.
set -Eeuo pipefail
command -v sbatch >/dev/null
REPO=$(pwd -P)
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLBACKEND=Agg
if [[ ${1:-} == --resume ]]; then
  export FENIKS_RTP_ROOT=$(realpath "${2:?existing root}")
  source "$FENIKS_RTP_ROOT/INPUT.env"
else
  if [[ ${1:-} == --explore-cdf ]]; then
    SOURCE=$(realpath "${2:?completed failed qualification run}")
    export FENIKS_RTP_ROOT=$(realpath -m "${3:?new exploratory root}")
    INIT=(init-exploratory --source-run "$SOURCE")
  else
    SOURCE=$(realpath "${1:?completed reference redesign root}")
    export FENIKS_RTP_ROOT=$(realpath -m "${2:?new root}")
    CONFIG=$(realpath "${3:-configs/experiments/feniks_reference_to_parent.yaml}")
    INIT=(init --redesign "$SOURCE" --config "$CONFIG")
  fi
  test ! -e "$FENIKS_RTP_ROOT"
  test ! -e "$FENIKS_RTP_ROOT.code.tar"
  python -m scripts.feniks_reference_to_parent "${INIT[@]}" --root "$FENIKS_RTP_ROOT"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_RTP_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
  DIGEST=$(sha256sum "$FENIKS_RTP_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_RTP_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/reference-to-parent-$DIGEST"
  mkdir -p "$FENIKS_RTP_CODE"
  tar -xf "$FENIKS_RTP_ROOT.code.tar" -C "$FENIKS_RTP_CODE"
  [[ -e "$FENIKS_RTP_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_RTP_CODE/Data"
  printf '%s\n' "$DIGEST" > "$FENIKS_RTP_ROOT/CODE_SHA256"
  printf 'export FENIKS_RTP_CODE=%q\n' "$FENIKS_RTP_CODE" > "$FENIKS_RTP_ROOT/INPUT.env"
fi
export FENIKS_RTP_CODE
ALL_JOBS=""
if [[ -f "$FENIKS_RTP_ROOT/JOBS.env" ]]; then
  source "$FENIKS_RTP_ROOT/JOBS.env"
  LIVE=$(squeue -h -u "${USER:?}" -o '%i')
  ACTIVE=$(printf '%s\n' "$LIVE" | awk -v ids="$ALL_JOBS" 'BEGIN {n=split(ids,a,","); for(i=1;i<=n;i++) wanted[a[i]]=1} {base=$1; sub(/_.*/,"",base); if(base in wanted) print $1}')
  [[ -z "$ACTIVE" ]] || { echo "Jobs still active: $ACTIVE" >&2; exit 1; }
fi
cd "$FENIKS_RTP_CODE"
python -m scripts.feniks_reference_to_parent schedule --root "$FENIKS_RTP_ROOT" > "$FENIKS_RTP_ROOT/SCHEDULE.env"
source "$FENIKS_RTP_ROOT/SCHEDULE.env"
if [[ $NEED_REPORT == 0 && $NEED_QUALIFICATION == 0 && $NEED_POPULATION == 0 && -z $MISSING_BANKS ]]; then
  echo "Benchmark already complete: $FENIKS_RTP_ROOT"; exit 0
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
    --job-name="feniks_parent_$mode" --export="ALL,FENIKS_RTP_MODE=$mode,FENIKS_RTP_GPU=$gpu" \
    --output="$FENIKS_RTP_ROOT/logs/$mode-$ATTEMPT-%A_%a.out" \
    --error="$FENIKS_RTP_ROOT/logs/$mode-$ATTEMPT-%A_%a.err" "$@" \
    "$FENIKS_RTP_CODE/scripts/feniks_reference_to_parent.slurm")
  SUBMITTED_ID=${raw%%;*}
  [[ $SUBMITTED_ID =~ ^[0-9]+$ ]]
  ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$SUBMITTED_ID"
  CURRENT_JOBS="${CURRENT_JOBS:+$CURRENT_JOBS,}$SUBMITTED_ID"
  printf 'export ALL_JOBS=%q\nexport CURRENT_JOBS=%q\n' "$ALL_JOBS" "$CURRENT_JOBS" > "$FENIKS_RTP_ROOT/JOBS.env"
  printf 'export PARENT_RUN=%q\n' "$FENIKS_RTP_ROOT" > "$(dirname "$FENIKS_RTP_ROOT")/avi_reference_to_parent_latest.env"
  echo "$mode=$SUBMITTED_ID"
}
DEP=()
if [[ $NEED_QUALIFICATION == 1 ]]; then
  submit qualify "$QUALIFY_MINUTES" 0
  DEP=(--dependency="afterok:$SUBMITTED_ID" --kill-on-invalid-dep=yes)
fi
if [[ -n $MISSING_BANKS ]]; then
  submit bank "$BANK_MINUTES" 1 "${DEP[@]}" --array="$MISSING_BANKS%$BANK_CONCURRENCY"
  DEP=(--dependency="afterok:$SUBMITTED_ID" --kill-on-invalid-dep=yes)
fi
if [[ $NEED_POPULATION == 1 ]]; then
  submit population "$POPULATION_MINUTES" 1 "${DEP[@]}"
fi
REPORT_DEP=()
[[ -z $CURRENT_JOBS ]] || REPORT_DEP=(--dependency="afterany:${CURRENT_JOBS//,/:}")
submit report "$REPORT_MINUTES" 0 "${REPORT_DEP[@]}"
echo 'Reference banks -> blind parent -> report. Completed qualification is reused.'
echo 'GPU work requires capacity PASS or explicit bounded CDF-only exploration. No posterior training.'
echo 'Default full allocation ceiling: 11 H100-hours; not an expected runtime.'
printf 'watch=bash scripts/watch_feniks_reference_to_parent.sh %q\n' "$FENIKS_RTP_ROOT"
