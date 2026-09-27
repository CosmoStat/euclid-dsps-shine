#!/bin/bash
# --exploratory SOURCE_PARENT NEW_ROOT [CONFIG], or --resume EXISTING_ROOT.
set -Eeuo pipefail
command -v sbatch >/dev/null
REPO=$(pwd -P)
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLBACKEND=Agg
if [[ ${1:-} == --resume ]]; then
  export FENIKS_PTP_ROOT=$(realpath "${2:?existing posterior run}")
  source "$FENIKS_PTP_ROOT/INPUT.env"
else
  [[ ${1:-} == --exploratory ]] || { echo 'Require explicit --exploratory SOURCE NEW_ROOT [CONFIG]' >&2; exit 2; }
  SOURCE=$(realpath "${2:?completed blind parent run}")
  export FENIKS_PTP_ROOT=$(realpath -m "${3:?new posterior root}")
  CONFIG=$(realpath "${4:-configs/experiments/feniks_parent_to_posterior.yaml}")
  test ! -e "$FENIKS_PTP_ROOT"
  test ! -e "$FENIKS_PTP_ROOT.code.tar"
  python -m scripts.feniks_parent_to_posterior init --exploratory \
    --source "$SOURCE" --root "$FENIKS_PTP_ROOT" --config "$CONFIG"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_PTP_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
  DIGEST=$(sha256sum "$FENIKS_PTP_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_PTP_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/parent-to-posterior-$DIGEST"
  mkdir -p "$FENIKS_PTP_CODE"
  tar -xf "$FENIKS_PTP_ROOT.code.tar" -C "$FENIKS_PTP_CODE"
  [[ -e "$FENIKS_PTP_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_PTP_CODE/Data"
  printf '%s\n' "$DIGEST" > "$FENIKS_PTP_ROOT/CODE_SHA256"
  printf 'export FENIKS_PTP_CODE=%q\n' "$FENIKS_PTP_CODE" > "$FENIKS_PTP_ROOT/INPUT.env"
fi
export FENIKS_PTP_CODE
exec 9>"$FENIKS_PTP_ROOT/.submit.lock"
flock -n 9 || { echo 'Another submission is active' >&2; exit 1; }
ALL_JOBS=""
if [[ -f "$FENIKS_PTP_ROOT/JOBS.env" ]]; then
  source "$FENIKS_PTP_ROOT/JOBS.env"
  LIVE=$(squeue -h -u "${USER:?}" -o '%i')
  ACTIVE=$(printf '%s\n' "$LIVE" | awk -v ids="$ALL_JOBS" 'BEGIN {n=split(ids,a,","); for(i=1;i<=n;i++) wanted[a[i]]=1} {base=$1; sub(/_.*/,"",base); if(base in wanted) print $1}')
  [[ -z "$ACTIVE" ]] || { echo "Jobs still active: $ACTIVE" >&2; exit 1; }
fi
cd "$FENIKS_PTP_CODE"
python -m scripts.feniks_parent_to_posterior schedule --root "$FENIKS_PTP_ROOT" > "$FENIKS_PTP_ROOT/SCHEDULE.env"
source "$FENIKS_PTP_ROOT/SCHEDULE.env"
if [[ -z $MISSING_BANKS && $NEED_POSTERIOR == 0 && $NEED_EVALUATION == 0 && $NEED_REPORT == 0 ]]; then
  echo "Run already complete: $FENIKS_PTP_ROOT"; exit 0
fi
ATTEMPT=$(date +%Y%m%d_%H%M%S)
CURRENT_JOBS=""
submit() {
  local mode=$1 minutes=$2 gpu=$3 raw
  shift 3
  local resource=(--account="${FENIKS_CPU_ACCOUNT:-jrx@cpu}" --partition="${FENIKS_CPU_PARTITION:-cpu_p1}")
  if [[ $gpu == 1 ]]; then
    resource=(--account="${FENIKS_GPU_ACCOUNT:-jrx@h100}" --partition="${FENIKS_GPU_PARTITION:-gpu_p6}" --constraint=h100 --gres=gpu:1)
  fi
  raw=$(sbatch --parsable "${resource[@]}" --nodes=1 --ntasks=1 \
    --cpus-per-task="$CPU_THREADS" --hint=nomultithread --time="$minutes" \
    --job-name="feniks_ptp_$mode" --export="ALL,FENIKS_PTP_MODE=$mode,FENIKS_PTP_GPU=$gpu" \
    --output="$FENIKS_PTP_ROOT/logs/$mode-$ATTEMPT-%A_%a.out" \
    --error="$FENIKS_PTP_ROOT/logs/$mode-$ATTEMPT-%A_%a.err" "$@" \
    "$FENIKS_PTP_CODE/scripts/feniks_parent_to_posterior.slurm")
  SUBMITTED_ID=${raw%%;*}
  [[ $SUBMITTED_ID =~ ^[0-9]+$ ]]
  ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$SUBMITTED_ID"
  CURRENT_JOBS="${CURRENT_JOBS:+$CURRENT_JOBS,}$SUBMITTED_ID"
  printf 'export ALL_JOBS=%q\nexport CURRENT_JOBS=%q\n' "$ALL_JOBS" "$CURRENT_JOBS" > "$FENIKS_PTP_ROOT/JOBS.env"
  printf 'export POSTERIOR_RUN=%q\n' "$FENIKS_PTP_ROOT" > "$(dirname "$FENIKS_PTP_ROOT")/avi_parent_to_posterior_latest.env"
  printf '%s=%s\n' "$mode" "$SUBMITTED_ID"
}
DEP=()
if [[ -n $MISSING_BANKS ]]; then
  submit bank "$BANK_MINUTES" 1 --array="$MISSING_BANKS%$BANK_CONCURRENCY"
  DEP=(--dependency="afterok:$SUBMITTED_ID" --kill-on-invalid-dep=yes)
fi
if [[ $NEED_POSTERIOR == 1 ]]; then
  submit train "$TRAIN_MINUTES" 1 "${DEP[@]}"
  DEP=(--dependency="afterok:$SUBMITTED_ID" --kill-on-invalid-dep=yes)
fi
if [[ $NEED_EVALUATION == 1 ]]; then
  submit evaluate "$EVALUATE_MINUTES" 1 "${DEP[@]}"
fi
DEP=()
[[ -z "$CURRENT_JOBS" ]] || DEP=(--dependency="afterany:${CURRENT_JOBS//,/:}")
submit report "$REPORT_MINUTES" 0 "${DEP[@]}"
echo 'Fresh direct-parent bank -> supervised 15D posterior -> two-cohort evaluation -> report.'
echo 'Existing parent-recovery diagnostic is independent and untouched.'
echo 'Default: 1048576 simulations; peak 4 H100s; 11 H100-hour allocation ceiling, not runtime estimate.'
printf 'watch=bash scripts/watch_feniks_parent_to_posterior.sh %q\n' "$FENIKS_PTP_ROOT"
