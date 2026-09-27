#!/bin/bash
# SOURCE_PARENT_RUN NEW_ROOT [CONFIG], or --resume EXISTING_ROOT.
set -Eeuo pipefail
command -v sbatch >/dev/null
REPO=$(pwd -P)
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MPLBACKEND=Agg
if [[ ${1:-} == --resume ]]; then
  export FENIKS_RECOVERY_ROOT=$(realpath "${2:?existing diagnostic root}")
  source "$FENIKS_RECOVERY_ROOT/INPUT.env"
else
  SOURCE=$(realpath "${1:?completed parent run}")
  export FENIKS_RECOVERY_ROOT=$(realpath -m "${2:?new diagnostic root}")
  CONFIG=$(realpath "${3:-configs/experiments/feniks_parent_recovery.yaml}")
  test ! -e "$FENIKS_RECOVERY_ROOT"
  test ! -e "$FENIKS_RECOVERY_ROOT.code.tar"
  python -m scripts.feniks_parent_recovery init --source "$SOURCE" \
    --root "$FENIKS_RECOVERY_ROOT" --config "$CONFIG"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_RECOVERY_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
  DIGEST=$(sha256sum "$FENIKS_RECOVERY_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_RECOVERY_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/parent-recovery-$DIGEST"
  mkdir -p "$FENIKS_RECOVERY_CODE"
  tar -xf "$FENIKS_RECOVERY_ROOT.code.tar" -C "$FENIKS_RECOVERY_CODE"
  [[ -e "$FENIKS_RECOVERY_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_RECOVERY_CODE/Data"
  printf '%s\n' "$DIGEST" > "$FENIKS_RECOVERY_ROOT/CODE_SHA256"
  printf 'export FENIKS_RECOVERY_CODE=%q\n' "$FENIKS_RECOVERY_CODE" > "$FENIKS_RECOVERY_ROOT/INPUT.env"
fi
export FENIKS_RECOVERY_CODE
exec 9>"$FENIKS_RECOVERY_ROOT/.submit.lock"
flock -n 9 || { echo 'Another submission is active' >&2; exit 1; }
ALL_JOBS=""
if [[ -f "$FENIKS_RECOVERY_ROOT/JOBS.env" ]]; then
  source "$FENIKS_RECOVERY_ROOT/JOBS.env"
  LIVE=$(squeue -h -u "${USER:?}" -o '%i')
  ACTIVE=$(printf '%s\n' "$LIVE" | awk -v ids="$ALL_JOBS" 'BEGIN {n=split(ids,a,","); for(i=1;i<=n;i++) wanted[a[i]]=1} {base=$1; sub(/_.*/,"",base); if(base in wanted) print $1}')
  [[ -z "$ACTIVE" ]] || { echo "Jobs still active: $ACTIVE" >&2; exit 1; }
fi
cd "$FENIKS_RECOVERY_CODE"
python -m scripts.feniks_parent_recovery schedule --root "$FENIKS_RECOVERY_ROOT" > "$FENIKS_RECOVERY_ROOT/SCHEDULE.env"
source "$FENIKS_RECOVERY_ROOT/SCHEDULE.env"
if [[ $NEED_CACHE == 0 && -z $MISSING_CASES && $NEED_REPORT == 0 ]]; then
  echo "Diagnostic already complete: $FENIKS_RECOVERY_ROOT"; exit 0
fi
ATTEMPT=$(date +%Y%m%d_%H%M%S)
CURRENT_JOBS=""
submit() {
  local mode=$1 minutes=$2
  shift 2
  local raw
  raw=$(sbatch --parsable --account="${FENIKS_CPU_ACCOUNT:-jrx@cpu}" \
    --partition="${FENIKS_CPU_PARTITION:-cpu_p1}" --nodes=1 --ntasks=1 \
    --cpus-per-task="$CPU_THREADS" --hint=nomultithread --time="$minutes" \
    --job-name="feniks_recovery_$mode" --export="ALL,FENIKS_RECOVERY_MODE=$mode" \
    --output="$FENIKS_RECOVERY_ROOT/logs/$mode-$ATTEMPT-%A_%a.out" \
    --error="$FENIKS_RECOVERY_ROOT/logs/$mode-$ATTEMPT-%A_%a.err" "$@" \
    "$FENIKS_RECOVERY_CODE/scripts/feniks_parent_recovery.slurm")
  SUBMITTED_ID=${raw%%;*}
  [[ $SUBMITTED_ID =~ ^[0-9]+$ ]]
  ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$SUBMITTED_ID"
  CURRENT_JOBS="${CURRENT_JOBS:+$CURRENT_JOBS,}$SUBMITTED_ID"
  printf 'export ALL_JOBS=%q\nexport CURRENT_JOBS=%q\n' "$ALL_JOBS" "$CURRENT_JOBS" > "$FENIKS_RECOVERY_ROOT/JOBS.env"
  printf 'export RECOVERY=%q\n' "$FENIKS_RECOVERY_ROOT" > "$(dirname "$FENIKS_RECOVERY_ROOT")/avi_parent_recovery_latest.env"
  printf '%s=%s\n' "$mode" "$SUBMITTED_ID"
}
DEP=()
if [[ $NEED_CACHE == 1 ]]; then
  submit cache "$CACHE_MINUTES"
  DEP=(--dependency="afterok:$SUBMITTED_ID" --kill-on-invalid-dep=yes)
fi
if [[ -n $MISSING_CASES ]]; then
  submit case "$CASE_MINUTES" "${DEP[@]}" --array="$MISSING_CASES%$CASE_CONCURRENCY"
fi
DEP=()
[[ -z "$CURRENT_JOBS" ]] || DEP=(--dependency="afterany:${CURRENT_JOBS//,/:}")
submit report "$REPORT_MINUTES" "${DEP[@]}"
echo 'Frozen CPU classifier evaluation -> two-case CPU array -> report. No DSPS, training or GPU.'
echo 'Default limits: cache 25 min; each case 25 min; report 5 min. Not runtime estimates.'
printf 'watch=bash scripts/watch_feniks_parent_recovery.sh %q\n' "$FENIKS_RECOVERY_ROOT"
