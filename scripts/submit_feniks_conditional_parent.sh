#!/bin/bash
# SOURCE_PARENT NEW_ROOT [CONFIG], or --resume ROOT.
set -Eeuo pipefail
command -v sbatch >/dev/null
REPO=$(pwd -P)
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLBACKEND=Agg
if [[ ${1:-} == --resume ]]; then
  export FENIKS_CPAR_ROOT=$(realpath "${2:?existing root}")
  source "$FENIKS_CPAR_ROOT/INPUT.env"
else
  SOURCE=$(realpath "${1:?completed blind parent}")
  export FENIKS_CPAR_ROOT=$(realpath -m "${2:?new root}")
  CONFIG=$(realpath "${3:-configs/experiments/feniks_conditional_parent.yaml}")
  test ! -e "$FENIKS_CPAR_ROOT"
  test ! -e "$FENIKS_CPAR_ROOT.code.tar"
  python -m scripts.feniks_conditional_parent init --source "$SOURCE" --root "$FENIKS_CPAR_ROOT" --config "$CONFIG"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_CPAR_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
  DIGEST=$(sha256sum "$FENIKS_CPAR_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_CPAR_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/conditional-parent-$DIGEST"
  mkdir -p "$FENIKS_CPAR_CODE"
  tar -xf "$FENIKS_CPAR_ROOT.code.tar" -C "$FENIKS_CPAR_CODE"
  [[ -e "$FENIKS_CPAR_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_CPAR_CODE/Data"
  printf '%s\n' "$DIGEST" > "$FENIKS_CPAR_ROOT/CODE_SHA256"
  printf 'export FENIKS_CPAR_CODE=%q\n' "$FENIKS_CPAR_CODE" > "$FENIKS_CPAR_ROOT/INPUT.env"
fi
export FENIKS_CPAR_CODE
exec 9>"$FENIKS_CPAR_ROOT/.submit.lock"
flock -n 9 || { echo 'Another submission is active' >&2; exit 1; }
ALL_JOBS=""
if [[ -f "$FENIKS_CPAR_ROOT/JOBS.env" ]]; then
  source "$FENIKS_CPAR_ROOT/JOBS.env"
  LIVE=$(squeue -h -u "${USER:?}" -o '%i')
  ACTIVE=$(printf '%s\n' "$LIVE" | awk -v ids="$ALL_JOBS" 'BEGIN {n=split(ids,a,","); for(i=1;i<=n;i++) wanted[a[i]]=1} {base=$1; sub(/_.*/,"",base); if(base in wanted) print $1}')
  [[ -z "$ACTIVE" ]] || { echo "Jobs still active: $ACTIVE" >&2; exit 1; }
fi
cd "$FENIKS_CPAR_CODE"
python -m scripts.feniks_conditional_parent schedule --root "$FENIKS_CPAR_ROOT" > "$FENIKS_CPAR_ROOT/SCHEDULE.env"
source "$FENIKS_CPAR_ROOT/SCHEDULE.env"
if [[ $NEED_PREPARE == 0 && -z $MISSING_BANKS && $NEED_POPULATION == 0 && $NEED_REPORT == 0 ]]; then
  echo "Run already complete: $FENIKS_CPAR_ROOT"; exit 0
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
    --job-name="feniks_cpar_$mode" --export="ALL,FENIKS_CPAR_MODE=$mode,FENIKS_CPAR_GPU=$gpu" \
    --output="$FENIKS_CPAR_ROOT/logs/$mode-$ATTEMPT-%A_%a.out" \
    --error="$FENIKS_CPAR_ROOT/logs/$mode-$ATTEMPT-%A_%a.err" "$@" \
    "$FENIKS_CPAR_CODE/scripts/feniks_conditional_parent.slurm")
  SUBMITTED_ID=${raw%%;*}
  [[ $SUBMITTED_ID =~ ^[0-9]+$ ]]
  ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$SUBMITTED_ID"
  CURRENT_JOBS="${CURRENT_JOBS:+$CURRENT_JOBS,}$SUBMITTED_ID"
  printf 'export ALL_JOBS=%q\nexport CURRENT_JOBS=%q\n' "$ALL_JOBS" "$CURRENT_JOBS" > "$FENIKS_CPAR_ROOT/JOBS.env"
  printf 'export CONDITIONAL=%q\n' "$FENIKS_CPAR_ROOT" > "$(dirname "$FENIKS_CPAR_ROOT")/avi_conditional_parent_latest.env"
  printf '%s=%s\n' "$mode" "$SUBMITTED_ID"
}
DEP=()
if [[ $NEED_PREPARE == 1 ]]; then
  submit prepare "$PREPARE_MINUTES" 0
  DEP=(--dependency="afterok:$SUBMITTED_ID" --kill-on-invalid-dep=yes)
fi
if [[ -n $MISSING_BANKS ]]; then
  submit relabel "$RELABEL_MINUTES" 0 "${DEP[@]}" --array="$MISSING_BANKS%$RELABEL_CONCURRENCY"
  DEP=(--dependency="afterok:$SUBMITTED_ID" --kill-on-invalid-dep=yes)
fi
if [[ $NEED_POPULATION == 1 ]]; then
  submit population "$POPULATION_MINUTES" 1 "${DEP[@]}"
fi
DEP=()
[[ -z "$CURRENT_JOBS" ]] || DEP=(--dependency="afterany:${CURRENT_JOBS//,/:}")
submit report "$REPORT_MINUTES" 0 "${DEP[@]}"
echo 'CPU normalized split -> CPU exact bank replay -> one GPU classifier + blind parent/tied control -> CPU report.'
echo 'Default: zero new DSPS simulations; peak 1 H100; 3 H100-hour allocation ceiling, not runtime estimate.'
printf 'watch=bash scripts/watch_feniks_conditional_parent.sh %q\n' "$FENIKS_CPAR_ROOT"
