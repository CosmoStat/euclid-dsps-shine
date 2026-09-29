#!/bin/bash
# CONDITIONAL POSTERIOR NEW_ROOT [CONFIG], --reaudit OLD_ROOT NEW_ROOT, or --resume ROOT.
set -Eeuo pipefail
REPO=$(pwd -P)
command -v sbatch >/dev/null
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLBACKEND=Agg
REUSE=()
if [[ ${1:-} == --resume ]]; then
  export FENIKS_RECOVERY_ROOT=$(realpath "${2:?existing root}")
  source "$FENIKS_RECOVERY_ROOT/INPUT.env"
elif [[ ${1:-} == --reaudit ]]; then
  PREVIOUS=$(realpath "${2:?previous recovery}")
  mapfile -t SOURCES < <(python - "$PREVIOUS/MANIFEST.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m['source_conditional'])
print(m['source_posterior'])
PY
  )
  [[ ${#SOURCES[@]} == 2 ]]
  CONDITIONAL=$(realpath "${SOURCES[0]}")
  POSTERIOR=$(realpath "${SOURCES[1]}")
  export FENIKS_RECOVERY_ROOT=$(realpath -m "${3:?new recovery root}")
  CONFIG=$(realpath "${4:-configs/experiments/feniks_overnight_recovery.yaml}")
  REUSE=(--reuse-parent "$PREVIOUS")
  test ! -e "$FENIKS_RECOVERY_ROOT"
  test ! -e "$FENIKS_RECOVERY_ROOT.code.tar"
else
  CONDITIONAL=$(realpath "${1:?conditional-parent root}")
  POSTERIOR=$(realpath "${2:?posterior root}")
  export FENIKS_RECOVERY_ROOT=$(realpath -m "${3:?new recovery root}")
  CONFIG=$(realpath "${4:-configs/experiments/feniks_overnight_recovery.yaml}")
  test ! -e "$FENIKS_RECOVERY_ROOT"
  test ! -e "$FENIKS_RECOVERY_ROOT.code.tar"
fi
# Refuse active source jobs as well as duplicate recovery submissions. A failed
# squeue is NOT evidence of inactivity. Never cancel or modify the old runs.
LIVE=$(squeue -h -u "${USER:?}" -o '%i')
check_active() {
  local folder=$1 ids found
  [[ -f "$folder/JOBS.env" ]] || return 0
  ids=$(bash -c 'source "$1"; printf "%s" "${ALL_JOBS:?}"' _ "$folder/JOBS.env")
  found=$(printf '%s\n' "$LIVE" | awk -v ids="$ids" 'BEGIN {n=split(ids,a,","); for(i=1;i<=n;i++) wanted[a[i]]=1} {base=$1; sub(/_.*/,"",base); if(base in wanted) print $1}')
  [[ -z "$found" ]] || { echo "Active jobs retained: $folder $found" >&2; exit 1; }
}
if [[ ${1:-} != --resume ]]; then
  [[ ${1:-} != --reaudit ]] || check_active "$PREVIOUS"
  check_active "$CONDITIONAL"
  check_active "$POSTERIOR"
  BASE=$(dirname "$FENIKS_RECOVERY_ROOT")
  exec 8>"$BASE/.feniks_overnight_recovery.lock"
  flock -n 8 || { echo 'Another recovery launch is active' >&2; exit 1; }
  if [[ -f "$BASE/avi_overnight_recovery_latest.env" ]]; then
    PREVIOUS=$(bash -c 'source "$1"; printf "%s" "${RECOVERY:?}"' _ "$BASE/avi_overnight_recovery_latest.env")
    check_active "$PREVIOUS"
  fi
  python -m scripts.feniks_overnight_recovery init --root "$FENIKS_RECOVERY_ROOT" \
    --conditional "$CONDITIONAL" --posterior "$POSTERIOR" --config "$CONFIG" "${REUSE[@]}"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_RECOVERY_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
  DIGEST=$(sha256sum "$FENIKS_RECOVERY_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_RECOVERY_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/overnight-recovery-$DIGEST"
  mkdir -p "$FENIKS_RECOVERY_CODE"
  tar -xf "$FENIKS_RECOVERY_ROOT.code.tar" -C "$FENIKS_RECOVERY_CODE"
  [[ -e "$FENIKS_RECOVERY_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_RECOVERY_CODE/Data"
  printf '%s\n' "$DIGEST" > "$FENIKS_RECOVERY_ROOT/CODE_SHA256"
  printf 'export FENIKS_RECOVERY_CODE=%q\n' "$FENIKS_RECOVERY_CODE" > "$FENIKS_RECOVERY_ROOT/INPUT.env"
fi
export FENIKS_RECOVERY_CODE
exec 9>"$FENIKS_RECOVERY_ROOT/.submit.lock"
flock -n 9 || { echo 'Another submission is active' >&2; exit 1; }
LIVE=$(squeue -h -u "${USER:?}" -o '%i')
check_active "$FENIKS_RECOVERY_ROOT"
ALL_JOBS=""
[[ ! -f "$FENIKS_RECOVERY_ROOT/JOBS.env" ]] || source "$FENIKS_RECOVERY_ROOT/JOBS.env"
cd "$FENIKS_RECOVERY_CODE"
python -m scripts.feniks_overnight_recovery schedule --root "$FENIKS_RECOVERY_ROOT" > "$FENIKS_RECOVERY_ROOT/SCHEDULE.env"
source "$FENIKS_RECOVERY_ROOT/SCHEDULE.env"
if [[ $NEED_PARENT == 0 && $NEED_AUDIT == 0 && $NEED_POSTERIOR == 0 && $NEED_EVALUATION == 0 && $NEED_REPORT == 0 ]]; then
  echo 'Recovery already complete'; exit 0
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
  raw=$(sbatch --parsable "${resource[@]}" --nodes=1 --ntasks=1 --cpus-per-task="$CPU_THREADS" \
    --hint=nomultithread --time="$minutes" --job-name="feniks_recover_$mode" \
    --export="ALL,FENIKS_RECOVERY_MODE=$mode,FENIKS_RECOVERY_GPU=$gpu" \
    --output="$FENIKS_RECOVERY_ROOT/logs/$mode-$ATTEMPT-%j.out" \
    --error="$FENIKS_RECOVERY_ROOT/logs/$mode-$ATTEMPT-%j.err" "$@" \
    "$FENIKS_RECOVERY_CODE/scripts/feniks_overnight_recovery.slurm")
  ID=${raw%%;*}
  [[ $ID =~ ^[0-9]+$ ]]
  ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$ID"
  CURRENT_JOBS="${CURRENT_JOBS:+$CURRENT_JOBS,}$ID"
  printf 'export ALL_JOBS=%q\nexport CURRENT_JOBS=%q\n' "$ALL_JOBS" "$CURRENT_JOBS" > "$FENIKS_RECOVERY_ROOT/JOBS.env"
  printf 'export RECOVERY=%q\n' "$FENIKS_RECOVERY_ROOT" > "$(dirname "$FENIKS_RECOVERY_ROOT")/avi_overnight_recovery_latest.env"
  printf '%s=%s\n' "$mode" "$ID"
}
[[ $NEED_PARENT == 0 ]] || submit parent "$PARENT_MINUTES" 0
DEP=()
if [[ $NEED_AUDIT == 1 ]]; then
  submit audit "$AUDIT_MINUTES" "$AUDIT_GPU"
  DEP=(--dependency="afterok:$ID" --kill-on-invalid-dep=yes)
fi
if [[ $NEED_POSTERIOR == 1 || $NEED_EVALUATION == 1 ]]; then
  submit train "$TRAIN_MINUTES" 1 "${DEP[@]}"
fi
DEP=()
[[ -z "$CURRENT_JOBS" ]] || DEP=(--dependency="afterany:${CURRENT_JOBS//,/:}")
submit report "$REPORT_MINUTES" 0 "${DEP[@]}"
if [[ $NEED_PARENT == 0 ]]; then
  echo 'Completed parent reused; no parent or classifier job.'
else
  echo 'CPU parent recovery and backend-matched tail audit run in parallel.'
fi
echo 'Backend-matched numerical audit gates one H100 continuation.'
echo 'Zero new DSPS/classifier training. Default ceiling: 4.5 H100-hours including audit; not expected runtime.'
printf 'watch=bash scripts/watch_feniks_overnight_recovery.sh %q\n' "$FENIKS_RECOVERY_ROOT"
