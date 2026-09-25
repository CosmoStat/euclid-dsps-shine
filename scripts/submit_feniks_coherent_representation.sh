#!/bin/bash
# COHERENT_ROOT OLD_SPEC NEW_ROOT [CONFIG], or --resume EXISTING_ROOT
set -Eeuo pipefail
REPO=$(pwd -P)
command -v sbatch >/dev/null
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
if [[ ${1:-} == --resume ]]; then
  export FENIKS_REP_ROOT=$(realpath "${2:?existing root}")
  if [[ -f "$FENIKS_REP_ROOT/JOBS.env" ]]; then
    source "$FENIKS_REP_ROOT/JOBS.env"
    ACTIVE=$(squeue -h -j "$ALL_JOBS" -o '%i' 2>/dev/null || true)
    [[ -z "$ACTIVE" ]] || { echo "Jobs still active: $ACTIVE" >&2; exit 1; }
  fi
  export FENIKS_REP_CODE=$(<"$FENIKS_REP_ROOT/CODE_DIR")
else
  SOURCE=$(realpath "${1:?coherent root}")
  SPEC=$(realpath "${2:?old spec JSON}")
  export FENIKS_REP_ROOT=$(realpath -m "${3:?new root}")
  CONFIG=$(realpath "${4:-configs/experiments/feniks_coherent_representation.yaml}")
  test ! -e "$FENIKS_REP_ROOT"
  test ! -e "$FENIKS_REP_ROOT.code.tar"
  mkdir -p "$(dirname "$FENIKS_REP_ROOT")"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_REP_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
  DIGEST=$(sha256sum "$FENIKS_REP_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_REP_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/coherent-representation-$DIGEST"
  mkdir -p "$FENIKS_REP_CODE"
  tar -xf "$FENIKS_REP_ROOT.code.tar" -C "$FENIKS_REP_CODE"
  [[ -e "$FENIKS_REP_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_REP_CODE/Data"
  cd "$FENIKS_REP_CODE"
  python -m scripts.feniks_coherent_representation prepare --source "$SOURCE" \
    --old-spec "$SPEC" --root "$FENIKS_REP_ROOT" --config "$CONFIG"
  printf '%s\n' "$FENIKS_REP_CODE" > "$FENIKS_REP_ROOT/CODE_DIR"
  printf '%s\n' "$DIGEST" > "$FENIKS_REP_ROOT/CODE_SHA256"
fi
cd "$FENIKS_REP_CODE"
mapfile -t SETTINGS < <(python - "$FENIKS_REP_ROOT" <<'PY'
import sys
from pathlib import Path
from scripts.feniks_coherent_representation import FACTORS, settings, complete
root=Path(sys.argv[1]); _,cfg,digest=settings(root)
for name in ('fit_minutes','report_minutes','cpu_threads'): print(cfg['resources'][name])
print(','.join(str(i) for i,s in enumerate(FACTORS) if not complete(root/s,digest)))
print(int(not complete(root/'sfh_zeros',digest)))
print(int(not complete(root/'report',digest)))
PY
)
[[ ${#SETTINGS[@]} == 6 ]] || { echo 'Invalid prepared state' >&2; exit 1; }
FIT_MIN=${SETTINGS[0]} REPORT_MIN=${SETTINGS[1]} THREADS=${SETTINGS[2]}
TASKS=${SETTINGS[3]} ZEROS_NEEDED=${SETTINGS[4]} REPORT_NEEDED=${SETTINGS[5]}
ALL_JOBS= FIT_JOB= ZEROS_JOB= REPORT_JOB=
ATTEMPT=$(date +%Y%m%d_%H%M%S)
record() {
  printf 'export ALL_JOBS=%q\nexport FIT_JOB=%q\nexport ZEROS_JOB=%q\nexport REPORT_JOB=%q\n' \
    "$ALL_JOBS" "$FIT_JOB" "$ZEROS_JOB" "$REPORT_JOB" > "$FENIKS_REP_ROOT/JOBS_$ATTEMPT.env"
  cp "$FENIKS_REP_ROOT/JOBS_$ATTEMPT.env" "$FENIKS_REP_ROOT/JOBS.env"
}
submit() {
  local mode=$1 gpu=$2 minutes=$3
  shift 3
  local resource=(--account="${FENIKS_CPU_ACCOUNT:-jrx@cpu}" --partition="${FENIKS_CPU_PARTITION:-cpu_p1}")
  if [[ $gpu == 1 ]]; then
    resource=(--account="${FENIKS_GPU_ACCOUNT:-jrx@h100}" --partition="${FENIKS_GPU_PARTITION:-gpu_p6}" --constraint=h100 --gres=gpu:1)
  fi
  local raw
  raw=$(sbatch --parsable --job-name="feniks_rep_$mode" "${resource[@]}" \
    --nodes=1 --ntasks=1 --cpus-per-task="$THREADS" --hint=nomultithread \
    --time="$minutes" --kill-on-invalid-dep=yes \
    --export="ALL,FENIKS_REP_MODE=$mode,FENIKS_REP_GPU=$gpu" \
    --output="$FENIKS_REP_ROOT/logs/$mode-$ATTEMPT-%A_%a.out" \
    --error="$FENIKS_REP_ROOT/logs/$mode-$ATTEMPT-%A_%a.err" \
    "$@" scripts/feniks_coherent_representation.slurm)
  LAST_JOB=${raw%%;*}
  ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$LAST_JOB"
}
if [[ -n "$TASKS" ]]; then
  submit fit 1 "$FIT_MIN" --array="$TASKS%2"
  FIT_JOB=$LAST_JOB; record
fi
if [[ $ZEROS_NEEDED == 1 ]]; then
  submit zeros 0 "$REPORT_MIN"
  ZEROS_JOB=$LAST_JOB; record
fi
if [[ $REPORT_NEEDED == 1 ]]; then
  AFTER=()
  [[ -z "$ALL_JOBS" ]] || AFTER=(--dependency="afterany:${ALL_JOBS//,/:}")
  submit report 0 "$REPORT_MIN" "${AFTER[@]}"
  REPORT_JOB=$LAST_JOB; record
fi
printf 'fit=%s zeros=%s report=%s\n' "$FIT_JOB" "$ZEROS_JOB" "$REPORT_JOB"
echo "Peak: 2 H100s; GPU allocation ceiling: $((2*FIT_MIN)) GPU-minutes, not a runtime estimate."
echo 'No new photometry or classifier; no production parent/posterior training.'
printf 'watch=bash scripts/watch_feniks_coherent_representation.sh %q\n' "$FENIKS_REP_ROOT"
