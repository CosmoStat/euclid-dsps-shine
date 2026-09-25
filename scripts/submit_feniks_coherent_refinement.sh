#!/bin/bash
# REPRESENTATION_ROOT NEW_ROOT [CONFIG], or --resume EXISTING_ROOT
set -Eeuo pipefail
REPO=$(pwd -P)
command -v sbatch >/dev/null
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
if [[ ${1:-} == --resume ]]; then
  export FENIKS_REFINE_ROOT=$(realpath "${2:?existing root}")
  if [[ -f "$FENIKS_REFINE_ROOT/JOBS.env" ]]; then
    source "$FENIKS_REFINE_ROOT/JOBS.env"
    ACTIVE=$(squeue -h -j "$ALL_JOBS" -o '%i' 2>/dev/null || true)
    [[ -z "$ACTIVE" ]] || { echo "Jobs still active: $ACTIVE" >&2; exit 1; }
  fi
  export FENIKS_REFINE_CODE=$(<"$FENIKS_REFINE_ROOT/CODE_DIR")
else
  SOURCE=$(realpath "${1:?completed representation root}")
  export FENIKS_REFINE_ROOT=$(realpath -m "${2:?new root}")
  CONFIG=$(realpath "${3:-configs/experiments/feniks_coherent_refinement.yaml}")
  test ! -e "$FENIKS_REFINE_ROOT"
  test ! -e "$FENIKS_REFINE_ROOT.code.tar"
  mkdir -p "$(dirname "$FENIKS_REFINE_ROOT")"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_REFINE_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
  DIGEST=$(sha256sum "$FENIKS_REFINE_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_REFINE_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/coherent-refinement-$DIGEST"
  mkdir -p "$FENIKS_REFINE_CODE"
  tar -xf "$FENIKS_REFINE_ROOT.code.tar" -C "$FENIKS_REFINE_CODE"
  [[ -e "$FENIKS_REFINE_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_REFINE_CODE/Data"
  cd "$FENIKS_REFINE_CODE"
  python -m scripts.feniks_coherent_refinement prepare --source "$SOURCE" \
    --root "$FENIKS_REFINE_ROOT" --config "$CONFIG"
  printf '%s\n' "$FENIKS_REFINE_CODE" > "$FENIKS_REFINE_ROOT/CODE_DIR"
  printf '%s\n' "$DIGEST" > "$FENIKS_REFINE_ROOT/CODE_SHA256"
fi
cd "$FENIKS_REFINE_CODE"
mapfile -t SETTINGS < <(python - "$FENIKS_REFINE_ROOT" <<'PY'
import sys
from pathlib import Path
from scripts.feniks_coherent_refinement import settings, complete
root=Path(sys.argv[1]); _,cfg,digest=settings(root)
for name in ('fit_minutes','audit_minutes','report_minutes','cpu_threads'): print(cfg['resources'][name])
for stage in ('physical','audit','report'): print(int(not complete(root/stage,digest)))
PY
)
[[ ${#SETTINGS[@]} == 7 ]] || { echo 'Invalid prepared state' >&2; exit 1; }
FIT_MIN=${SETTINGS[0]} AUDIT_MIN=${SETTINGS[1]} REPORT_MIN=${SETTINGS[2]} THREADS=${SETTINGS[3]}
ALL_JOBS= FIT_JOB= AUDIT_JOB= REPORT_JOB=
ATTEMPT=$(date +%Y%m%d_%H%M%S)
record() {
  printf 'export ALL_JOBS=%q\nexport FIT_JOB=%q\nexport AUDIT_JOB=%q\nexport REPORT_JOB=%q\n' \
    "$ALL_JOBS" "$FIT_JOB" "$AUDIT_JOB" "$REPORT_JOB" > "$FENIKS_REFINE_ROOT/JOBS_$ATTEMPT.env"
  cp "$FENIKS_REFINE_ROOT/JOBS_$ATTEMPT.env" "$FENIKS_REFINE_ROOT/JOBS.env"
}
submit() {
  local mode=$1 gpu=$2 minutes=$3
  shift 3
  local resource=(--account="${FENIKS_CPU_ACCOUNT:-jrx@cpu}" --partition="${FENIKS_CPU_PARTITION:-cpu_p1}")
  if [[ $gpu == 1 ]]; then
    resource=(--account="${FENIKS_GPU_ACCOUNT:-jrx@h100}" --partition="${FENIKS_GPU_PARTITION:-gpu_p6}" --constraint=h100 --gres=gpu:1)
  fi
  local raw
  raw=$(sbatch --parsable --job-name="feniks_refine_$mode" "${resource[@]}" \
    --nodes=1 --ntasks=1 --cpus-per-task="$THREADS" --hint=nomultithread \
    --time="$minutes" --kill-on-invalid-dep=yes \
    --export="ALL,FENIKS_REFINE_MODE=$mode,FENIKS_REFINE_GPU=$gpu" \
    --output="$FENIKS_REFINE_ROOT/logs/$mode-$ATTEMPT-%A.out" \
    --error="$FENIKS_REFINE_ROOT/logs/$mode-$ATTEMPT-%A.err" \
    "$@" scripts/feniks_coherent_refinement.slurm)
  LAST_JOB=${raw%%;*}
  [[ $LAST_JOB =~ ^[0-9]+$ ]] || { echo "Invalid job ID: $raw" >&2; exit 1; }
  ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$LAST_JOB"
}
if [[ ${SETTINGS[4]} == 1 ]]; then
  submit fit 1 "$FIT_MIN"
  FIT_JOB=$LAST_JOB; record
fi
if [[ ${SETTINGS[5]} == 1 ]]; then
  submit audit 0 "$AUDIT_MIN"
  AUDIT_JOB=$LAST_JOB; record
fi
if [[ ${SETTINGS[6]} == 1 ]]; then
  AFTER=()
  [[ -z "$ALL_JOBS" ]] || AFTER=(--dependency="afterany:${ALL_JOBS//,/:}")
  submit report 0 "$REPORT_MIN" "${AFTER[@]}"
  REPORT_JOB=$LAST_JOB; record
fi
printf 'fit=%s audit=%s report=%s\n' "$FIT_JOB" "$AUDIT_JOB" "$REPORT_JOB"
echo "Peak: 1 H100; GPU allocation ceiling: $FIT_MIN minutes, not a runtime estimate."
echo 'No new photometry; native SFH replay only. SFH conditional is frozen.'
printf 'watch=bash scripts/watch_feniks_coherent_refinement.sh %q\n' "$FENIKS_REFINE_ROOT"
