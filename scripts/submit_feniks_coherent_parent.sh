#!/bin/bash
# SOURCE_DATASET DECODER_CONFIG NEW_ROOT [CONFIG], or --resume EXISTING_ROOT
set -Eeuo pipefail
REPO=$(pwd -P)
command -v sbatch >/dev/null
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
if [[ "${1:-}" == --resume ]]; then
  export FENIKS_PARENT_ROOT=$(realpath "${2:?existing root}")
  if [[ -f "$FENIKS_PARENT_ROOT/JOBS.env" ]]; then
    source "$FENIKS_PARENT_ROOT/JOBS.env"
    ACTIVE=$(squeue -h -j "$ALL_JOBS" -o '%i' 2>/dev/null || true)
    [[ -z "$ACTIVE" ]] || { echo "Jobs still active: $ACTIVE" >&2; exit 1; }
  fi
  export FENIKS_PARENT_CODE=$(<"$FENIKS_PARENT_ROOT/CODE_DIR")
  test -d "$FENIKS_PARENT_CODE"
else
  SOURCE=$(realpath "${1:?raw proposal dataset root}")
  DECODER=$(realpath "${2:?current source_config.yaml}")
  export FENIKS_PARENT_ROOT=$(realpath -m "${3:?new root}")
  CONFIG=$(realpath "${4:-configs/experiments/feniks_coherent_parent.yaml}")
  test ! -e "$FENIKS_PARENT_ROOT"
  test ! -e "$FENIKS_PARENT_ROOT.code.tar"
  mkdir -p "$(dirname "$FENIKS_PARENT_ROOT")"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_PARENT_ROOT.code.tar" \
    euclid_dsps scripts configs pyproject.toml
  tar --dereference -rf "$FENIKS_PARENT_ROOT.code.tar" filters
  DIGEST=$(sha256sum "$FENIKS_PARENT_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_PARENT_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/coherent-parent-$DIGEST"
  mkdir -p "$FENIKS_PARENT_CODE"
  tar -xf "$FENIKS_PARENT_ROOT.code.tar" -C "$FENIKS_PARENT_CODE"
  [[ -e "$FENIKS_PARENT_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_PARENT_CODE/Data"
  cd "$FENIKS_PARENT_CODE"
  python -m scripts.feniks_coherent_parent prepare --source "$SOURCE" \
    --decoder-config "$DECODER" --root "$FENIKS_PARENT_ROOT" --config "$CONFIG"
  printf '%s\n' "$FENIKS_PARENT_CODE" > "$FENIKS_PARENT_ROOT/CODE_DIR"
  printf '%s\n' "$DIGEST" > "$FENIKS_PARENT_ROOT/CODE_SHA256"
fi
cd "$FENIKS_PARENT_CODE"
mapfile -t SETTINGS < <(python - "$FENIKS_PARENT_ROOT" <<'PY'
import sys
from pathlib import Path
from scripts.feniks_coherent_parent import SPLITS, settings, complete
root = Path(sys.argv[1]); m, cfg, digest = settings(root)
for name in ('sample_minutes','photometry_minutes','report_minutes','concurrency','cpu_threads'):
    value = cfg['resources'][name]
    assert isinstance(value, int) and value > 0
    print(value)
print(','.join(str(i) for i,s in enumerate(SPLITS) if not complete(root/'sampling'/s,digest)))
print(','.join(str(i) for i in range(len(m['tasks'])) if not complete(root/'photometry'/f'task_{i:03d}',digest)))
print(int(not complete(root/'report',digest)))
print(len(m['tasks']))
PY
)
[[ ${#SETTINGS[@]} == 9 ]] || { echo 'Invalid prepared state' >&2; exit 1; }
SAMPLE_MIN=${SETTINGS[0]} PHOTO_MIN=${SETTINGS[1]} REPORT_MIN=${SETTINGS[2]}
CONCURRENCY=${SETTINGS[3]} THREADS=${SETTINGS[4]} SAMPLE_TASKS=${SETTINGS[5]}
PHOTO_TASKS=${SETTINGS[6]} REPORT_NEEDED=${SETTINGS[7]} TOTAL_TASKS=${SETTINGS[8]}
ATTEMPT=$(date +%Y%m%d_%H%M%S)
ALL_JOBS= SAMPLE_JOB= PHOTO_JOB= REPORT_JOB=
record() {
  printf 'export ALL_JOBS=%q\nexport SAMPLE_JOB=%q\nexport PHOTO_JOB=%q\nexport REPORT_JOB=%q\n' \
    "$ALL_JOBS" "$SAMPLE_JOB" "$PHOTO_JOB" "$REPORT_JOB" > "$FENIKS_PARENT_ROOT/JOBS_$ATTEMPT.env"
  cp "$FENIKS_PARENT_ROOT/JOBS_$ATTEMPT.env" "$FENIKS_PARENT_ROOT/JOBS.env"
}
submit() {
  local mode=$1 gpu=$2 minutes=$3
  shift 3
  local resource=(--account="${FENIKS_CPU_ACCOUNT:-jrx@cpu}" --partition="${FENIKS_CPU_PARTITION:-cpu_p1}")
  if [[ $gpu == 1 ]]; then
    resource=(--account="${FENIKS_GPU_ACCOUNT:-jrx@h100}" --partition="${FENIKS_GPU_PARTITION:-gpu_p6}" --constraint=h100 --gres=gpu:1)
  fi
  local raw
  raw=$(sbatch --parsable --job-name="feniks_parent_$mode" "${resource[@]}" \
    --nodes=1 --ntasks=1 --cpus-per-task="$THREADS" --hint=nomultithread \
    --time="$minutes" --kill-on-invalid-dep=yes \
    --export="ALL,FENIKS_PARENT_MODE=$mode,FENIKS_PARENT_GPU=$gpu" \
    --output="$FENIKS_PARENT_ROOT/logs/$mode-$ATTEMPT-%A_%a.out" \
    --error="$FENIKS_PARENT_ROOT/logs/$mode-$ATTEMPT-%A_%a.err" \
    "$@" scripts/feniks_coherent_parent.slurm)
  LAST_JOB=${raw%%;*}
  ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$LAST_JOB"
}
DEP=()
if [[ -n "$SAMPLE_TASKS" ]]; then
  submit sample 0 "$SAMPLE_MIN" --array="$SAMPLE_TASKS%3"
  SAMPLE_JOB=$LAST_JOB; record
  DEP=(--dependency="afterok:$SAMPLE_JOB")
fi
if [[ -n "$PHOTO_TASKS" ]]; then
  submit photometry 1 "$PHOTO_MIN" --array="$PHOTO_TASKS%$CONCURRENCY" "${DEP[@]}"
  PHOTO_JOB=$LAST_JOB; record
fi
if [[ $REPORT_NEEDED == 1 ]]; then
  AFTER=()
  [[ -z "$ALL_JOBS" ]] || AFTER=(--dependency="afterany:${ALL_JOBS//,/:}")
  submit report 0 "$REPORT_MIN" "${AFTER[@]}"
  REPORT_JOB=$LAST_JOB; record
fi
printf 'sampling=%s photometry=%s report=%s\n' "$SAMPLE_JOB" "$PHOTO_JOB" "$REPORT_JOB"
echo "No training. At most $CONCURRENCY concurrent H100s; ${PHOTO_MIN}m ceiling per photometry task."
echo "Full-run GPU allocation ceiling: $((TOTAL_TASKS * PHOTO_MIN)) GPU-minutes, not a runtime prediction."
printf 'watch=bash scripts/watch_feniks_coherent_parent.sh %q\n' "$FENIKS_PARENT_ROOT"
