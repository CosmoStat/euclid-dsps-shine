#!/bin/bash
# Usage: SOURCE_LOW_RANK NEW_ROOT [CONFIG], or --resume EXISTING_ROOT
set -Eeuo pipefail
REPO=$(pwd -P)
command -v sbatch >/dev/null
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true EUCLID_DSPS_REQUIRE_GPU=0
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1

if [[ "${1:-}" == --resume ]]; then
  export FENIKS_PRECISION_ROOT=$(realpath "${2:?existing precision root}")
  if [[ -f "$FENIKS_PRECISION_ROOT/JOBS.env" ]]; then
    source "$FENIKS_PRECISION_ROOT/JOBS.env"
    ACTIVE=$(squeue -h -j "$ALL_JOBS" -o '%i' 2>/dev/null || true)
    [[ -z "$ACTIVE" ]] || { echo "Jobs are still active: $ACTIVE" >&2; exit 1; }
  fi
  export FENIKS_PRECISION_CODE=$(<"$FENIKS_PRECISION_ROOT/CODE_DIR")
  test -d "$FENIKS_PRECISION_CODE"
  CONFIG="$FENIKS_PRECISION_ROOT/precision.yaml"
else
  SOURCE=$(realpath "${1:?completed or partial low-rank audit}")
  export FENIKS_PRECISION_ROOT=$(realpath -m "${2:?new precision root}")
  CONFIG=$(realpath "${3:-configs/experiments/feniks_population_precision.yaml}")
  test ! -e "$FENIKS_PRECISION_ROOT"
  test ! -e "$FENIKS_PRECISION_ROOT.code.tar"
  mkdir -p "$(dirname "$FENIKS_PRECISION_ROOT")"
  tar --exclude='__pycache__' --exclude='*.pyc' \
    -cf "$FENIKS_PRECISION_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
  tar --dereference -rf "$FENIKS_PRECISION_ROOT.code.tar" filters
  DIGEST=$(sha256sum "$FENIKS_PRECISION_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_PRECISION_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/precision-$DIGEST"
  mkdir -p "$FENIKS_PRECISION_CODE"
  tar -xf "$FENIKS_PRECISION_ROOT.code.tar" -C "$FENIKS_PRECISION_CODE"
  [[ -e "$FENIKS_PRECISION_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_PRECISION_CODE/Data"
  cd "$FENIKS_PRECISION_CODE"
  python -m scripts.feniks_population_precision prepare --source "$SOURCE" \
    --root "$FENIKS_PRECISION_ROOT" --config "$CONFIG"
  printf '%s\n' "$FENIKS_PRECISION_CODE" > "$FENIKS_PRECISION_ROOT/CODE_DIR"
  printf '%s\n' "$DIGEST" > "$FENIKS_PRECISION_ROOT/CODE_SHA256"
fi
cd "$FENIKS_PRECISION_CODE"

mapfile -t SETTINGS < <(python - "$FENIKS_PRECISION_ROOT" <<'PY'
import sys
from pathlib import Path
from scripts.feniks_population_precision import ARMS, RATIOS, config, receipt_valid
root = Path(sys.argv[1])
_, s = config(root)
for key in ('cache_minutes', 'decoder_minutes', 'metric_minutes', 'bootstrap_minutes',
            'report_minutes', 'bootstrap_concurrency', 'cpu_threads'):
    value = s['resources'][key]
    assert isinstance(value, int) and value > 0
    print(value)
print(int(not receipt_valid(root / 'cache')))
print(int(not receipt_valid(root / 'decoder')))
print(','.join(str(i) for i, arm in enumerate(ARMS) if not receipt_valid(root / 'metric' / arm)))
print(','.join(str(i) for i in range(s['bootstraps']) if not all(
    receipt_valid(root / 'bootstrap' / f'repeat_{i:03d}' / f'{a}_{r}')
    for a in ARMS for r in RATIOS)))
PY
)
[[ ${#SETTINGS[@]} == 11 ]] || { echo 'Invalid settings/preflight' >&2; exit 1; }
CACHE_MIN=${SETTINGS[0]} DECODER_MIN=${SETTINGS[1]} METRIC_MIN=${SETTINGS[2]}
BOOT_MIN=${SETTINGS[3]} REPORT_MIN=${SETTINGS[4]} CONCURRENCY=${SETTINGS[5]} THREADS=${SETTINGS[6]}
CACHE_NEEDED=${SETTINGS[7]} DECODER_NEEDED=${SETTINGS[8]}
METRIC_TASKS=${SETTINGS[9]} BOOT_TASKS=${SETTINGS[10]}
ATTEMPT=$(date +%Y%m%d_%H%M%S)
JOBS_FILE="$FENIKS_PRECISION_ROOT/JOBS_$ATTEMPT.env"
ALL_JOBS= CACHE_JOB= DECODER_JOB= METRIC_JOB= BOOTSTRAP_JOB= REPORT_JOB=

record_jobs() {
  printf 'export ALL_JOBS=%q\nexport CACHE_JOB=%q\nexport DECODER_JOB=%q\nexport METRIC_JOB=%q\nexport BOOTSTRAP_JOB=%q\nexport REPORT_JOB=%q\n' \
    "$ALL_JOBS" "$CACHE_JOB" "$DECODER_JOB" "$METRIC_JOB" "$BOOTSTRAP_JOB" "$REPORT_JOB" > "$JOBS_FILE"
  cp "$JOBS_FILE" "$FENIKS_PRECISION_ROOT/JOBS.env"
}
submit() {
  local mode=$1 gpu=$2 minutes=$3
  shift 3
  # Jean-Zay sets memory from the allocation; explicit memory flags are rejected.
  local resource=(--account="${FENIKS_CPU_ACCOUNT:-jrx@cpu}" --partition="${FENIKS_CPU_PARTITION:-cpu_p1}")
  if [[ $gpu == 1 ]]; then
    resource=(--account="${FENIKS_GPU_ACCOUNT:-jrx@h100}" --partition="${FENIKS_GPU_PARTITION:-gpu_p6}" --constraint=h100 --gres=gpu:1)
  fi
  local raw
  raw=$(sbatch --parsable --job-name="feniks_precision_$mode" "${resource[@]}" \
    --nodes=1 --ntasks=1 --cpus-per-task="$THREADS" --hint=nomultithread \
    --time="$minutes" --kill-on-invalid-dep=yes \
    --export="ALL,FENIKS_PRECISION_MODE=$mode,FENIKS_PRECISION_GPU=$gpu" \
    --output="$FENIKS_PRECISION_ROOT/logs/$mode-$ATTEMPT-%A_%a.out" \
    --error="$FENIKS_PRECISION_ROOT/logs/$mode-$ATTEMPT-%A_%a.err" \
    "$@" scripts/feniks_population_precision.slurm)
  LAST_JOB=${raw%%;*}
  ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$LAST_JOB"
}
DEP=()
if [[ $CACHE_NEEDED == 1 ]]; then
  submit cache 1 "$CACHE_MIN"; CACHE_JOB=$LAST_JOB; record_jobs
  DEP=(--dependency="afterok:$CACHE_JOB")
fi
if [[ $DECODER_NEEDED == 1 ]]; then
  submit decoder 1 "$DECODER_MIN"; DECODER_JOB=$LAST_JOB; record_jobs
fi
if [[ -n "$METRIC_TASKS" ]]; then
  submit metric 0 "$METRIC_MIN" "${DEP[@]}" --array="$METRIC_TASKS%2"
  METRIC_JOB=$LAST_JOB; record_jobs
fi
if [[ -n "$BOOT_TASKS" ]]; then
  submit bootstrap 0 "$BOOT_MIN" "${DEP[@]}" --array="$BOOT_TASKS%$CONCURRENCY"
  BOOTSTRAP_JOB=$LAST_JOB; record_jobs
fi
AFTER=()
[[ -z "$ALL_JOBS" ]] || AFTER=(--dependency="afterany:${ALL_JOBS//,/:}")
submit report 0 "$REPORT_MIN" "${AFTER[@]}"
REPORT_JOB=$LAST_JOB; record_jobs
printf 'cache=%s decoder=%s metrics=%s bootstrap=%s report=%s\n' \
  "$CACHE_JOB" "$DECODER_JOB" "$METRIC_JOB" "$BOOTSTRAP_JOB" "$REPORT_JOB"
echo "Peak: 2 H100; fits/metrics use CPU. GPU allocation ceiling: $((CACHE_MIN + DECODER_MIN)) minutes."
echo "Task ceilings: cache ${CACHE_MIN}m; decoder ${DECODER_MIN}m; bootstrap ${BOOT_MIN}m, concurrency $CONCURRENCY."
printf 'watch=bash scripts/watch_feniks_population_precision.sh %q\n' "$FENIKS_PRECISION_ROOT"
