#!/bin/bash
# Usage: bash scripts/submit_feniks_failure_modes.sh PARENT DIAGNOSTICS NEW_ROOT [CONFIG]
set -Eeuo pipefail

PARENT=$(realpath "${1:?completed forward-population root}")
DIAGNOSTICS=$(realpath "${2:?completed forward-diagnostics root}")
export FENIKS_AUDIT_ROOT=$(realpath -m "${3:?new failure-mode audit root}")
REPO=$(pwd -P)
CONFIG=$(realpath "${4:-configs/experiments/feniks_failure_modes_r29.yaml}")

test ! -e "$FENIKS_AUDIT_ROOT"
test ! -e "$FENIKS_AUDIT_ROOT.code.tar"
command -v sbatch >/dev/null
mkdir -p "$(dirname "$FENIKS_AUDIT_ROOT")"

tar --exclude='__pycache__' --exclude='*.pyc' \
  -cf "$FENIKS_AUDIT_ROOT.code.tar" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$FENIKS_AUDIT_ROOT.code.tar" filters
DIGEST=$(sha256sum "$FENIKS_AUDIT_ROOT.code.tar" | cut -d' ' -f1)
export FENIKS_AUDIT_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/failure-modes-$DIGEST"
mkdir -p "$FENIKS_AUDIT_CODE"
tar -xf "$FENIKS_AUDIT_ROOT.code.tar" -C "$FENIKS_AUDIT_CODE"
[[ -e "$FENIKS_AUDIT_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_AUDIT_CODE/Data"

cd "$FENIKS_AUDIT_CODE"
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu \
  EUCLID_DSPS_REQUIRE_GPU=0 JAX_ENABLE_X64=true \
  python -m scripts.feniks_failure_modes prepare \
    --parent "$PARENT" \
    --diagnostics "$DIAGNOSTICS" \
    --root "$FENIKS_AUDIT_ROOT" \
    --config "$CONFIG"

printf '%s\n' "$FENIKS_AUDIT_CODE" > "$FENIKS_AUDIT_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FENIKS_AUDIT_ROOT/CODE_SHA256"

mapfile -t RESOURCE < <(python - "$FENIKS_AUDIT_ROOT/audit.yaml" <<'PY'
import sys
import yaml

resources = yaml.safe_load(open(sys.argv[1]))["resources"]
for key in ("array_hours", "report_hours", "concurrency"):
    value = resources[key]
    assert isinstance(value, int) and value > 0
    print(value)
PY
)
ARRAY_HOURS=${RESOURCE[0]}
REPORT_HOURS=${RESOURCE[1]}
CONCURRENCY=${RESOURCE[2]}

: > "$FENIKS_AUDIT_ROOT/JOBS.env"
export FENIKS_AUDIT_MODE=task
RAW=$(sbatch --parsable \
  --time="$ARRAY_HOURS:00:00" \
  --array="0-4%$CONCURRENCY" \
  --export=ALL \
  --output="$FENIKS_AUDIT_ROOT/logs/task-%A_%a.out" \
  --error="$FENIKS_AUDIT_ROOT/logs/task-%A_%a.err" \
  "$FENIKS_AUDIT_CODE/scripts/feniks_failure_modes.slurm")
ARRAY_JOB=${RAW%%;*}

export FENIKS_AUDIT_MODE=report
RAW=$(sbatch --parsable \
  --time="$REPORT_HOURS:00:00" \
  --dependency="afterok:$ARRAY_JOB" \
  --export=ALL \
  --output="$FENIKS_AUDIT_ROOT/logs/report-%j.out" \
  --error="$FENIKS_AUDIT_ROOT/logs/report-%j.err" \
  "$FENIKS_AUDIT_CODE/scripts/feniks_failure_modes.slurm")
REPORT_JOB=${RAW%%;*}

printf 'export ARRAY_JOB=%q\n' "$ARRAY_JOB" >> "$FENIKS_AUDIT_ROOT/JOBS.env"
printf 'export REPORT_JOB=%q\n' "$REPORT_JOB" >> "$FENIKS_AUDIT_ROOT/JOBS.env"
printf 'export ALL_JOBS=%q\n' "$ARRAY_JOB,$REPORT_JOB" >> "$FENIKS_AUDIT_ROOT/JOBS.env"

echo "array=$ARRAY_JOB"
echo "report=$REPORT_JOB"
echo "Peak H100s: $CONCURRENCY; allocation ceiling: $((5 * ARRAY_HOURS + REPORT_HOURS)) H100-hours."
printf 'watch=bash scripts/watch_feniks_failure_modes.sh %q\n' "$FENIKS_AUDIT_ROOT"
