#!/bin/bash
# Usage: bash scripts/submit_feniks_weighted_truth_flow_capacity.sh PARENT NEW_ROOT [CONFIG]
set -Eeuo pipefail

PARENT=$(realpath "${1:?completed forward-population root}")
export FENIKS_CAPACITY_ROOT=$(realpath -m "${2:?new weighted-capacity root}")
REPO=$(pwd -P)
CONFIG=$(realpath "${3:-configs/experiments/feniks_weighted_truth_flow_capacity.yaml}")

test ! -e "$FENIKS_CAPACITY_ROOT"
test ! -e "$FENIKS_CAPACITY_ROOT.code.tar"
command -v sbatch >/dev/null
mkdir -p "$(dirname "$FENIKS_CAPACITY_ROOT")"

tar --exclude='__pycache__' --exclude='*.pyc' \
  -cf "$FENIKS_CAPACITY_ROOT.code.tar" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$FENIKS_CAPACITY_ROOT.code.tar" filters
DIGEST=$(sha256sum "$FENIKS_CAPACITY_ROOT.code.tar" | cut -d' ' -f1)
export FENIKS_CAPACITY_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/weighted-truth-$DIGEST"
mkdir -p "$FENIKS_CAPACITY_CODE"
tar -xf "$FENIKS_CAPACITY_ROOT.code.tar" -C "$FENIKS_CAPACITY_CODE"
[[ -e "$FENIKS_CAPACITY_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_CAPACITY_CODE/Data"

cd "$FENIKS_CAPACITY_CODE"
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu \
  EUCLID_DSPS_REQUIRE_GPU=0 JAX_ENABLE_X64=true \
  python -m scripts.feniks_weighted_truth_flow_capacity prepare \
    --parent "$PARENT" \
    --root "$FENIKS_CAPACITY_ROOT" \
    --config "$CONFIG"

printf '%s\n' "$FENIKS_CAPACITY_CODE" > "$FENIKS_CAPACITY_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FENIKS_CAPACITY_ROOT/CODE_SHA256"
mapfile -t RESOURCE < <(python - "$FENIKS_CAPACITY_ROOT/capacity.yaml" <<'PY'
import sys
import yaml

resources = yaml.safe_load(open(sys.argv[1]))["resources"]
for key in ("replica_hours", "report_hours", "concurrency"):
    value = resources[key]
    assert isinstance(value, int) and value > 0
    print(value)
PY
)
REPLICA_HOURS=${RESOURCE[0]}
REPORT_HOURS=${RESOURCE[1]}
CONCURRENCY=${RESOURCE[2]}

: > "$FENIKS_CAPACITY_ROOT/JOBS.env"
export FENIKS_CAPACITY_MODE=fit
RAW=$(sbatch --parsable \
  --time="$REPLICA_HOURS:00:00" \
  --array="0-1%$CONCURRENCY" \
  --export=ALL \
  --output="$FENIKS_CAPACITY_ROOT/logs/fit-%A_%a.out" \
  --error="$FENIKS_CAPACITY_ROOT/logs/fit-%A_%a.err" \
  "$FENIKS_CAPACITY_CODE/scripts/feniks_weighted_truth_flow_capacity.slurm")
FIT_JOB=${RAW%%;*}

export FENIKS_CAPACITY_MODE=report
RAW=$(sbatch --parsable \
  --time="$REPORT_HOURS:00:00" \
  --dependency="afterok:$FIT_JOB" \
  --export=ALL \
  --output="$FENIKS_CAPACITY_ROOT/logs/report-%j.out" \
  --error="$FENIKS_CAPACITY_ROOT/logs/report-%j.err" \
  "$FENIKS_CAPACITY_CODE/scripts/feniks_weighted_truth_flow_capacity.slurm")
REPORT_JOB=${RAW%%;*}

printf 'export FIT_JOB=%q\n' "$FIT_JOB" >> "$FENIKS_CAPACITY_ROOT/JOBS.env"
printf 'export REPORT_JOB=%q\n' "$REPORT_JOB" >> "$FENIKS_CAPACITY_ROOT/JOBS.env"
printf 'export ALL_JOBS=%q\n' "$FIT_JOB,$REPORT_JOB" >> "$FENIKS_CAPACITY_ROOT/JOBS.env"

echo "fit=$FIT_JOB"
echo "report=$REPORT_JOB"
echo "Peak H100s: $CONCURRENCY; allocation ceiling: $((2 * REPLICA_HOURS + REPORT_HOURS)) H100-hours."
printf 'watch=bash scripts/watch_feniks_weighted_truth_flow_capacity.sh %q\n' "$FENIKS_CAPACITY_ROOT"
