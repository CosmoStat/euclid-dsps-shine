#!/bin/bash
# Usage: bash scripts/submit_feniks_weighted_truth_flow_followup.sh SOURCE_CAPACITY NEW_ROOT [CONFIG]
set -Eeuo pipefail

SOURCE_CAPACITY=$(realpath "${1:?completed weighted-truth capacity root}")
export FENIKS_FOLLOWUP_ROOT=$(realpath -m "${2:?new follow-up root}")
REPO=$(pwd -P)
CONFIG=$(realpath "${3:-configs/experiments/feniks_weighted_truth_flow_followup.yaml}")

test ! -e "$FENIKS_FOLLOWUP_ROOT"
test ! -e "$FENIKS_FOLLOWUP_ROOT.code.tar"
command -v sbatch >/dev/null
mkdir -p "$(dirname "$FENIKS_FOLLOWUP_ROOT")"

tar --exclude='__pycache__' --exclude='*.pyc' \
  -cf "$FENIKS_FOLLOWUP_ROOT.code.tar" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$FENIKS_FOLLOWUP_ROOT.code.tar" filters
DIGEST=$(sha256sum "$FENIKS_FOLLOWUP_ROOT.code.tar" | cut -d' ' -f1)
export FENIKS_FOLLOWUP_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/truth-followup-$DIGEST"
mkdir -p "$FENIKS_FOLLOWUP_CODE"
tar -xf "$FENIKS_FOLLOWUP_ROOT.code.tar" -C "$FENIKS_FOLLOWUP_CODE"
[[ -e "$FENIKS_FOLLOWUP_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_FOLLOWUP_CODE/Data"

cd "$FENIKS_FOLLOWUP_CODE"
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu \
  EUCLID_DSPS_REQUIRE_GPU=0 JAX_ENABLE_X64=true \
  python -m scripts.feniks_weighted_truth_flow_followup prepare \
    --source-capacity "$SOURCE_CAPACITY" \
    --root "$FENIKS_FOLLOWUP_ROOT" \
    --config "$CONFIG"

printf '%s\n' "$FENIKS_FOLLOWUP_CODE" > "$FENIKS_FOLLOWUP_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FENIKS_FOLLOWUP_ROOT/CODE_SHA256"
mapfile -t RESOURCE < <(python - "$FENIKS_FOLLOWUP_ROOT/followup.yaml" <<'PY'
import sys
import yaml

resources = yaml.safe_load(open(sys.argv[1]))["resources"]
for key in ("task_hours", "report_hours", "concurrency"):
    value = resources[key]
    assert isinstance(value, int) and value > 0
    print(value)
PY
)
TASK_HOURS=${RESOURCE[0]}
REPORT_HOURS=${RESOURCE[1]}
CONCURRENCY=${RESOURCE[2]}

: > "$FENIKS_FOLLOWUP_ROOT/JOBS.env"
export FENIKS_FOLLOWUP_MODE=fit
RAW=$(sbatch --parsable \
  --time="$TASK_HOURS:00:00" \
  --array="0-3%$CONCURRENCY" \
  --export=ALL \
  --output="$FENIKS_FOLLOWUP_ROOT/logs/fit-%A_%a.out" \
  --error="$FENIKS_FOLLOWUP_ROOT/logs/fit-%A_%a.err" \
  "$FENIKS_FOLLOWUP_CODE/scripts/feniks_weighted_truth_flow_followup.slurm")
FIT_JOB=${RAW%%;*}

export FENIKS_FOLLOWUP_MODE=report
RAW=$(sbatch --parsable \
  --time="$REPORT_HOURS:00:00" \
  --dependency="afterok:$FIT_JOB" \
  --export=ALL \
  --output="$FENIKS_FOLLOWUP_ROOT/logs/report-%j.out" \
  --error="$FENIKS_FOLLOWUP_ROOT/logs/report-%j.err" \
  "$FENIKS_FOLLOWUP_CODE/scripts/feniks_weighted_truth_flow_followup.slurm")
REPORT_JOB=${RAW%%;*}

printf 'export FIT_JOB=%q\n' "$FIT_JOB" >> "$FENIKS_FOLLOWUP_ROOT/JOBS.env"
printf 'export REPORT_JOB=%q\n' "$REPORT_JOB" >> "$FENIKS_FOLLOWUP_ROOT/JOBS.env"
printf 'export ALL_JOBS=%q\n' "$FIT_JOB,$REPORT_JOB" >> "$FENIKS_FOLLOWUP_ROOT/JOBS.env"

echo "fit=$FIT_JOB"
echo "report=$REPORT_JOB"
echo "Peak H100s: $CONCURRENCY; allocation ceiling: $((4 * TASK_HOURS + REPORT_HOURS)) H100-hours."
printf 'watch=bash scripts/watch_feniks_weighted_truth_flow_followup.sh %q\n' "$FENIKS_FOLLOWUP_ROOT"
