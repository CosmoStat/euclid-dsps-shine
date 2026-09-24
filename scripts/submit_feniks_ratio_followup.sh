#!/bin/bash
# Usage: bash scripts/submit_feniks_ratio_followup.sh SOURCE_RATIO_ROOT NEW_ROOT [CONFIG]
set -Eeuo pipefail

SOURCE=$(realpath "${1:?completed ratio-ladder root}")
export FENIKS_RATIO_FOLLOWUP_ROOT=$(realpath -m "${2:?new follow-up root}")
CONFIG=$(realpath "${3:-configs/experiments/feniks_ratio_followup.yaml}")
REPO=$(pwd -P)

test ! -e "$FENIKS_RATIO_FOLLOWUP_ROOT"
test ! -e "$FENIKS_RATIO_FOLLOWUP_ROOT.code.tar"
command -v sbatch >/dev/null
mkdir -p "$(dirname "$FENIKS_RATIO_FOLLOWUP_ROOT")"

tar --exclude='__pycache__' --exclude='*.pyc' \
  -cf "$FENIKS_RATIO_FOLLOWUP_ROOT.code.tar" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$FENIKS_RATIO_FOLLOWUP_ROOT.code.tar" filters
DIGEST=$(sha256sum "$FENIKS_RATIO_FOLLOWUP_ROOT.code.tar" | cut -d' ' -f1)
export FENIKS_RATIO_FOLLOWUP_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/ratio-followup-$DIGEST"
mkdir -p "$FENIKS_RATIO_FOLLOWUP_CODE"
tar -xf "$FENIKS_RATIO_FOLLOWUP_ROOT.code.tar" -C "$FENIKS_RATIO_FOLLOWUP_CODE"
[[ -e "$FENIKS_RATIO_FOLLOWUP_CODE/Data" ]] || \
  ln -s "$REPO/Data" "$FENIKS_RATIO_FOLLOWUP_CODE/Data"

cd "$FENIKS_RATIO_FOLLOWUP_CODE"
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu \
  EUCLID_DSPS_REQUIRE_GPU=0 JAX_ENABLE_X64=true \
  python -m scripts.feniks_ratio_followup prepare \
    --source "$SOURCE" --root "$FENIKS_RATIO_FOLLOWUP_ROOT" --config "$CONFIG"
printf '%s\n' "$FENIKS_RATIO_FOLLOWUP_CODE" > "$FENIKS_RATIO_FOLLOWUP_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FENIKS_RATIO_FOLLOWUP_ROOT/CODE_SHA256"

mapfile -t RESOURCE < <(python - "$CONFIG" <<'PY'
import sys
import yaml

resources = yaml.safe_load(open(sys.argv[1]))["resources"]
for key in ("classifier_hours", "report_hours", "classifier_concurrency"):
    value = resources[key]
    assert isinstance(value, int) and value > 0
    print(value)
PY
)
CLASSIFIER_HOURS=${RESOURCE[0]}
REPORT_HOURS=${RESOURCE[1]}
CLASSIFIER_CONCURRENCY=${RESOURCE[2]}

: > "$FENIKS_RATIO_FOLLOWUP_ROOT/JOBS.env"
record() {
  printf 'export %s=%q\n' "$1" "$2" >> "$FENIKS_RATIO_FOLLOWUP_ROOT/JOBS.env"
}

export FENIKS_RATIO_FOLLOWUP_MODE=run
RAW=$(sbatch --parsable --time="$CLASSIFIER_HOURS:00:00" \
  --array="0-3%$CLASSIFIER_CONCURRENCY" --export=ALL \
  --output="$FENIKS_RATIO_FOLLOWUP_ROOT/logs/classifier-%A_%a.out" \
  --error="$FENIKS_RATIO_FOLLOWUP_ROOT/logs/classifier-%A_%a.err" \
  scripts/feniks_ratio_followup.slurm)
CLASSIFIER_JOB=${RAW%%;*}
record CLASSIFIER_JOB "$CLASSIFIER_JOB"
record ALL_JOBS "$CLASSIFIER_JOB"

export FENIKS_RATIO_FOLLOWUP_MODE=report
RAW=$(sbatch --parsable --time="$REPORT_HOURS:00:00" \
  --dependency="afterok:$CLASSIFIER_JOB" --export=ALL \
  --output="$FENIKS_RATIO_FOLLOWUP_ROOT/logs/report-%j.out" \
  --error="$FENIKS_RATIO_FOLLOWUP_ROOT/logs/report-%j.err" \
  scripts/feniks_ratio_followup.slurm)
REPORT_JOB=${RAW%%;*}
record REPORT_JOB "$REPORT_JOB"
record ALL_JOBS "$CLASSIFIER_JOB,$REPORT_JOB"

echo "classifier=$CLASSIFIER_JOB report=$REPORT_JOB"
echo "Banks reused; no DSPS simulation. Peak H100s: $CLASSIFIER_CONCURRENCY"
echo "Allocation ceiling: $((4 * CLASSIFIER_HOURS + REPORT_HOURS)) H100-hours, not expected runtime."
printf 'watch=bash scripts/watch_feniks_ratio_followup.sh %q\n' \
  "$FENIKS_RATIO_FOLLOWUP_ROOT"
