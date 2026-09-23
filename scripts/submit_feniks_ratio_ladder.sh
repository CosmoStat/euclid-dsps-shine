#!/bin/bash
# Usage: bash scripts/submit_feniks_ratio_ladder.sh CLEAN_ROOT NEW_ROOT [CONFIG]
set -Eeuo pipefail

CLEAN=$(realpath "${1:?completed clean-parent root}")
export FENIKS_RATIO_ROOT=$(realpath -m "${2:?new ratio-ladder root}")
CONFIG=$(realpath "${3:-configs/experiments/feniks_ratio_ladder.yaml}")
REPO=$(pwd -P)

test ! -e "$FENIKS_RATIO_ROOT"
test ! -e "$FENIKS_RATIO_ROOT.code.tar"
command -v sbatch >/dev/null
mkdir -p "$(dirname "$FENIKS_RATIO_ROOT")"

tar --exclude='__pycache__' --exclude='*.pyc' \
  -cf "$FENIKS_RATIO_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$FENIKS_RATIO_ROOT.code.tar" filters
DIGEST=$(sha256sum "$FENIKS_RATIO_ROOT.code.tar" | cut -d' ' -f1)
export FENIKS_RATIO_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/ratio-ladder-$DIGEST"
mkdir -p "$FENIKS_RATIO_CODE"
tar -xf "$FENIKS_RATIO_ROOT.code.tar" -C "$FENIKS_RATIO_CODE"
[[ -e "$FENIKS_RATIO_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_RATIO_CODE/Data"

cd "$FENIKS_RATIO_CODE"
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu \
  EUCLID_DSPS_REQUIRE_GPU=0 JAX_ENABLE_X64=true \
  python -m scripts.feniks_ratio_ladder prepare \
    --clean "$CLEAN" --root "$FENIKS_RATIO_ROOT" --config "$CONFIG"
printf '%s\n' "$FENIKS_RATIO_CODE" > "$FENIKS_RATIO_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FENIKS_RATIO_ROOT/CODE_SHA256"

mapfile -t RESOURCE < <(python - "$CONFIG" <<'PY'
import sys
import yaml

settings = yaml.safe_load(open(sys.argv[1]))
resources = settings["resources"]
for key in (
    "observation_hours",
    "bank_hours",
    "ratio_hours",
    "report_hours",
    "bank_concurrency",
    "ratio_concurrency",
):
    value = resources[key]
    assert isinstance(value, int) and value > 0
    print(value)
print(settings["reference_shards"])
PY
)
OBSERVATION_HOURS=${RESOURCE[0]}
BANK_HOURS=${RESOURCE[1]}
RATIO_HOURS=${RESOURCE[2]}
REPORT_HOURS=${RESOURCE[3]}
BANK_CONCURRENCY=${RESOURCE[4]}
RATIO_CONCURRENCY=${RESOURCE[5]}
REFERENCE_SHARDS=${RESOURCE[6]}

: > "$FENIKS_RATIO_ROOT/JOBS.env"
record() {
  printf 'export %s=%q\n' "$1" "$2" >> "$FENIKS_RATIO_ROOT/JOBS.env"
}

export FENIKS_RATIO_MODE=observation
RAW=$(sbatch --parsable --time="$OBSERVATION_HOURS:00:00" \
  --export=ALL --output="$FENIKS_RATIO_ROOT/logs/observation-%j.out" \
  --error="$FENIKS_RATIO_ROOT/logs/observation-%j.err" \
  scripts/feniks_ratio_ladder.slurm)
OBSERVATION_JOB=${RAW%%;*}
record OBSERVATION_JOB "$OBSERVATION_JOB"
record ALL_JOBS "$OBSERVATION_JOB"

export FENIKS_RATIO_MODE=bank
RAW=$(sbatch --parsable --time="$BANK_HOURS:00:00" \
  --array="0-${REFERENCE_SHARDS}%${BANK_CONCURRENCY}" --export=ALL \
  --output="$FENIKS_RATIO_ROOT/logs/bank-%A_%a.out" \
  --error="$FENIKS_RATIO_ROOT/logs/bank-%A_%a.err" \
  scripts/feniks_ratio_ladder.slurm)
BANK_JOB=${RAW%%;*}
record BANK_JOB "$BANK_JOB"
record ALL_JOBS "$OBSERVATION_JOB,$BANK_JOB"

export FENIKS_RATIO_MODE=ratio
RAW=$(sbatch --parsable --time="$RATIO_HOURS:00:00" --array="0-3%$RATIO_CONCURRENCY" \
  --dependency="afterok:$BANK_JOB" --export=ALL \
  --output="$FENIKS_RATIO_ROOT/logs/ratio-%A_%a.out" \
  --error="$FENIKS_RATIO_ROOT/logs/ratio-%A_%a.err" \
  scripts/feniks_ratio_ladder.slurm)
RATIO_JOB=${RAW%%;*}
record RATIO_JOB "$RATIO_JOB"
record ALL_JOBS "$OBSERVATION_JOB,$BANK_JOB,$RATIO_JOB"

export FENIKS_RATIO_MODE=report
RAW=$(sbatch --parsable --time="$REPORT_HOURS:00:00" \
  --dependency="afterok:$OBSERVATION_JOB:$RATIO_JOB" --export=ALL \
  --output="$FENIKS_RATIO_ROOT/logs/report-%j.out" \
  --error="$FENIKS_RATIO_ROOT/logs/report-%j.err" \
  scripts/feniks_ratio_ladder.slurm)
REPORT_JOB=${RAW%%;*}
record REPORT_JOB "$REPORT_JOB"
record ALL_JOBS "$OBSERVATION_JOB,$BANK_JOB,$RATIO_JOB,$REPORT_JOB"

echo "observation=$OBSERVATION_JOB bank=$BANK_JOB ratio=$RATIO_JOB report=$REPORT_JOB"
echo "Peak H100s: $((BANK_CONCURRENCY > RATIO_CONCURRENCY ? BANK_CONCURRENCY : RATIO_CONCURRENCY))"
echo "Allocation ceiling: $((OBSERVATION_HOURS + (REFERENCE_SHARDS + 1) * BANK_HOURS + 4 * RATIO_HOURS + REPORT_HOURS)) H100-hours, not expected runtime."
printf 'watch=bash scripts/watch_feniks_ratio_ladder.sh %q\n' "$FENIKS_RATIO_ROOT"
