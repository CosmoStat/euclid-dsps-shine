#!/bin/bash
# Usage: bash scripts/submit_feniks_population_inversion_audit.sh SOURCE NEW_ROOT [CONFIG]
set -Eeuo pipefail

SOURCE=$(realpath "${1:?completed ratio-convergence root}")
export FENIKS_POPULATION_INVERSION_ROOT=$(realpath -m "${2:?new audit root}")
CONFIG=$(realpath "${3:-configs/experiments/feniks_population_inversion_audit.yaml}")
REPO=$(pwd -P)

test ! -e "$FENIKS_POPULATION_INVERSION_ROOT"
test ! -e "$FENIKS_POPULATION_INVERSION_ROOT.code.tar"
command -v sbatch >/dev/null
mkdir -p "$(dirname "$FENIKS_POPULATION_INVERSION_ROOT")"

tar --exclude='__pycache__' --exclude='*.pyc' \
  -cf "$FENIKS_POPULATION_INVERSION_ROOT.code.tar" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$FENIKS_POPULATION_INVERSION_ROOT.code.tar" filters
DIGEST=$(sha256sum "$FENIKS_POPULATION_INVERSION_ROOT.code.tar" | cut -d' ' -f1)
export FENIKS_POPULATION_INVERSION_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/population-inversion-$DIGEST"
mkdir -p "$FENIKS_POPULATION_INVERSION_CODE"
tar -xf "$FENIKS_POPULATION_INVERSION_ROOT.code.tar" \
  -C "$FENIKS_POPULATION_INVERSION_CODE"
[[ -e "$FENIKS_POPULATION_INVERSION_CODE/Data" ]] || \
  ln -s "$REPO/Data" "$FENIKS_POPULATION_INVERSION_CODE/Data"

cd "$FENIKS_POPULATION_INVERSION_CODE"
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu \
  EUCLID_DSPS_REQUIRE_GPU=0 JAX_ENABLE_X64=true \
  python -m scripts.feniks_population_inversion_audit prepare \
    --source "$SOURCE" --root "$FENIKS_POPULATION_INVERSION_ROOT" \
    --config "$CONFIG"
printf '%s\n' "$FENIKS_POPULATION_INVERSION_CODE" \
  > "$FENIKS_POPULATION_INVERSION_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FENIKS_POPULATION_INVERSION_ROOT/CODE_SHA256"

mapfile -t RESOURCE < <(python - "$CONFIG" <<'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1]))["resources"]
for key in ("arm_hours", "report_hours", "arm_concurrency"):
    value = r[key]
    assert isinstance(value, int) and value > 0
    print(value)
PY
)
ARM_HOURS=${RESOURCE[0]}
REPORT_HOURS=${RESOURCE[1]}
ARM_CONCURRENCY=${RESOURCE[2]}

export FENIKS_POPULATION_INVERSION_MODE=run
RAW=$(sbatch --parsable --time="$ARM_HOURS:00:00" \
  --array="0-1%$ARM_CONCURRENCY" --export=ALL \
  --output="$FENIKS_POPULATION_INVERSION_ROOT/logs/arm-%A_%a.out" \
  --error="$FENIKS_POPULATION_INVERSION_ROOT/logs/arm-%A_%a.err" \
  scripts/feniks_population_inversion_audit.slurm)
ARM_JOB=${RAW%%;*}

export FENIKS_POPULATION_INVERSION_MODE=report
RAW=$(sbatch --parsable --time="$REPORT_HOURS:00:00" \
  --dependency="afterok:$ARM_JOB" --export=ALL \
  --output="$FENIKS_POPULATION_INVERSION_ROOT/logs/report-%j.out" \
  --error="$FENIKS_POPULATION_INVERSION_ROOT/logs/report-%j.err" \
  scripts/feniks_population_inversion_audit.slurm)
REPORT_JOB=${RAW%%;*}
ALL_JOBS="$ARM_JOB,$REPORT_JOB"
printf 'export ARM_JOB=%q\nexport REPORT_JOB=%q\nexport ALL_JOBS=%q\n' \
  "$ARM_JOB" "$REPORT_JOB" "$ALL_JOBS" \
  > "$FENIKS_POPULATION_INVERSION_ROOT/JOBS.env"

echo "arms=$ARM_JOB report=$REPORT_JOB"
echo 'Frozen classifiers and reused banks; no DSPS simulation or neural training.'
echo "Peak H100s: $ARM_CONCURRENCY; allocation ceiling: $((2 * ARM_HOURS + REPORT_HOURS)) H100-hours."
printf 'watch=bash scripts/watch_feniks_population_inversion_audit.sh %q\n' \
  "$FENIKS_POPULATION_INVERSION_ROOT"
