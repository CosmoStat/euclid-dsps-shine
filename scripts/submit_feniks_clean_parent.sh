#!/bin/bash
# Usage: bash scripts/submit_feniks_clean_parent.sh CAPACITY NEW_ROOT [CONFIG]
set -Eeuo pipefail
CAPACITY=$(realpath "${1:?completed weighted truth capacity root}")
export FENIKS_CLEAN_ROOT=$(realpath -m "${2:?new experiment root}")
CONFIG=$(realpath "${3:-configs/experiments/feniks_clean_parent.yaml}")
REPO=$(pwd -P)
test ! -e "$FENIKS_CLEAN_ROOT"
test ! -e "$FENIKS_CLEAN_ROOT.code.tar"
command -v sbatch >/dev/null
mkdir -p "$(dirname "$FENIKS_CLEAN_ROOT")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_CLEAN_ROOT.code.tar" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$FENIKS_CLEAN_ROOT.code.tar" filters
DIGEST=$(sha256sum "$FENIKS_CLEAN_ROOT.code.tar" | cut -d' ' -f1)
export FENIKS_CLEAN_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/clean-parent-$DIGEST"
mkdir -p "$FENIKS_CLEAN_CODE"
tar -xf "$FENIKS_CLEAN_ROOT.code.tar" -C "$FENIKS_CLEAN_CODE"
[[ -e "$FENIKS_CLEAN_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_CLEAN_CODE/Data"
cd "$FENIKS_CLEAN_CODE"
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu EUCLID_DSPS_REQUIRE_GPU=0 \
  JAX_ENABLE_X64=true python -m scripts.feniks_clean_parent prepare \
  --capacity "$CAPACITY" --root "$FENIKS_CLEAN_ROOT" --config "$CONFIG"
printf '%s\n' "$FENIKS_CLEAN_CODE" > "$FENIKS_CLEAN_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FENIKS_CLEAN_ROOT/CODE_SHA256"
mapfile -t RESOURCE < <(python - "$CONFIG" <<'PY'
import sys
import yaml
r = yaml.safe_load(open(sys.argv[1]))['resources']
for key in ('contract_hours', 'fit_hours', 'closure_hours', 'report_hours', 'concurrency'):
    value = r[key]
    assert isinstance(value, int) and value > 0
    print(value)
PY
)
CONTRACT_HOURS=${RESOURCE[0]}
FIT_HOURS=${RESOURCE[1]}
CLOSURE_HOURS=${RESOURCE[2]}
REPORT_HOURS=${RESOURCE[3]}
CONCURRENCY=${RESOURCE[4]}
: > "$FENIKS_CLEAN_ROOT/JOBS.env"
record() {
  printf 'export %s=%q\n' "$1" "$2" >> "$FENIKS_CLEAN_ROOT/JOBS.env"
}
export FENIKS_CLEAN_MODE=contracts
RAW=$(sbatch --parsable --time="$CONTRACT_HOURS:00:00" \
  --export=ALL --output="$FENIKS_CLEAN_ROOT/logs/contracts-%j.out" \
  --error="$FENIKS_CLEAN_ROOT/logs/contracts-%j.err" scripts/feniks_clean_parent.slurm)
CONTRACT_JOB=${RAW%%;*}
record CONTRACT_JOB "$CONTRACT_JOB"
record ALL_JOBS "$CONTRACT_JOB"
export FENIKS_CLEAN_MODE=fit
RAW=$(sbatch --parsable --time="$FIT_HOURS:00:00" --array="0-3%$CONCURRENCY" \
  --dependency="afterok:$CONTRACT_JOB" \
  --export=ALL --output="$FENIKS_CLEAN_ROOT/logs/fit-%A_%a.out" \
  --error="$FENIKS_CLEAN_ROOT/logs/fit-%A_%a.err" scripts/feniks_clean_parent.slurm)
FIT_JOB=${RAW%%;*}
record FIT_JOB "$FIT_JOB"
record ALL_JOBS "$CONTRACT_JOB,$FIT_JOB"
export FENIKS_CLEAN_MODE=closure
RAW=$(sbatch --parsable --time="$CLOSURE_HOURS:00:00" --dependency="afterok:$FIT_JOB" \
  --export=ALL --output="$FENIKS_CLEAN_ROOT/logs/closure-%j.out" \
  --error="$FENIKS_CLEAN_ROOT/logs/closure-%j.err" scripts/feniks_clean_parent.slurm)
CLOSURE_JOB=${RAW%%;*}
record CLOSURE_JOB "$CLOSURE_JOB"
record ALL_JOBS "$CONTRACT_JOB,$FIT_JOB,$CLOSURE_JOB"
export FENIKS_CLEAN_MODE=report
RAW=$(sbatch --parsable --time="$REPORT_HOURS:00:00" --dependency="afterok:$CLOSURE_JOB" \
  --export=ALL --output="$FENIKS_CLEAN_ROOT/logs/report-%j.out" \
  --error="$FENIKS_CLEAN_ROOT/logs/report-%j.err" scripts/feniks_clean_parent.slurm)
REPORT_JOB=${RAW%%;*}
record REPORT_JOB "$REPORT_JOB"
record ALL_JOBS "$CONTRACT_JOB,$FIT_JOB,$CLOSURE_JOB,$REPORT_JOB"
echo "contracts=$CONTRACT_JOB fit=$FIT_JOB closure=$CLOSURE_JOB report=$REPORT_JOB"
echo "Allocation ceiling: $((CONTRACT_HOURS + 4 * FIT_HOURS + CLOSURE_HOURS + REPORT_HOURS)) H100-hours, not a runtime estimate."
printf 'watch=bash scripts/watch_feniks_clean_parent.sh %q\n' "$FENIKS_CLEAN_ROOT"
