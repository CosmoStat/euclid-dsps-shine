#!/bin/bash
# One forward-only campaign: submit_feniks_forward_population.sh SOURCE ROOT [CONFIG] [BASELINE]
# SOURCE is the previous benchmark runtime/r29, not its learned prior.
set -Eeuo pipefail
SOURCE=$(realpath "${1:?source runtime/r29 required}")
export FORWARD_ROOT=$(realpath -m "${2:?new output root required}")
CONFIG=$(realpath "${3:-configs/experiments/feniks_forward_population_r29.yaml}")
BASELINE=$(realpath "${4:-$SOURCE/../..}")
REPO=$(pwd -P)
test ! -e "$FORWARD_ROOT"
test -s "$SOURCE/MANIFEST.json"
test -d "$REPO/filters"
command -v sbatch >/dev/null
ARCHIVE="$FORWARD_ROOT.code.tar"
test ! -e "$ARCHIVE"
mkdir -p "$(dirname "$FORWARD_ROOT")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export FORWARD_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/forward-$DIGEST"
mkdir -p "$FORWARD_CODE"
tar -xf "$ARCHIVE" -C "$FORWARD_CODE"
if [[ ! -e "$FORWARD_CODE/Data" && ! -L "$FORWARD_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$FORWARD_CODE/Data"
fi
cd "$FORWARD_CODE"
CUT=$(python -c 'import yaml,sys; print(int(yaml.safe_load(open(sys.argv[1]))["cut"]))' "$CONFIG")
TRUTH="$BASELINE/cohorts/r$CUT/selection"
test -s "$TRUTH/true_parent.parquet"
test -s "$TRUTH/true_selected.parquet"
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu EUCLID_DSPS_REQUIRE_GPU=0 JAX_ENABLE_X64=true \
  python -m scripts.feniks_forward_population prepare --source "$SOURCE" \
  --root "$FORWARD_ROOT" --config "$CONFIG" --baseline "$BASELINE" \
  --truth-parent "$TRUTH/true_parent.parquet" --truth-selected "$TRUTH/true_selected.parquet"
printf '%s\n' "$FORWARD_CODE" > "$FORWARD_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FORWARD_ROOT/CODE_SHA256"
mapfile -t SETTINGS < <(python - "$FORWARD_ROOT/experiment.yaml" <<'PY'
import sys, yaml
c=yaml.safe_load(open(sys.argv[1])); r=c['resources']
assert r['gpus_per_task']==1, 'This implementation uses one GPU per process'
for v in (c['reference_shards'],c['posterior_shards'],r['reference_concurrency'],
          r['posterior_bank_concurrency'],r['bank_hours'],r['classifier_hours'],
          r['posterior_hours'],r['report_hours']):
    assert isinstance(v,int) and v>0
    print(v)
PY
)
[[ ${#SETTINGS[@]} == 8 ]]
read -r REF POST RC PC BH CH QH RH <<< "${SETTINGS[*]}"
printf 'Peak H100s: %s; allocated-time ceiling: %s H100-hours (not an expected runtime).\n' \
  "$(( RC > PC ? RC : PC ))" "$(( (REF+POST+1)*BH+CH+QH+RH+1 ))"
ALL=()
submit() {
  local mode=$1 hours=$2 dependency=$3 array=${4:-}
  export FORWARD_MODE=$mode
  local opts=()
  [[ -z "$dependency" ]] || opts+=(--dependency="afterok:$dependency")
  [[ -z "$array" ]] || opts+=(--array="$array")
  local raw
  raw=$(sbatch --parsable --time="$hours:00:00" --export=ALL "${opts[@]}" \
    --output="$FORWARD_ROOT/logs/$mode-%A_%a.out" \
    --error="$FORWARD_ROOT/logs/$mode-%A_%a.err" "$FORWARD_CODE/scripts/feniks_forward_population.slurm")
  JOB=${raw%%;*}
  ALL+=("$JOB")
  printf 'export %s_JOB=%q\n' "${mode^^}" "$JOB" | tr '-' '_' >> "$FORWARD_ROOT/JOBS.env"
  printf 'export ALL_JOBS=%q\n' "$(IFS=,; echo "${ALL[*]}")" >> "$FORWARD_ROOT/JOBS.env"
  printf '%s=%s\n' "$mode" "$JOB"
}
submit preflight 1 ''
submit reference "$BH" "$JOB" "0-$((REF-1))%$RC"
submit population "$CH" "$JOB"
submit posterior-bank "$BH" "$JOB" "0-$POST%$PC"
submit train "$QH" "$JOB"
submit report "$RH" "$JOB"
printf 'export ALL_JOBS=%q\n' "$(IFS=,; echo "${ALL[*]}")" >> "$FORWARD_ROOT/JOBS.env"
printf 'watch=bash scripts/watch_feniks_forward_population.sh %q\n' "$FORWARD_ROOT"
