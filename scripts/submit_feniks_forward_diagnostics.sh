#!/bin/bash
# Usage: bash scripts/submit_feniks_forward_diagnostics.sh COMPLETED_PARENT NEW_ROOT
set -Eeuo pipefail
PARENT=$(realpath "${1:?completed forward root}")
export FORWARD_ROOT=$(realpath -m "${2:?new diagnostic root}")
REPO=$(pwd -P)
CONFIG=$(realpath "${3:-configs/experiments/feniks_forward_diagnostics.yaml}")
test ! -e "$FORWARD_ROOT"
test ! -e "$FORWARD_ROOT.code.tar"
command -v sbatch >/dev/null
mkdir -p "$(dirname "$FORWARD_ROOT")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FORWARD_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$FORWARD_ROOT.code.tar" filters
DIGEST=$(sha256sum "$FORWARD_ROOT.code.tar" | cut -d' ' -f1)
export FORWARD_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/diagnostics-$DIGEST"
mkdir -p "$FORWARD_CODE"
tar -xf "$FORWARD_ROOT.code.tar" -C "$FORWARD_CODE"
[[ -e "$FORWARD_CODE/Data" ]] || ln -s "$REPO/Data" "$FORWARD_CODE/Data"
cd "$FORWARD_CODE"
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu EUCLID_DSPS_REQUIRE_GPU=0 JAX_ENABLE_X64=true \
  python -m scripts.feniks_forward_diagnostics prepare --root "$FORWARD_ROOT" --parent "$PARENT" --config "$CONFIG"
printf '%s\n' "$FORWARD_CODE" > "$FORWARD_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FORWARD_ROOT/CODE_SHA256"
mapfile -t HOURS < <(python - "$FORWARD_ROOT/diagnostics.yaml" <<'PY'
import sys,yaml
r=yaml.safe_load(open(sys.argv[1]))['resources']
for key in ('evaluation_hours','classifier_hours','posterior_hours','capacity_hours','summary_hours'):
    assert isinstance(r[key],int) and r[key]>0
    print(r[key])
PY
)
[[ ${#HOURS[@]} == 5 ]]
ALL=()
submit() {
  export FORWARD_MODE=$1
  local hours=$2 dep=${3:-} array=${4:-} raw opts=()
  [[ -z "$dep" ]] || opts+=(--dependency="afterok:$dep")
  [[ -z "$array" ]] || opts+=(--array="$array")
  raw=$(sbatch --parsable --time="$hours:00:00" --export=ALL "${opts[@]}" \
    --output="$FORWARD_ROOT/logs/$FORWARD_MODE-%A_%a.out" \
    --error="$FORWARD_ROOT/logs/$FORWARD_MODE-%A_%a.err" "$FORWARD_CODE/scripts/feniks_forward_diagnostics.slurm")
  JOB=${raw%%;*}
  ALL+=("$JOB")
  printf 'export %s_JOB=%q\n' "${FORWARD_MODE^^}" "$JOB" >> "$FORWARD_ROOT/JOBS.env"
  printf 'export ALL_JOBS=%q\n' "$(IFS=,; echo "${ALL[*]}")" >> "$FORWARD_ROOT/JOBS.env"
  echo "$FORWARD_MODE=$JOB"
}
submit baseline "${HOURS[0]}"
submit classifier "${HOURS[1]}"
submit posterior "${HOURS[2]}"
submit capacity "${HOURS[3]}" '' '0-1%2'
DEPENDENCIES=$(IFS=:; echo "${ALL[*]}")
submit summary "${HOURS[4]}" "$DEPENDENCIES"
echo "Peak: 5 H100; no new DSPS bank. Allocation ceiling: $((HOURS[0]+HOURS[1]+HOURS[2]+2*HOURS[3]+HOURS[4])) GPU-hours."
printf 'watch=bash scripts/watch_feniks_forward_diagnostics.sh %q\n' "$FORWARD_ROOT"
