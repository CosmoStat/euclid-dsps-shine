#!/bin/bash
# COHERENT_DATASET NEW_ROOT [CONFIG], or --resume EXISTING_ROOT
set -Eeuo pipefail
REPO=$(pwd -P)
command -v sbatch >/dev/null
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
if [[ ${1:-} == --resume ]]; then
  export FENIKS_CI_ROOT=$(realpath "${2:?existing root}")
  if [[ -f "$FENIKS_CI_ROOT/JOBS.env" ]]; then
    source "$FENIKS_CI_ROOT/JOBS.env"
    ACTIVE=$(squeue -h -j "$ALL_JOBS" -o '%i' 2>/dev/null || true)
    [[ -z "$ACTIVE" ]] || { echo "Jobs still active: $ACTIVE" >&2; exit 1; }
  fi
  export FENIKS_CI_CODE=$(<"$FENIKS_CI_ROOT/CODE_DIR")
else
  SOURCE=$(realpath "${1:?coherent dataset root}")
  export FENIKS_CI_ROOT=$(realpath -m "${2:?new root}")
  CONFIG=$(realpath "${3:-configs/experiments/feniks_coherent_inference.yaml}")
  test ! -e "$FENIKS_CI_ROOT"
  test ! -e "$FENIKS_CI_ROOT.code.tar"
  mkdir -p "$(dirname "$FENIKS_CI_ROOT")"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_CI_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
  DIGEST=$(sha256sum "$FENIKS_CI_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_CI_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/coherent-inference-$DIGEST"
  mkdir -p "$FENIKS_CI_CODE"
  tar -xf "$FENIKS_CI_ROOT.code.tar" -C "$FENIKS_CI_CODE"
  [[ -e "$FENIKS_CI_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_CI_CODE/Data"
  cd "$FENIKS_CI_CODE"
  python -m scripts.feniks_coherent_inference prepare --source "$SOURCE" \
    --root "$FENIKS_CI_ROOT" --config "$CONFIG"
  printf '%s\n' "$FENIKS_CI_CODE" > "$FENIKS_CI_ROOT/CODE_DIR"
  printf '%s\n' "$DIGEST" > "$FENIKS_CI_ROOT/CODE_SHA256"
fi
cd "$FENIKS_CI_CODE"
mapfile -t INFO < <(python - "$FENIKS_CI_ROOT" <<'PY'
import sys
from pathlib import Path
from scripts.feniks_coherent_inference import settings, complete
r=Path(sys.argv[1]); _, c, d=settings(r)
for k in ('reference_minutes','bank_minutes','oracle_minutes','population_minutes','posterior_minutes','report_minutes','bank_concurrency','cpu_threads'):
    print(c['resources'][k])
for k in ('reference','oracle','population','posterior','report'): print(int(not complete(r/k,d)))
print(','.join(str(i) for i in range(c['bank']['shards']) if not complete(r/'banks'/f'shard_{i:03d}',d)))
PY
)
[[ ${#INFO[@]} == 14 ]] || { echo 'Invalid prepared state' >&2; exit 1; }
ALL_JOBS= REFERENCE_JOB= BANK_JOB= ORACLE_JOB= POPULATION_JOB= POSTERIOR_JOB= REPORT_JOB=
ATTEMPT=$(date +%Y%m%d_%H%M%S)
record() {
  for key in ALL_JOBS REFERENCE_JOB BANK_JOB ORACLE_JOB POPULATION_JOB POSTERIOR_JOB REPORT_JOB; do
    printf 'export %s=%q\n' "$key" "${!key}"
  done > "$FENIKS_CI_ROOT/JOBS_$ATTEMPT.env"
  cp "$FENIKS_CI_ROOT/JOBS_$ATTEMPT.env" "$FENIKS_CI_ROOT/JOBS.env"
}
submit() {
  local mode=$1 gpu=$2 minutes=$3 dependency=$4
  shift 4
  local resources=(--account="${FENIKS_CPU_ACCOUNT:-jrx@cpu}" --partition="${FENIKS_CPU_PARTITION:-cpu_p1}")
  [[ $gpu == 0 ]] || resources=(--account="${FENIKS_GPU_ACCOUNT:-jrx@h100}" --partition="${FENIKS_GPU_PARTITION:-gpu_p6}" --constraint=h100 --gres=gpu:1)
  local after=()
  [[ -z "$dependency" ]] || after=(--dependency="$dependency")
  local raw
  raw=$(sbatch --parsable "${resources[@]}" --job-name="feniks_ci_$mode" \
    --nodes=1 --ntasks=1 --cpus-per-task="${INFO[7]}" --hint=nomultithread \
    --time="$minutes" --kill-on-invalid-dep=yes "${after[@]}" \
    --export="ALL,FENIKS_CI_MODE=$mode,FENIKS_CI_GPU=$gpu" \
    --output="$FENIKS_CI_ROOT/logs/$mode-$ATTEMPT-%A_%a.out" \
    --error="$FENIKS_CI_ROOT/logs/$mode-$ATTEMPT-%A_%a.err" \
    "$@" scripts/feniks_coherent_inference.slurm)
  LAST_JOB=${raw%%;*}
  [[ $LAST_JOB =~ ^[0-9]+$ ]] || { echo "Invalid job ID: $raw" >&2; exit 1; }
  ALL_JOBS="${ALL_JOBS:+$ALL_JOBS,}$LAST_JOB"
}
dep() { [[ -z $1 ]] || printf 'afterok:%s' "$1"; }
if [[ ${INFO[8]} == 1 ]]; then
  submit reference 0 "${INFO[0]}" ''; REFERENCE_JOB=$LAST_JOB; record
fi
if [[ ${INFO[9]} == 1 ]]; then
  submit oracle 1 "${INFO[2]}" "$(dep "$REFERENCE_JOB")"; ORACLE_JOB=$LAST_JOB; record
fi
if [[ -n ${INFO[13]} ]]; then
  submit bank 1 "${INFO[1]}" "$(dep "$REFERENCE_JOB")" --array="${INFO[13]}%${INFO[6]}"
  BANK_JOB=$LAST_JOB; record
fi
if [[ ${INFO[10]} == 1 ]]; then
  submit population 1 "${INFO[3]}" "$(dep "${BANK_JOB:-$REFERENCE_JOB}")"; POPULATION_JOB=$LAST_JOB; record
fi
if [[ ${INFO[11]} == 1 ]]; then
  submit posterior 1 "${INFO[4]}" "$(dep "${POPULATION_JOB:-${BANK_JOB:-$REFERENCE_JOB}}")"; POSTERIOR_JOB=$LAST_JOB; record
fi
if [[ ${INFO[12]} == 1 ]]; then
  AFTER=''
  [[ -z "$ALL_JOBS" ]] || AFTER="afterany:${ALL_JOBS//,/:}"
  submit report 0 "${INFO[5]}" "$AFTER"; REPORT_JOB=$LAST_JOB; record
fi
echo "reference=$REFERENCE_JOB oracle=$ORACLE_JOB banks=$BANK_JOB population=$POPULATION_JOB posterior=$POSTERIOR_JOB report=$REPORT_JOB"
echo "Peak H100s <= $((${INFO[6]}+1)); stage time limits are ceilings, not runtime estimates."
printf 'watch=bash scripts/watch_feniks_coherent_inference.sh %q\n' "$FENIKS_CI_ROOT"
