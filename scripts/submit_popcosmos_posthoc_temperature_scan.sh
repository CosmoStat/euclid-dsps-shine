#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SMC_ROOT="${SMC_ROOT:?Source the completed SMC refresh env file first}"
SMC_REFRESH_OUT="${SMC_REFRESH_OUT:?Source the completed SMC refresh env file first}"
PROPOSAL_TEMPERATURES_CSV="${PROPOSAL_TEMPERATURES_CSV:-1.1,1.25,1.5,2.0}"
PROBE_SAMPLES="${PROBE_SAMPLES:-2048}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-4}"
MIN_MEDIAN_ESS_FRACTION="${MIN_MEDIAN_ESS_FRACTION:-0.05}"
MAX_FRACTION_PARETO_K_GT_0P7="${MAX_FRACTION_PARETO_K_GT_0P7:-0.2}"
TEMPERATURE_SCAN_ROOT="${TEMPERATURE_SCAN_ROOT:-$SMC_REFRESH_OUT/proposal_temperature_scan_$(date +%Y%m%d_%H%M%S)}"

cd "$REPO_DIR"
mkdir -p outputs/logs
test -s "$SMC_REFRESH_OUT/encoder_refresh/refresh_summary.json"
test -s "$SMC_REFRESH_OUT/moderate_k${PROBE_SAMPLES}_importance/importance_summary.json"
test ! -e "$TEMPERATURE_SCAN_ROOT"

temperature_count=$(python - "$PROPOSAL_TEMPERATURES_CSV" <<'PY'
import math
import sys

values = [float(item.strip()) for item in sys.argv[1].split(",") if item.strip()]
if not values or any(not math.isfinite(item) or item <= 0.0 for item in values):
    raise SystemExit("temperatures must be a non-empty CSV of positive finite values")
if len(set(values)) != len(values) or 1.0 in values:
    raise SystemExit("temperatures must be unique and exclude baseline 1.0")
print(len(values))
PY
)
if ((ARRAY_CONCURRENCY <= 0)); then
  echo "[proposal-temperature-submit][error] ARRAY_CONCURRENCY must be positive" >&2
  exit 2
fi
mkdir -p "$TEMPERATURE_SCAN_ROOT"

export REPO_DIR MINICONDA_PATH CONDA_ENV SMC_ROOT SMC_REFRESH_OUT
export PROPOSAL_TEMPERATURES_CSV PROBE_SAMPLES ARRAY_CONCURRENCY
export MIN_MEDIAN_ESS_FRACTION MAX_FRACTION_PARETO_K_GT_0P7
export TEMPERATURE_SCAN_ROOT

array_spec="0-$((temperature_count - 1))%${ARRAY_CONCURRENCY}"
scan_raw=$(sbatch --parsable \
  --array="$array_spec" \
  --export=ALL \
  scripts/popcosmos_posthoc_temperature_scan_h100.slurm)
TEMPERATURE_SCAN_JOB="${scan_raw%%;*}"
final_raw=$(sbatch --parsable \
  --dependency="afterok:${TEMPERATURE_SCAN_JOB}" \
  --export=ALL \
  scripts/popcosmos_posthoc_temperature_finalize.slurm)
TEMPERATURE_FINALIZER_JOB="${final_raw%%;*}"

env_file=outputs/logs/popcosmos_posthoc_temperature_scan_latest.env
printf 'export TEMPERATURE_SCAN_JOB=%q\nexport TEMPERATURE_FINALIZER_JOB=%q\nexport TEMPERATURE_SCAN_ROOT=%q\nexport SMC_REFRESH_OUT=%q\nexport SMC_ROOT=%q\nexport PROPOSAL_TEMPERATURES_CSV=%q\nexport PROBE_SAMPLES=%q\n' \
  "$TEMPERATURE_SCAN_JOB" "$TEMPERATURE_FINALIZER_JOB" "$TEMPERATURE_SCAN_ROOT" \
  "$SMC_REFRESH_OUT" "$SMC_ROOT" "$PROPOSAL_TEMPERATURES_CSV" "$PROBE_SAMPLES" \
  > "$env_file"

echo "temperature_scan_job=$TEMPERATURE_SCAN_JOB"
echo "temperature_finalizer_job=$TEMPERATURE_FINALIZER_JOB"
echo "temperature_scan_root=$TEMPERATURE_SCAN_ROOT"
echo "temperatures=$PROPOSAL_TEMPERATURES_CSV baseline=1.0"
echo "array_tasks=$temperature_count concurrency=$ARRAY_CONCURRENCY"
echo "monitor: squeue -j $TEMPERATURE_SCAN_JOB,$TEMPERATURE_FINALIZER_JOB"
echo "logs: outputs/logs/cosmos_qtemp-${TEMPERATURE_SCAN_JOB}_<taskid>.out"
echo "latest_env=$env_file"
