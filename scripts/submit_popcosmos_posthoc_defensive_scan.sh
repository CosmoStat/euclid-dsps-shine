#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SMC_ROOT="${SMC_ROOT:?Set SMC_ROOT to the completed floor-0.05 SMC run}"
BASELINE_REFRESH_OUT="${BASELINE_REFRESH_OUT:?Set BASELINE_REFRESH_OUT to the unimodal refresh}"
TEMPERATURE_SCAN_ROOT="${TEMPERATURE_SCAN_ROOT:?Set TEMPERATURE_SCAN_ROOT to the completed exact-temperature scan}"
TAIL_TEMPERATURES_CSV="${TAIL_TEMPERATURES_CSV:-1.25,1.5}"
TAIL_FRACTIONS_CSV="${TAIL_FRACTIONS_CSV:-0.02,0.05,0.1,0.2}"
PROBE_SAMPLES="${PROBE_SAMPLES:-2048}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
MIN_MEDIAN_ESS_FRACTION="${MIN_MEDIAN_ESS_FRACTION:-0.05}"
MAX_FRACTION_PARETO_K_GT_0P7="${MAX_FRACTION_PARETO_K_GT_0P7:-0.2}"
DEFENSIVE_SCAN_ROOT="${DEFENSIVE_SCAN_ROOT:-$BASELINE_REFRESH_OUT/defensive_proposal_scan_$(date +%Y%m%d_%H%M%S)}"

cd "$REPO_DIR"
mkdir -p outputs/logs
test -s "$BASELINE_REFRESH_OUT/encoder_refresh/checkpoints/best.eqx"
test -s "$BASELINE_REFRESH_OUT/moderate_k${PROBE_SAMPLES}_importance/importance_summary.json"
test ! -e "$DEFENSIVE_SCAN_ROOT"

temperature_count=$(python - "$TAIL_TEMPERATURES_CSV" <<'PY'
import math
import sys

values = [float(item.strip()) for item in sys.argv[1].split(",") if item.strip()]
if not values or any(not math.isfinite(value) or value <= 1.0 for value in values):
    raise SystemExit("tail temperatures must be a non-empty CSV greater than one")
if len(set(values)) != len(values):
    raise SystemExit("tail temperatures must be unique")
print(len(values))
PY
)
python - "$TAIL_FRACTIONS_CSV" <<'PY'
import math
import sys

values = [float(item.strip()) for item in sys.argv[1].split(",") if item.strip()]
if not values or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in values):
    raise SystemExit("tail fractions must be a non-empty CSV within (0, 1)")
if len(set(values)) != len(values):
    raise SystemExit("tail fractions must be unique")
PY
python - "$BASELINE_REFRESH_OUT" "$TEMPERATURE_SCAN_ROOT" \
  "$TAIL_TEMPERATURES_CSV" "$PROBE_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

baseline_root = Path(sys.argv[1])
temperature_root = Path(sys.argv[2])
temperatures = [float(item.strip()) for item in sys.argv[3].split(",") if item.strip()]
k = int(sys.argv[4])
base = baseline_root / f"moderate_k{k}_proposal"
base_summary = json.loads((base / "inference_summary.json").read_text())
if float(base_summary["posterior_base_temperature"]) != 1.0:
    raise SystemExit("baseline proposal is not the unit-temperature bank")
base_rows = np.load(base / "inference_indices.npy")
if np.unique(base_rows).size != base_rows.size:
    raise SystemExit("baseline proposal cohort contains duplicates")
for temperature in temperatures:
    slug = f"{temperature:.8g}".replace(".", "p")
    tail = temperature_root / f"temperature_{slug}" / f"proposal_k{k}"
    summary = json.loads((tail / "inference_summary.json").read_text())
    if float(summary["posterior_base_temperature"]) != temperature:
        raise SystemExit(f"tail proposal temperature mismatch: {tail}")
    tail_rows = np.load(tail / "inference_indices.npy")
    if np.unique(tail_rows).size != tail_rows.size:
        raise SystemExit(f"tail proposal cohort contains duplicates: {tail}")
    if tail_rows.size != base_rows.size or not np.array_equal(
        np.sort(tail_rows), np.sort(base_rows)
    ):
        raise SystemExit(f"tail proposal cohort differs from baseline: {tail}")
print(
    "[defensive-proposal-submit] preflight PASS "
    f"objects={base_rows.size} temperatures={temperatures} K={k}"
)
PY
if ((ARRAY_CONCURRENCY <= 0)); then
  echo "[defensive-proposal-submit][error] ARRAY_CONCURRENCY must be positive" >&2
  exit 2
fi
mkdir -p "$DEFENSIVE_SCAN_ROOT"

export REPO_DIR MINICONDA_PATH CONDA_ENV SMC_ROOT BASELINE_REFRESH_OUT
export TEMPERATURE_SCAN_ROOT DEFENSIVE_SCAN_ROOT TAIL_TEMPERATURES_CSV
export TAIL_FRACTIONS_CSV PROBE_SAMPLES ARRAY_CONCURRENCY
export MIN_MEDIAN_ESS_FRACTION MAX_FRACTION_PARETO_K_GT_0P7

array_spec="0-$((temperature_count - 1))%${ARRAY_CONCURRENCY}"
scan_raw=$(sbatch --parsable \
  --array="$array_spec" \
  --export=ALL \
  scripts/popcosmos_posthoc_defensive_scan_h100.slurm)
DEFENSIVE_SCAN_JOB="${scan_raw%%;*}"
final_raw=$(sbatch --parsable \
  --dependency="afterok:${DEFENSIVE_SCAN_JOB}" \
  --export=ALL \
  scripts/popcosmos_posthoc_defensive_finalize.slurm)
DEFENSIVE_FINALIZER_JOB="${final_raw%%;*}"

env_file=outputs/logs/popcosmos_posthoc_defensive_scan_latest.env
printf 'export DEFENSIVE_SCAN_JOB=%q\nexport DEFENSIVE_FINALIZER_JOB=%q\nexport DEFENSIVE_SCAN_ROOT=%q\nexport BASELINE_REFRESH_OUT=%q\nexport TEMPERATURE_SCAN_ROOT=%q\nexport SMC_ROOT=%q\nexport TAIL_TEMPERATURES_CSV=%q\nexport TAIL_FRACTIONS_CSV=%q\nexport PROBE_SAMPLES=%q\n' \
  "$DEFENSIVE_SCAN_JOB" "$DEFENSIVE_FINALIZER_JOB" "$DEFENSIVE_SCAN_ROOT" \
  "$BASELINE_REFRESH_OUT" "$TEMPERATURE_SCAN_ROOT" "$SMC_ROOT" \
  "$TAIL_TEMPERATURES_CSV" "$TAIL_FRACTIONS_CSV" "$PROBE_SAMPLES" \
  > "$env_file"

echo "defensive_scan_job=$DEFENSIVE_SCAN_JOB"
echo "defensive_finalizer_job=$DEFENSIVE_FINALIZER_JOB"
echo "defensive_scan_root=$DEFENSIVE_SCAN_ROOT"
echo "tail_temperatures=$TAIL_TEMPERATURES_CSV"
echo "tail_fractions=$TAIL_FRACTIONS_CSV"
echo "array_tasks=$temperature_count concurrency=$ARRAY_CONCURRENCY"
echo "monitor: squeue -j $DEFENSIVE_SCAN_JOB,$DEFENSIVE_FINALIZER_JOB"
echo "logs: outputs/logs/cosmos_qdef-${DEFENSIVE_SCAN_JOB}_<taskid>.out"
echo "latest_env=$env_file"
