#!/bin/bash
#SBATCH --job-name=cosmos_qtemp
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SMC_ROOT="${SMC_ROOT:?Set SMC_ROOT to the completed selected-likelihood SMC run}"
SMC_REFRESH_OUT="${SMC_REFRESH_OUT:?Set SMC_REFRESH_OUT to the completed refresh workflow}"
TEMPERATURE_SCAN_ROOT="${TEMPERATURE_SCAN_ROOT:?Set TEMPERATURE_SCAN_ROOT}"
PROPOSAL_TEMPERATURES_CSV="${PROPOSAL_TEMPERATURES_CSV:?Set PROPOSAL_TEMPERATURES_CSV}"
PROBE_SAMPLES="${PROBE_SAMPLES:-2048}"
MIN_MEDIAN_ESS_FRACTION="${MIN_MEDIAN_ESS_FRACTION:-0.05}"
MAX_FRACTION_PARETO_K_GT_0P7="${MAX_FRACTION_PARETO_K_GT_0P7:-0.2}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export MPLBACKEND=Agg

cd "$REPO_DIR"
mkdir -p outputs/logs
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

IFS=',' read -r -a TEMPERATURES <<< "$PROPOSAL_TEMPERATURES_CSV"
task_id="${SLURM_ARRAY_TASK_ID:?This script must run as an array job}"
if ((task_id < 0 || task_id >= ${#TEMPERATURES[@]})); then
  echo "[proposal-temperature][error] task $task_id outside ${#TEMPERATURES[@]} temperatures" >&2
  exit 2
fi
temperature="${TEMPERATURES[$task_id]}"
temperature_slug=$(python - "$temperature" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value <= 0.0:
    raise SystemExit("proposal temperature must be finite and positive")
print(f"{value:.8g}".replace(".", "p"))
PY
)

SELECTION="$SMC_ROOT/pilot_selection/selection_summary.json"
REFRESH_SUMMARY="$SMC_REFRESH_OUT/encoder_refresh/refresh_summary.json"
test -s "$SELECTION"
test -s "$REFRESH_SUMMARY"
variant=$(python - "$SELECTION" "$REFRESH_SUMMARY" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1]))
refresh = json.load(open(sys.argv[2]))
if selection.get("selection_status") != "PASS":
    raise SystemExit("SMC likelihood selection did not pass")
if refresh.get("refresh_gate") != "PASS" or not refresh.get("prior_frozen_exactly"):
    raise SystemExit("encoder refresh or frozen-prior contract did not pass")
print(selection["selected_variant"])
PY
)
case "$variant" in
  floor_0p00) CONFIG=configs/experiments/popcosmos_native15d_rws.yaml ;;
  floor_0p02) CONFIG=configs/experiments/popcosmos_native15d_rws_floor02.yaml ;;
  floor_0p05) CONFIG=configs/experiments/popcosmos_native15d_rws_floor05.yaml ;;
  *) echo "[proposal-temperature][error] unsupported selected variant: $variant" >&2; exit 2 ;;
esac

DATASET=Data/cosmos2020/prepared/farmer_a24_full.parquet
CHECKPOINT="$SMC_REFRESH_OUT/encoder_refresh/checkpoints/best.eqx"
FEATURE_STATS="$SMC_REFRESH_OUT/encoder_refresh/feature_stats.json"
ROW_INDICES="$SMC_ROOT/cohorts/proposal_probe_indices.npy"
CANDIDATE="$TEMPERATURE_SCAN_ROOT/temperature_${temperature_slug}"
PROPOSAL="$CANDIDATE/proposal_k${PROBE_SAMPLES}"
CORRECTION="$CANDIDATE/importance_k${PROBE_SAMPLES}"
test -s "$CHECKPOINT"
test -s "$FEATURE_STATS"
test -s "$ROW_INDICES"
test ! -e "$CANDIDATE"
mkdir -p "$CANDIDATE"
PROBE_LIMIT=$(python - "$ROW_INDICES" <<'PY'
import numpy as np
import sys

print(len(np.load(sys.argv[1])))
PY
)

echo "[proposal-temperature] task=$task_id temperature=$temperature objects=$PROBE_LIMIT samples=$PROBE_SAMPLES"
echo "[proposal-temperature] selected_variant=$variant cohort=$ROW_INDICES out=$CANDIDATE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-cosmos \
  --runtime gpu \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" \
  --row-indices-file "$ROW_INDICES" \
  --selection-mode sequential \
  --limit "$PROBE_LIMIT" \
  --batch-size 128 \
  --jax-batch-size 64 \
  --posterior-samples "$PROBE_SAMPLES" \
  --posterior-base-temperature "$temperature" \
  --prior-samples 256 \
  --decoder-sample-chunk-size 1 \
  --prior-predictive-batch-size 64 \
  --out "$PROPOSAL" \
  --shard-outputs \
  --no-posterior-predictive \
  --no-residual-samples
python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-inference --out "$PROPOSAL" --limit "$PROBE_LIMIT"

python scripts/importance_correct_posterior.py \
  --posterior "$PROPOSAL" \
  --out "$CORRECTION" \
  --config "$CONFIG" \
  --target-checkpoint "$CHECKPOINT" \
  --resample-count 256 \
  --seed 260817 \
  --min-median-ess-fraction "$MIN_MEDIAN_ESS_FRACTION" \
  --max-fraction-pareto-k-gt-0p7 "$MAX_FRACTION_PARETO_K_GT_0P7"

python - "$temperature" "$PROPOSAL/inference_summary.json" \
  "$CORRECTION/importance_summary.json" "$CANDIDATE" <<'PY'
import json
import sys
from pathlib import Path

temperature, inference_path, importance_path, candidate = sys.argv[1:]
inference = json.load(open(inference_path))
importance = json.load(open(importance_path))
expected = float(temperature)
actual = float(inference["posterior_base_temperature"])
if actual != expected:
    raise SystemExit(f"recorded proposal temperature mismatch: {actual} != {expected}")
payload = {
    "status": "complete",
    "posterior_base_temperature": actual,
    "n_objects": importance["n_objects"],
    "n_joint_draws": importance["n_joint_draws"],
    "ordinary_importance_support_gate": importance["support_gate"]["status"],
    "median_raw_ess_fraction": importance["median_raw_ess_fraction"],
    "median_psis_ess_fraction": importance["median_psis_ess_fraction"],
    "fraction_pareto_k_gt_0p7": importance["fraction_pareto_k_gt_0p7"],
    "fraction_pareto_k_gt_1": importance["fraction_pareto_k_gt_1"],
    "density_contract": "samples and saved logq use the same exact tempered encoder proposal",
    "cohort_role": "proposal-temperature tuning; spectroscopy unused",
    "ready_for_empirical_bayes": False,
}
candidate = Path(candidate)
(candidate / "candidate_summary.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
(candidate / "DONE").touch()
print(json.dumps(payload, indent=2))
PY
