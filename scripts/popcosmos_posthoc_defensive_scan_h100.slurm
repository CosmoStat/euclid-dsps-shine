#!/bin/bash
#SBATCH --job-name=cosmos_qdef
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SMC_ROOT="${SMC_ROOT:?Set SMC_ROOT}"
BASELINE_REFRESH_OUT="${BASELINE_REFRESH_OUT:?Set BASELINE_REFRESH_OUT}"
TEMPERATURE_SCAN_ROOT="${TEMPERATURE_SCAN_ROOT:?Set TEMPERATURE_SCAN_ROOT}"
DEFENSIVE_SCAN_ROOT="${DEFENSIVE_SCAN_ROOT:?Set DEFENSIVE_SCAN_ROOT}"
TAIL_TEMPERATURES_CSV="${TAIL_TEMPERATURES_CSV:?Set TAIL_TEMPERATURES_CSV}"
TAIL_FRACTIONS_CSV="${TAIL_FRACTIONS_CSV:?Set TAIL_FRACTIONS_CSV}"
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

IFS=',' read -r -a TEMPERATURES <<< "$TAIL_TEMPERATURES_CSV"
task_id="${SLURM_ARRAY_TASK_ID:?This script must run as an array job}"
if ((task_id < 0 || task_id >= ${#TEMPERATURES[@]})); then
  echo "[defensive-proposal][error] invalid task $task_id" >&2
  exit 2
fi
temperature="${TEMPERATURES[$task_id]}"
temperature_slug=$(python - "$temperature" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value <= 1.0:
    raise SystemExit("tail temperature must be finite and greater than one")
print(f"{value:.8g}".replace(".", "p"))
PY
)

SELECTION="$SMC_ROOT/pilot_selection/selection_summary.json"
test -s "$SELECTION"
variant=$(python - "$SELECTION" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("selection_status") != "PASS":
    raise SystemExit("SMC likelihood selection did not pass")
print(payload["selected_variant"])
PY
)
case "$variant" in
  floor_0p05) CONFIG=configs/experiments/popcosmos_native15d_rws_floor05.yaml ;;
  *) echo "[defensive-proposal][error] expected floor_0p05, got $variant" >&2; exit 2 ;;
esac

DATASET=Data/cosmos2020/prepared/farmer_a24_full.parquet
CHECKPOINT="$BASELINE_REFRESH_OUT/encoder_refresh/checkpoints/best.eqx"
FEATURE_STATS="$BASELINE_REFRESH_OUT/encoder_refresh/feature_stats.json"
BASE_PROPOSAL="$BASELINE_REFRESH_OUT/moderate_k${PROBE_SAMPLES}_proposal"
TAIL_PROPOSAL="$TEMPERATURE_SCAN_ROOT/temperature_${temperature_slug}/proposal_k${PROBE_SAMPLES}"
OUT="$DEFENSIVE_SCAN_ROOT/tail_temperature_${temperature_slug}"
test -s "$CHECKPOINT"
test -s "$FEATURE_STATS"
test -s "$BASE_PROPOSAL/inference_summary.json"
test -s "$TAIL_PROPOSAL/inference_summary.json"
test ! -e "$OUT"

echo "[defensive-proposal] task=$task_id temperature=$temperature fractions=$TAIL_FRACTIONS_CSV"
echo "[defensive-proposal] base=$BASE_PROPOSAL tail=$TAIL_PROPOSAL out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/evaluate_popcosmos_defensive_proposal.py \
  --base-posterior "$BASE_PROPOSAL" \
  --tail-posterior "$TAIL_PROPOSAL" \
  --tail-temperature "$temperature" \
  --tail-fractions "$TAIL_FRACTIONS_CSV" \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" \
  --out "$OUT" \
  --object-batch-size 64 \
  --sample-chunk-size 128 \
  --resample-count 256 \
  --seed 260817 \
  --min-median-ess-fraction "$MIN_MEDIAN_ESS_FRACTION" \
  --max-fraction-pareto-k-gt-0p7 "$MAX_FRACTION_PARETO_K_GT_0P7" \
  --require-gpu
