#!/bin/bash
#SBATCH --job-name=feniks_parent_xgate
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:?CONFIG is required}"
DATASET="${DATASET:?DATASET is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
FEATURE_STATS="${FEATURE_STATS:?FEATURE_STATS is required}"
EXACT_ROOT="${EXACT_ROOT:?EXACT_ROOT is required}"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

cd "$REPO_DIR"
mkdir -p "$MPLCONFIGDIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python scripts/run_feniks_exact_posterior_benchmark.py finalize-run \
  --config "$CONFIG" --dataset "$DATASET" --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" --out "$EXACT_ROOT" --samplers nuts

python scripts/collect_feniks_parentprior_exact_calibration.py \
  --root "$EXACT_ROOT" --samples-per-object 128

TRUTH="$EXACT_ROOT/calibration/inference_truth.parquet"
POSTERIORS=(
  --posterior "q=$EXACT_ROOT/calibration/encoder"
  --posterior "q_is=$EXACT_ROOT/calibration/q_is"
  --posterior "defensive_is=$EXACT_ROOT/calibration/defensive_is"
  --posterior "nuts=$EXACT_ROOT/calibration/nuts"
)
python scripts/evaluate_feniks_tarp.py \
  --truth "$TRUTH" "${POSTERIORS[@]}" \
  --out "$EXACT_ROOT/calibration/tarp" \
  --samples-per-object 128 --num-bootstrap 500 --drop-nonfinite-truth
python scripts/evaluate_feniks_mira.py \
  --truth "$TRUTH" "${POSTERIORS[@]}" \
  --out "$EXACT_ROOT/calibration/mira" \
  --samples-per-object 128 --num-regions 100 --num-bootstrap 500 \
  --drop-nonfinite-truth

python scripts/validate_feniks_parentprior_exact.py --root "$EXACT_ROOT"
echo "[feniks-parent-exact] aggregate complete -> $EXACT_ROOT"
