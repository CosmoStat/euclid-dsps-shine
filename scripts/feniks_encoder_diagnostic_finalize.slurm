#!/bin/bash
#SBATCH --job-name=feniks_qdiag_gate
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
: "${CONFIG:?CONFIG is required}"
: "${DATASET:?DATASET is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${FEATURE_STATS:?FEATURE_STATS is required}"
: "${EXACT_ROOT:?EXACT_ROOT is required}"

export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

cd "$REPO_DIR"
mkdir -p "$MPLCONFIGDIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python scripts/run_feniks_exact_posterior_benchmark.py finalize-run \
  --config "$CONFIG" --dataset "$DATASET" --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" --out "$EXACT_ROOT" --samplers nuts

if [[ "${DIAGNOSTIC_MODE:-full}" == "smoke" ]]; then
  SAMPLES_PER_OBJECT=8
  BOOTSTRAP=20
  REGIONS=20
else
  SAMPLES_PER_OBJECT=128
  BOOTSTRAP=500
  REGIONS=100
fi

for domain in observed_catalog sleep_synthetic; do
  python scripts/collect_feniks_parentprior_exact_calibration.py \
    --root "$EXACT_ROOT" --samples-per-object "$SAMPLES_PER_OBJECT" \
    --domain "$domain"
  CALIBRATION="$EXACT_ROOT/calibration_${domain}"
  TRUTH="$CALIBRATION/inference_truth.parquet"
  POSTERIORS=(
    --posterior "q=$CALIBRATION/encoder"
    --posterior "q_is=$CALIBRATION/q_is"
    --posterior "defensive_is=$CALIBRATION/defensive_is"
    --posterior "nuts=$CALIBRATION/nuts"
  )
  python scripts/evaluate_feniks_tarp.py \
    --truth "$TRUTH" "${POSTERIORS[@]}" --out "$CALIBRATION/tarp" \
    --samples-per-object "$SAMPLES_PER_OBJECT" --num-bootstrap "$BOOTSTRAP" \
    --drop-nonfinite-truth
  python scripts/evaluate_feniks_mira.py \
    --truth "$TRUTH" "${POSTERIORS[@]}" --out "$CALIBRATION/mira" \
    --samples-per-object "$SAMPLES_PER_OBJECT" --num-regions "$REGIONS" \
    --num-bootstrap "$BOOTSTRAP" --drop-nonfinite-truth
done

python scripts/validate_feniks_encoder_diagnostic.py --root "$EXACT_ROOT"
touch "$EXACT_ROOT/DIAGNOSTIC_DONE"
echo "[feniks-qdiag] aggregate complete -> $EXACT_ROOT"
