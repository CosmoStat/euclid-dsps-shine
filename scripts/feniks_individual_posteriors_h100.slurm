#!/bin/bash
#SBATCH --job-name=feniks_indpost
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"
LABELS=(
  selfsup_rws_k8_t2
  selfsup_rws_mix2_k8_t2
  selfsup_smcwake_mix2_k4_t2
)
CONFIGS=(
  configs/experiments/feniks_selfsup_learned_rws_k8_t2.yaml
  configs/experiments/feniks_selfsup_learned_rws_mix2_k8_t2.yaml
  configs/experiments/feniks_selfsup_learned_smcwake_mix2_k4_t2.yaml
)
if (( TASK_ID < 0 || TASK_ID >= ${#LABELS[@]} )); then
  echo "[individual-posterior][error] invalid task id: $TASK_ID"
  exit 2
fi

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_selfsup_learned_t2_jaxcosmo_v1}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-2048}"
PRIOR_SAMPLES="${PRIOR_SAMPLES:-4096}"

LABEL="${LABELS[$TASK_ID]}"
CONFIG="${CONFIGS[$TASK_ID]}"
TASK_DIR="$ROOT_DIR/$LABEL"
CHECKPOINT="$TASK_DIR/train/checkpoints/best.eqx"
FEATURE_STATS="$TASK_DIR/train/feature_stats.json"
DATASET="$CATALOG_DIR/test.parquet"
OUT_DIR="$TASK_DIR/individual_posteriors"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export JAX_PLATFORMS=cuda
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}-individual-posterior"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"
# H100 nodes use a different CPU architecture and module tree than the login node.
# Do not inherit an incompatible login-node environment into the batch job.
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

for path in "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" \
  "$FEATURE_STATS" scripts/generate_feniks_individual_posteriors.py; do
  test -s "$path" || {
    echo "[individual-posterior][error] missing: $path"
    exit 2
  }
done
test ! -f "$OUT_DIR/DONE" || {
  echo "[individual-posterior][error] output already complete: $OUT_DIR"
  exit 2
}

python - <<'PY'
import jax

print("[individual-posterior] backend:", jax.default_backend())
print("[individual-posterior] devices:", jax.devices())
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
    raise SystemExit("[individual-posterior][error] expected exactly one local GPU")
PY

echo "[individual-posterior] label=$LABEL dataset=$DATASET"
echo "[individual-posterior] checkpoint=$CHECKPOINT"
nvidia-smi

python scripts/generate_feniks_individual_posteriors.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" \
  --out "$OUT_DIR" \
  --posterior-samples "$POSTERIOR_SAMPLES" \
  --prior-samples "$PRIOR_SAMPLES" \
  --seed 260723
