#!/bin/bash
#SBATCH --job-name=feniks_t2jlens
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-11%12
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

CONFIGS=(
  configs/experiments/feniks_selfsup_learned_rws_k8_t2.yaml
  configs/experiments/feniks_selfsup_learned_rws_mix2_k8_t2.yaml
  configs/experiments/feniks_selfsup_learned_smcwake_mix2_k4_t2.yaml
)
LABELS=(selfsup_rws_k8_t2 selfsup_rws_mix2_k8_t2 selfsup_smcwake_mix2_k4_t2)
NUM_SHARDS=4
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
RUN_INDEX=$((TASK_ID / NUM_SHARDS))
SHARD_INDEX=$((TASK_ID % NUM_SHARDS))
CONFIG="${CONFIGS[$RUN_INDEX]}"
LABEL="${LABELS[$RUN_INDEX]}"
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SMOKE="${SMOKE:-0}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
TASK_DIR="$ROOT_DIR/$LABEL"
TRAIN_DIR="$TASK_DIR/train"
LENS_DIR="$TASK_DIR/jacobian_lens"
MAX_OBJECTS=512
if [[ "$SMOKE" == "1" ]]; then MAX_OBJECTS=12; fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MPLBACKEND=Agg
export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$LENS_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
test -f "$TASK_DIR/DONE" || { echo "[jlens][error] incomplete run: $TASK_DIR"; exit 2; }

python - <<'PY'
import jax
print("[jlens] devices:", jax.devices())
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
    raise SystemExit("[jlens][error] expected exactly one local GPU")
PY

python -m euclid_dsps.cli --config "$CONFIG" amortized-jacobian-lens-diffsky \
  --dataset "$CATALOG_DIR/test.parquet" \
  --checkpoint "$TRAIN_DIR/checkpoints/best.eqx" \
  --feature-stats "$TRAIN_DIR/feature_stats.json" \
  --out "$LENS_DIR" \
  --batch-size 16 \
  --selection-mode stratified_redshift \
  --stratified-strategy proportional \
  --selection-seed 260722 \
  --mode both \
  --include-prior-score \
  --include-ae-lens \
  --max-objects "$MAX_OBJECTS" \
  --direction-top-k 8 \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --resume
