#!/bin/bash
#SBATCH --job-name=cosmos_smc
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT}"
LIMIT="${LIMIT:-256}"
N_SHARDS="${N_SHARDS:-4}"
PARTICLES="${PARTICLES:-1024}"
OBJECT_BATCH_SIZE="${OBJECT_BATCH_SIZE:-4}"
TARGET_ESS_FRACTION="${TARGET_ESS_FRACTION:-0.5}"
MAX_STAGES="${MAX_STAGES:-64}"
MALA_STEPS="${MALA_STEPS:-2}"
MALA_STEP_SIZE="${MALA_STEP_SIZE:-0.02}"
MALA_PARTICLE_CHUNK_SIZE="${MALA_PARTICLE_CHUNK_SIZE:-64}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SMC_VARIANTS_CSV="${SMC_VARIANTS_CSV:-floor_0p00,floor_0p02,floor_0p05}"
SMC_SEEDS_CSV="${SMC_SEEDS_CSV:-260817,260818}"
IFS=',' read -r -a SMC_VARIANTS <<< "$SMC_VARIANTS_CSV"
IFS=',' read -r -a SMC_SEEDS <<< "$SMC_SEEDS_CSV"
N_SEEDS="${#SMC_SEEDS[@]}"
GROUP=$((TASK / N_SHARDS))
SHARD=$((TASK % N_SHARDS))
VARIANT_INDEX=$((GROUP / N_SEEDS))
SEED_INDEX=$((GROUP % N_SEEDS))
if (( VARIANT_INDEX >= ${#SMC_VARIANTS[@]} )); then
  echo "[posthoc-smc][error] invalid array task: $TASK" >&2
  exit 2
fi
VARIANT="${SMC_VARIANTS[$VARIANT_INDEX]}"
SEED="${SMC_SEEDS[$SEED_INDEX]}"
case "$VARIANT" in
  floor_0p00) CONFIG=configs/experiments/popcosmos_native15d_rws.yaml ;;
  floor_0p02) CONFIG=configs/experiments/popcosmos_native15d_rws_floor02.yaml ;;
  floor_0p05) CONFIG=configs/experiments/popcosmos_native15d_rws_floor05.yaml ;;
  *) echo "[posthoc-smc][error] unsupported variant: $VARIANT" >&2; exit 2 ;;
esac

DATASET="Data/cosmos2020/prepared/farmer_a24_full.parquet"
CHECKPOINT="$SOURCE_ROOT/train/checkpoints/best.eqx"
FEATURE_STATS="$SOURCE_ROOT/train/feature_stats.json"
printf -v SHARD_TAG '%03d' "$SHARD"
ROW_INDICES="$OUTPUT_ROOT/cohorts/smc_calibration_indices_shard_${SHARD_TAG}.npy"
OUT="$OUTPUT_ROOT/$VARIANT/seed_$SEED/shard_${SHARD_TAG}"

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
mkdir -p outputs/logs "$OUTPUT_ROOT"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

for path in "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" "$FEATURE_STATS" "$ROW_INDICES"; do
  test -s "$path" || { echo "[posthoc-smc][error] missing: $path" >&2; exit 2; }
done
test ! -e "$OUT"
SHARD_LIMIT=$(python - "$ROW_INDICES" <<'PY'
import numpy as np
import sys
print(len(np.load(sys.argv[1])))
PY
)

echo "[posthoc-smc] task=$TASK group=$GROUP shard=$SHARD/$N_SHARDS variant=$VARIANT seed=$SEED objects=$SHARD_LIMIT/$LIMIT out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/run_popcosmos_posthoc_smc.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" \
  --row-indices "$ROW_INDICES" \
  --out "$OUT" \
  --limit "$SHARD_LIMIT" \
  --object-batch-size "$OBJECT_BATCH_SIZE" \
  --particles "$PARTICLES" \
  --target-ess-fraction "$TARGET_ESS_FRACTION" \
  --max-stages "$MAX_STAGES" \
  --mala-steps "$MALA_STEPS" \
  --mala-step-size "$MALA_STEP_SIZE" \
  --mala-particle-chunk-size "$MALA_PARTICLE_CHUNK_SIZE" \
  --seed "$SEED" \
  --require-gpu
