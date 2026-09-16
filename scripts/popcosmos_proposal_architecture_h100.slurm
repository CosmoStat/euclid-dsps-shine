#!/bin/bash
#SBATCH --job-name=cosmos_qarch
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:?Set DIAGNOSTIC_ROOT}"
ARCHITECTURE_ROOT="${ARCHITECTURE_ROOT:?Set ARCHITECTURE_ROOT}"
UPDATED_CHECKPOINT="${UPDATED_CHECKPOINT:?Set UPDATED_CHECKPOINT}"
UPDATED_FEATURE_STATS="${UPDATED_FEATURE_STATS:?Set UPDATED_FEATURE_STATS}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script requires a Slurm array task}"

phase0_candidates=(
  current_compressed
  oracle_kde
  free_context_rqspline
  direct_context_realnvp
)
phase1_candidates=(
  current_compressed
  direct_context_realnvp
  direct_context_rqspline_medium
  direct_context_rqspline_large
  band_token_rqspline
)
phase1_seeds=(260820 260821 260822)

if (( TASK_ID < ${#phase0_candidates[@]} )); then
  phase=phase0
  candidate="${phase0_candidates[$TASK_ID]}"
  seed=260820
else
  offset=$((TASK_ID - ${#phase0_candidates[@]}))
  candidate_index=$((offset / ${#phase1_seeds[@]}))
  seed_index=$((offset % ${#phase1_seeds[@]}))
  if (( candidate_index >= ${#phase1_candidates[@]} )); then
    echo "[proposal-architecture][error] invalid array task: $TASK_ID" >&2
    exit 2
  fi
  phase=phase1
  candidate="${phase1_candidates[$candidate_index]}"
  seed="${phase1_seeds[$seed_index]}"
fi
OUT="$ARCHITECTURE_ROOT/$phase/$candidate/seed_$seed"

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

CACHE_ROOT="${SCRATCH:-${WORK:?Set SCRATCH or WORK}/tmp}/dsps-qarch-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export HF_HOME="$CACHE_ROOT/huggingface"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$CACHE_ROOT/tmp"
mkdir -p "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR" "$HF_HOME" "$WANDB_DIR" "$TMPDIR"

cd "$REPO_DIR"
mkdir -p outputs/logs
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

CONFIG=configs/experiments/popcosmos_native15d_rws_floor05.yaml
DATASET=Data/cosmos2020/prepared/farmer_a24_full.parquet
PANEL="$DIAGNOSTIC_ROOT/cohorts/diagnostic_panel.parquet"
BANK_A="$DIAGNOSTIC_ROOT/floor_0p05/seed_260817"
BANK_B="$DIAGNOSTIC_ROOT/floor_0p05/seed_260818"
for path in "$CONFIG" "$DATASET" "$UPDATED_CHECKPOINT" "${UPDATED_CHECKPOINT}.json" "$UPDATED_FEATURE_STATS" "$PANEL" "$BANK_A/DONE" "$BANK_B/DONE"; do
  test -e "$path" || { echo "[proposal-architecture][error] missing: $path" >&2; exit 2; }
done
test ! -e "$OUT" || {
  echo "[proposal-architecture][error] output exists: $OUT" >&2
  exit 2
}

echo "[proposal-architecture] task=$TASK_ID phase=$phase candidate=$candidate seed=$seed"
echo "[proposal-architecture] diagnostic_root=$DIAGNOSTIC_ROOT out=$OUT"
echo "[proposal-architecture] caches=$CACHE_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/evaluate_popcosmos_proposal_architecture.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --checkpoint "$UPDATED_CHECKPOINT" \
  --feature-stats "$UPDATED_FEATURE_STATS" \
  --panel "$PANEL" \
  --bank-a "$BANK_A" \
  --bank-b "$BANK_B" \
  --phase "$phase" \
  --candidate "$candidate" \
  --out "$OUT" \
  --epochs "${EPOCHS:-80}" \
  --object-batch-size "${OBJECT_BATCH_SIZE:-8}" \
  --learning-rate "${LEARNING_RATE:-5e-5}" \
  --weight-decay "${WEIGHT_DECAY:-1e-6}" \
  --proposal-samples "${PROPOSAL_SAMPLES:-2048}" \
  --decoder-sample-chunk-size "${DECODER_SAMPLE_CHUNK_SIZE:-1}" \
  --geometry-draws "${GEOMETRY_DRAWS:-256}" \
  --geometry-projections "${GEOMETRY_PROJECTIONS:-64}" \
  --kde-centers "${KDE_CENTERS:-128}" \
  --kde-bandwidth "${KDE_BANDWIDTH:-0.3}" \
  --seed "$seed" \
  --require-gpu
