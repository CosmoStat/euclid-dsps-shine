#!/bin/bash
#SBATCH --job-name=cosmos_qexpr
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:?Set DIAGNOSTIC_ROOT}"
UPDATED_CHECKPOINT="${UPDATED_CHECKPOINT:?Set UPDATED_CHECKPOINT}"
UPDATED_FEATURE_STATS="${UPDATED_FEATURE_STATS:?Set UPDATED_FEATURE_STATS}"
EPOCHS="${EPOCHS:-40}"
PROPOSAL_SAMPLES="${PROPOSAL_SAMPLES:-2048}"
DECODER_SAMPLE_CHUNK_SIZE="${DECODER_SAMPLE_CHUNK_SIZE:-1}"
EXPERTS="${EXPERTS:-2}"

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

CACHE_ROOT="${SCRATCH:-${WORK:?Set SCRATCH or WORK}/tmp}/dsps-qexpr-${SLURM_JOB_ID}"
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
OUT="$DIAGNOSTIC_ROOT/proposal_expressivity"

for path in "$CONFIG" "$DATASET" "$UPDATED_CHECKPOINT" "${UPDATED_CHECKPOINT}.json" "$UPDATED_FEATURE_STATS" "$PANEL"; do
  test -s "$path" || { echo "[proposal-expressivity][error] missing: $path" >&2; exit 2; }
done
for path in "$BANK_A/DONE" "$BANK_B/DONE"; do
  test -e "$path" || { echo "[proposal-expressivity][error] missing: $path" >&2; exit 2; }
done
test ! -e "$OUT"

python - "$BANK_A" "$BANK_B" <<'PY'
import json
import sys
from pathlib import Path

for value in sys.argv[1:]:
    root = Path(value)
    gate = json.loads((root / "support_gate.json").read_text())
    if gate["status"] != "PASS":
        raise RuntimeError(f"diagnostic SMC support failed: {root}")
print("[proposal-expressivity] diagnostic SMC support: PASS", flush=True)
PY

echo "[proposal-expressivity] repo=$REPO_DIR"
echo "[proposal-expressivity] root=$DIAGNOSTIC_ROOT"
echo "[proposal-expressivity] checkpoint=$UPDATED_CHECKPOINT"
echo "[proposal-expressivity] caches=$CACHE_ROOT"
echo "[proposal-expressivity] XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "[proposal-expressivity] JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/diagnose_popcosmos_proposal_expressivity.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --checkpoint "$UPDATED_CHECKPOINT" \
  --feature-stats "$UPDATED_FEATURE_STATS" \
  --panel "$PANEL" \
  --bank "$BANK_A" \
  --bank "$BANK_B" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --object-batch-size "${OBJECT_BATCH_SIZE:-8}" \
  --learning-rate "${LEARNING_RATE:-2e-5}" \
  --weight-decay "${WEIGHT_DECAY:-1e-6}" \
  --experts "$EXPERTS" \
  --mixture-mean-offset "${MIXTURE_MEAN_OFFSET:-0.05}" \
  --proposal-samples "$PROPOSAL_SAMPLES" \
  --decoder-sample-chunk-size "$DECODER_SAMPLE_CHUNK_SIZE" \
  --geometry-draws "${GEOMETRY_DRAWS:-256}" \
  --geometry-projections "${GEOMETRY_PROJECTIONS:-64}" \
  --min-median-ess-fraction 0.05 \
  --max-fraction-pareto-k-gt-0p7 0.2 \
  --seed "${SEED:-260819}" \
  --require-gpu
