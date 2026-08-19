#!/bin/bash
#SBATCH --job-name=cosmos_smc_em
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT}"
SMC_ROOT="${SMC_ROOT:?Set SMC_ROOT}"
OUT="${OUT:?Set OUT}"
EM_EPOCHS="${EM_EPOCHS:-5}"
EM_OBJECT_BATCH_SIZE="${EM_OBJECT_BATCH_SIZE:-16}"
EM_LEARNING_RATE="${EM_LEARNING_RATE:-1e-5}"
EM_TRUST_STRENGTH="${EM_TRUST_STRENGTH:-0.2}"
EM_VALIDATION_FRACTION="${EM_VALIDATION_FRACTION:-0.25}"

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

JOB_CACHE_ROOT="${SCRATCH:?Set SCRATCH}/dsps-popcosmos/smc-em-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$JOB_CACHE_ROOT/xdg"
export JAX_COMPILATION_CACHE_DIR="$JOB_CACHE_ROOT/jax"
export HF_HOME="$JOB_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$JOB_CACHE_ROOT/tmp"
mkdir -p "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR" "$HF_HUB_CACHE" "$TMPDIR"

cd "$REPO_DIR"
mkdir -p outputs/logs
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

SELECTION="$SMC_ROOT/pilot_selection/selection_summary.json"
test -s "$SELECTION"
variant=$(python - "$SELECTION" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("selection_status") != "PASS":
    raise SystemExit("SMC likelihood selection did not pass")
if payload.get("selected_variant") != "floor_0p05":
    raise SystemExit("Direct SMC empirical Bayes is frozen to floor_0p05")
print(payload["selected_variant"])
PY
)

CONFIG=configs/experiments/popcosmos_native15d_rws_floor05.yaml
SOURCE_CHECKPOINT="$SOURCE_ROOT/train/checkpoints/best.eqx"
BANK_A="$SMC_ROOT/$variant/seed_260817"
BANK_B="$SMC_ROOT/$variant/seed_260818"
for path in \
  "$CONFIG" \
  "$SOURCE_CHECKPOINT" \
  "$BANK_A/DONE" \
  "$BANK_A/smc_summary.json" \
  "$BANK_B/DONE" \
  "$BANK_B/smc_summary.json"; do
  test -e "$path"
done
test ! -e "$OUT"

echo "[smc-empirical-bayes] repo=$REPO_DIR"
echo "[smc-empirical-bayes] source_checkpoint=$SOURCE_CHECKPOINT"
echo "[smc-empirical-bayes] banks=$BANK_A,$BANK_B"
echo "[smc-empirical-bayes] out=$OUT"
echo "[smc-empirical-bayes] caches=$JOB_CACHE_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train_popcosmos_smc_empirical_bayes.py \
  --config "$CONFIG" \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --bank "$BANK_A" \
  --bank "$BANK_B" \
  --out "$OUT" \
  --epochs "$EM_EPOCHS" \
  --object-batch-size "$EM_OBJECT_BATCH_SIZE" \
  --learning-rate "$EM_LEARNING_RATE" \
  --trust-strength "$EM_TRUST_STRENGTH" \
  --validation-fraction "$EM_VALIDATION_FRACTION" \
  --require-gpu
