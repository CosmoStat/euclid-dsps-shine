#!/bin/bash
#SBATCH --job-name=feniks_sc_npe_arm
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
NPE_ROOT="${NPE_ROOT:?NPE_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

case "$TASK" in
  0) ARM=warm_start ;;
  1) ARM=scratch_encoder ;;
  *) echo "[frozen-npe][error] invalid task: $TASK" >&2; exit 2 ;;
esac

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1 EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"

cd "$REPO_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$CACHE_ROOT/jax" "$MPLCONFIGDIR" "$NPE_ROOT/arms/$ARM"

readarray -t VALUES < <(python - "$NPE_ROOT" "$REPO_DIR" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
repo = Path(sys.argv[2])
manifest = json.loads((root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
actual = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
).strip()
if actual != manifest["code_commit"]:
    raise SystemExit(f"runtime commit mismatch: {actual} != {manifest['code_commit']}")
print(manifest["config"]["path"])
print(manifest["dataset"]["train"])
print(manifest["cohorts"]["train"]["path"])
print(manifest["cohorts"]["validation"]["path"])
print(manifest["frozen_parent"]["checkpoint"])
print(manifest["frozen_parent"]["feature_stats"])
print(manifest["training"]["epochs"])
print(manifest["request"]["seed"])
PY
)
CONFIG="${VALUES[0]}"
TRAIN_CATALOG="${VALUES[1]}"
TRAIN_INDICES="${VALUES[2]}"
VALIDATION_INDICES="${VALUES[3]}"
PARENT_CHECKPOINT="${VALUES[4]}"
FEATURE_STATS="${VALUES[5]}"
EPOCHS="${VALUES[6]}"
BASE_SEED="${VALUES[7]}"
FINAL="$NPE_ROOT/arms/$ARM/train"
RECEIPT="$NPE_ROOT/arms/$ARM/ARM_COMPLETE.json"

if [[ -s "$RECEIPT" ]]; then
  echo "[frozen-npe] arm already complete: $ARM"
  exit 0
fi

INITIAL_CHECKPOINT="$PARENT_CHECKPOINT"
ARM_SEED="$((BASE_SEED + TASK * 1009))"
if [[ "$ARM" == scratch_encoder ]]; then
  SCRATCH_ROOT="$NPE_ROOT/arms/$ARM/initial"
  python scripts/build_feniks_sc_drws_scratch_encoder_checkpoint.py \
    --config "$CONFIG" --parent-checkpoint "$PARENT_CHECKPOINT" \
    --feature-stats "$FEATURE_STATS" --out "$SCRATCH_ROOT" \
    --seed "$ARM_SEED"
  INITIAL_CHECKPOINT="$SCRATCH_ROOT/scratch_encoder_projected_parent.eqx"
fi

if [[ -d "$FINAL" ]]; then
  echo "[frozen-npe] validating existing completed arm output: $FINAL"
  python scripts/validate_feniks_sc_drws_frozen_parent_npe_arm.py \
    --root "$NPE_ROOT" --arm "$ARM" \
    --initial-checkpoint "$INITIAL_CHECKPOINT"
  touch "$NPE_ROOT/arms/$ARM/DONE"
  exit 0
fi

ATTEMPT="$NPE_ROOT/arms/$ARM/.train-${SLURM_JOB_ID}"
rm -rf "$ATTEMPT"
echo "[frozen-npe] arm=$ARM seed=$ARM_SEED epochs=$EPOCHS gpus=4"
nvidia-smi
python -m euclid_dsps.cli --config "$CONFIG" amortized-train-diffsky \
  --runtime gpu --dataset "$TRAIN_CATALOG" --out "$ATTEMPT" \
  --train-indices-file "$TRAIN_INDICES" \
  --validation-indices-file "$VALIDATION_INDICES" \
  --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --fixed-feature-stats "$FEATURE_STATS" \
  --epochs "$EPOCHS" --batch-size 1024 --jax-batch-size 256 \
  --n-samples 1 --kl-annealing-epochs 0 --kl-weight-max 0.0 \
  --validation-every 2 --best-checkpoint-min-epoch 2 \
  --data-parallel pmap --freeze-prior --seed "$ARM_SEED" --no-progress

mv "$ATTEMPT" "$FINAL"
python scripts/validate_feniks_sc_drws_frozen_parent_npe_arm.py \
  --root "$NPE_ROOT" --arm "$ARM" \
  --initial-checkpoint "$INITIAL_CHECKPOINT"
touch "$NPE_ROOT/arms/$ARM/DONE"
echo "[frozen-npe] complete arm=$ARM"
