#!/bin/bash
#SBATCH --job-name=feniks_sc_popvi
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
PILOT_ROOT="${PILOT_ROOT:?PILOT_ROOT is required}"
FIT_BANK="${FIT_BANK:?FIT_BANK must be the complete direct-q training bank}"
VALIDATION_BANK="${VALIDATION_BANK:?VALIDATION_BANK must be the complete direct-q validation bank}"
EXPECTED_FIT_OBJECTS="${EXPECTED_FIT_OBJECTS:?EXPECTED_FIT_OBJECTS is required}"
EXPECTED_VALIDATION_OBJECTS="${EXPECTED_VALIDATION_OBJECTS:?EXPECTED_VALIDATION_OBJECTS is required}"
POPULATION_VI_OUT="${POPULATION_VI_OUT:-$PILOT_ROOT/population_vi/round_01}"
POPULATION_VI_STEPS="${POPULATION_VI_STEPS:-32}"
POPULATION_VI_LEARNING_RATE="${POPULATION_VI_LEARNING_RATE:-1e-5}"
SELECTION_REFERENCE_SAMPLES="${SELECTION_REFERENCE_SAMPLES:-16384}"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1 EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"

cd "$REPO_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$CACHE_ROOT/jax" "$(dirname "$POPULATION_VI_OUT")"

readarray -t VALUES < <(python - "$PILOT_ROOT" "$REPO_DIR" \
  "$EXPECTED_FIT_OBJECTS" "$EXPECTED_VALIDATION_OBJECTS" <<'PY'
import json
import sys
from pathlib import Path

from euclid_dsps.amortized.population_vem import require_git_commit, sha256_file

root = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
expected_fit = int(sys.argv[3])
expected_validation = int(sys.argv[4])
manifest = json.loads((root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
winner = json.loads(
    (root / "NPE_TOPOLOGY_WINNER_FROZEN.json").read_text(encoding="utf-8")
)
ready = json.loads((root / "POPULATION_VI_READY.json").read_text(encoding="utf-8"))
require_git_commit(repo, manifest["code_commit"])
if winner.get("status") != "FROZEN" or winner.get("prior_bitwise_unchanged") is not True:
    raise ValueError("topology-corrected posterior winner is not frozen")
if ready.get("status") != "PASS" or ready.get("truth_used") is not False:
    raise ValueError("population VI gate is not passing")
if ready.get("posterior_checkpoint_sha256") != winner.get("checkpoint_sha256"):
    raise ValueError("population gate and frozen winner identify different posteriors")
if sha256_file(winner["checkpoint"]) != winner["checkpoint_sha256"]:
    raise ValueError("frozen posterior checkpoint SHA256 mismatch")
if expected_fit != int(manifest["cohorts"]["train"]["objects"]):
    raise ValueError("fit bank must cover the complete manifest training split")
if expected_validation != int(manifest["cohorts"]["validation"]["objects"]):
    raise ValueError("validation bank must cover the complete manifest validation split")
print(winner["checkpoint"])
print(winner["feature_stats"])
print(root / "POPULATION_VI_READY.json")
PY
)
POSTERIOR_CHECKPOINT="${VALUES[0]}"
FEATURE_STATS="${VALUES[1]}"
READINESS_RECEIPT="${VALUES[2]}"
CONFIG="$REPO_DIR/configs/experiments/feniks_sc_drws_r29_population_marginal_vi_gated.yaml"

test ! -e "$POPULATION_VI_OUT/POPULATION_MSTEP_COMPLETE.json" || {
  echo "[population-vi][error] output already contains a completed M-step" >&2
  exit 2
}
rm -rf "$POPULATION_VI_OUT"
nvidia-smi
python scripts/train_feniks_sc_drws_population_marginal_vi.py \
  --config "$CONFIG" \
  --posterior-checkpoint "$POSTERIOR_CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" \
  --fit-bank "$FIT_BANK" \
  --validation-bank "$VALIDATION_BANK" \
  --readiness-receipt "$READINESS_RECEIPT" \
  --out "$POPULATION_VI_OUT" \
  --expected-fit-objects "$EXPECTED_FIT_OBJECTS" \
  --expected-validation-objects "$EXPECTED_VALIDATION_OBJECTS" \
  --steps "$POPULATION_VI_STEPS" \
  --learning-rate "$POPULATION_VI_LEARNING_RATE" \
  --selection-reference-samples "$SELECTION_REFERENCE_SAMPLES" \
  --seed 260906 \
  --activate-population-vi

echo "[population-vi] one prior M-step complete: $POPULATION_VI_OUT"
echo "[population-vi] adapt q to this prior before any subsequent M-step"
