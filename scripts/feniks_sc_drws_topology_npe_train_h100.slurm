#!/bin/bash
#SBATCH --job-name=feniks_sc_toponpe
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
PILOT_ROOT="${PILOT_ROOT:?PILOT_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
NPE_STAGE="${NPE_STAGE:?NPE_STAGE must be B or C}"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1 EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}-${NPE_STAGE}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"

cd "$REPO_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$CACHE_ROOT/jax" "$MPLCONFIGDIR" "$PILOT_ROOT/arms/$NPE_STAGE"

readarray -t VALUES < <(python - "$PILOT_ROOT/RUN_MANIFEST.json" "$NPE_STAGE" "$REPO_DIR" <<'PY'
import json
import sys
from pathlib import Path
from euclid_dsps.amortized.population_vem import require_git_commit

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
stage = sys.argv[2]
require_git_commit(Path(sys.argv[3]), manifest["code_commit"])
print(manifest["configs"][stage])
print(manifest["dataset"]["path"])
print(manifest["cohorts"]["train"]["path"])
print(manifest["cohorts"]["validation"]["path"])
print(manifest["source"]["checkpoint"])
print(manifest["source"]["config"])
print(manifest["source"]["feature_stats"])
print(manifest["budgets"][f"{stage}_epochs"])
PY
)
BASE_CONFIG="${VALUES[0]}"
DATASET="${VALUES[1]}"
TRAIN_INDICES="${VALUES[2]}"
VALIDATION_INDICES="${VALUES[3]}"
SOURCE_CHECKPOINT="${VALUES[4]}"
SOURCE_CONFIG="${VALUES[5]}"
FEATURE_STATS="${VALUES[6]}"
EPOCHS="${VALUES[7]}"
OUT="$PILOT_ROOT/arms/$NPE_STAGE/train"
RECEIPT="$PILOT_ROOT/arms/$NPE_STAGE/ARM_COMPLETE.json"
SLEEP_CACHE="$PILOT_ROOT/cache/frozen_parent_noiseless_sleep.npz"
CONFIG="$PILOT_ROOT/arms/$NPE_STAGE/runtime_config.yaml"

python - "$BASE_CONFIG" "$CONFIG" "$SLEEP_CACHE" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
cache = Path(sys.argv[3]).resolve()
config = yaml.safe_load(source.read_text(encoding="utf-8"))
cache_config = config["amortized"]["objective"]["sleep"][
    "noiseless_flux_cache"
]
cache_config["enabled"] = True
cache_config["path"] = str(cache)
payload = yaml.safe_dump(config, sort_keys=False)
if target.is_file() and target.read_text(encoding="utf-8") != payload:
    raise ValueError(f"runtime configuration already exists with different content: {target}")
target.write_text(payload, encoding="utf-8")
PY

if [[ -s "$RECEIPT" ]]; then
  echo "[topology-npe] arm $NPE_STAGE already complete"
  exit 0
fi
if [[ "$NPE_STAGE" == B ]]; then
  ARM_SEED=261006
  INITIAL_ROOT="$PILOT_ROOT/arms/B/topology_initial"
  if [[ ! -s "$INITIAL_ROOT/TOPOLOGY_CORRECTED_INITIAL_COMPLETE.json" ]]; then
    rm -rf "$INITIAL_ROOT"
    python scripts/build_feniks_sc_drws_topology_corrected_checkpoint.py \
      --source-config "$SOURCE_CONFIG" --target-config "$CONFIG" \
      --source-checkpoint "$SOURCE_CHECKPOINT" --feature-stats "$FEATURE_STATS" \
      --out "$INITIAL_ROOT" --seed 260906 --minimum-transform-count 2
  fi
  INITIAL_CHECKPOINT="$INITIAL_ROOT/topology_corrected_frozen_parent.eqx"
else
  ARM_SEED=261106
  test -s "$PILOT_ROOT/arms/B/ARM_COMPLETE.json"
  INITIAL_CHECKPOINT="$PILOT_ROOT/arms/B/train/checkpoints/best.eqx"
fi

rm -rf "$OUT"
mkdir -p "$PILOT_ROOT/cache"
echo "[topology-npe] stage=$NPE_STAGE epochs=$EPOCHS initial=$INITIAL_CHECKPOINT"
nvidia-smi
python -m euclid_dsps.cli --config "$CONFIG" amortized-train-diffsky \
  --runtime gpu --dataset "$DATASET" --out "$OUT" \
  --train-indices-file "$TRAIN_INDICES" \
  --validation-indices-file "$VALIDATION_INDICES" \
  --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --fixed-feature-stats "$FEATURE_STATS" \
  --epochs "$EPOCHS" --batch-size 512 --jax-batch-size 256 \
  --n-samples 1 --kl-annealing-epochs 0 --kl-weight-max 0.0 \
  --validation-every 1 --best-checkpoint-min-epoch 1 \
  --data-parallel pmap --freeze-prior --seed "$ARM_SEED" \
  --no-progress

python scripts/finalize_feniks_sc_drws_topology_npe_arm.py \
  --root "$PILOT_ROOT" --arm "$NPE_STAGE"
echo "[topology-npe] complete arm=$NPE_STAGE"
