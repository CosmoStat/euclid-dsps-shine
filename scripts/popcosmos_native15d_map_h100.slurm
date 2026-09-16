#!/bin/bash
#SBATCH --job-name=cosmos_map15
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
CONFIG="${CONFIG:-configs/experiments/popcosmos_native15d_rws.yaml}"
DATASET="${DATASET:-Data/cosmos2020/prepared/farmer_a24_n512.parquet}"
NORMALIZATION_CHECKPOINT="${NORMALIZATION_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
OUT="${OUT:-outputs/runs/popcosmos_native15d_map_${RUN_TAG}}"

LIMIT="${LIMIT:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_STARTS="${N_STARTS:-16}"
START_CHUNK_SIZE="${START_CHUNK_SIZE:-4}"
MAXITER="${MAXITER:-400}"
LEARNING_RATE="${LEARNING_RATE:-0.02}"
SEED="${SEED:-260730}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-50}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID:-local}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"

module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

test -s Data/cosmos2020/prepared/PREPOST_COMPLETE.json
test -s "$DATASET"
test -s "$NORMALIZATION_CHECKPOINT"
test -s "${NORMALIZATION_CHECKPOINT}.json"
test ! -e "$OUT" || {
  echo "[cosmos-map15][error] refusing to overwrite $OUT"
  exit 2
}

python - "$CONFIG" "$NORMALIZATION_CHECKPOINT" <<'PY'
import sys

import jax

from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.config import load_config
from euclid_dsps.cosmos2020 import cosmos_band_names_for_subset

config = load_config(sys.argv[1])
amortized = amortized_config(config)
configured_checkpoint = amortized["latent"]["normalization_checkpoint"]
if configured_checkpoint != sys.argv[2]:
    raise SystemExit(
        "[cosmos-map15][error] NORMALIZATION_CHECKPOINT must match the "
        f"config: {configured_checkpoint!r} != {sys.argv[2]!r}"
    )
names = tuple(config["fit"]["free_parameters"])
print("[cosmos-map15] latent:", len(names), names)
print("[cosmos-map15] model:", config["model"]["sfh_model"])
subset = (config.get("dataset", {}) or {}).get("band_subset", "cosmos26")
expected_names = cosmos_band_names_for_subset(subset)
configured_names = tuple(band["name"] for band in config["bands"])
print("[cosmos-map15] band_subset:", subset)
print("[cosmos-map15] bands:", len(config["bands"]))
print("[cosmos-map15] backend:", jax.default_backend())
print("[cosmos-map15] devices:", jax.devices())
if len(names) != 15 or config["model"]["sfh_model"] != "spline15d":
    raise SystemExit("[cosmos-map15][error] native 15D contract is not active")
if configured_names != expected_names:
    raise SystemExit(
        "[cosmos-map15][error] unexpected ordered COSMOS band subset: "
        f"subset={subset!r} expected={expected_names} got={configured_names}"
    )
if jax.default_backend() != "gpu":
    raise SystemExit("[cosmos-map15][error] JAX did not select the H100")
PY

echo "[cosmos-map15] dataset=$DATASET limit=$LIMIT out=$OUT"
echo "[cosmos-map15] likelihood-only MAP: prior_weight=0 checkpoint=none"
echo "[cosmos-map15] batch_size=$BATCH_SIZE start_chunk_size=$START_CHUNK_SIZE progress_interval=$PROGRESS_INTERVAL"
nvidia-smi

python -m euclid_dsps.cli \
  --config "$CONFIG" \
  diffsky-map-adam-prior \
  --dataset "$DATASET" \
  --out "$OUT" \
  --limit "$LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --n-starts "$N_STARTS" \
  --start-chunk-size "$START_CHUNK_SIZE" \
  --maxiter "$MAXITER" \
  --learning-rate "$LEARNING_RATE" \
  --prior-weight 0 \
  --start-mode latin_hypercube \
  --progress-interval "$PROGRESS_INTERVAL" \
  --seed "$SEED" \
  --selection-mode random \
  --selection-seed "$SEED"

test -s "$OUT/map_summary.json"
test -s "$OUT/map_estimates.parquet"
test -s "$OUT/map_photometry.parquet"
test -s "$OUT/map_normalized_residuals_by_band.csv"

python - "$OUT/map_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
fit = summary["photometric_fit"]
print("[cosmos-map15] photometric fit:", json.dumps(fit, sort_keys=True))
if summary["checkpoint"] is not None or summary["uses_learned_prior"]:
    raise SystemExit("[cosmos-map15][error] MAP unexpectedly used a learned model")
if fit["n_parameters"] != 15:
    raise SystemExit("[cosmos-map15][error] MAP did not use the native 15D latent")
PY

touch "$OUT/DONE"
echo "[cosmos-map15] complete -> $OUT"
