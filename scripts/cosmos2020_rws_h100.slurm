#!/bin/bash
#SBATCH --job-name=cosmos_rws
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

STAGE="${STAGE:-smoke}"
CONFIG="${CONFIG:-configs/experiments/popcosmos_a24_rws_joint.yaml}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
DATA_DIR="${DATA_DIR:-Data/cosmos2020/prepared}"
ASSET_DIR="${ASSET_DIR:-Data/cosmos2020/assets}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/popcosmos_a24_rws_v2}"

case "$STAGE" in
  smoke) SIZE=512; EPOCHS=4; INFER_LIMIT=512; WALL_LABEL=smoke ;;
  n5k) SIZE=5000; EPOCHS=12; INFER_LIMIT=512; WALL_LABEL=n5k ;;
  n20k) SIZE=20000; EPOCHS=20; INFER_LIMIT=1000; WALL_LABEL=n20k ;;
  n40k) SIZE=40000; EPOCHS=32; INFER_LIMIT=1000; WALL_LABEL=n40k ;;
  full) SIZE=full; EPOCHS=48; INFER_LIMIT=5000; WALL_LABEL=full ;;
  *) echo "[cosmos-rws][error] STAGE must be smoke,n5k,n20k,n40k,full"; exit 2 ;;
esac

DATASET="$DATA_DIR/farmer_a24_${SIZE}.parquet"
if [[ "$SIZE" != "full" ]]; then DATASET="$DATA_DIR/farmer_a24_n${SIZE}.parquet"; fi
RUN_DIR="$ROOT_DIR/$WALL_LABEL"
TRAIN_DIR="$RUN_DIR/train"
INFER_DIR="$RUN_DIR/inference"

PREVIOUS=""
case "$STAGE" in
  n5k) PREVIOUS="$ROOT_DIR/smoke" ;;
  n20k) PREVIOUS="$ROOT_DIR/n5k" ;;
  n40k) PREVIOUS="$ROOT_DIR/n20k" ;;
  full) PREVIOUS="$ROOT_DIR/n40k" ;;
esac

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID:-local}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$RUN_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

test -s "$DATA_DIR/PREPOST_COMPLETE.json"
test -s "$DATASET"
test ! -e "$TRAIN_DIR" || {
  echo "[cosmos-rws][error] refusing to overwrite $TRAIN_DIR"; exit 2;
}
python scripts/validate_cosmos2020_reproduction.py \
  --config "$CONFIG" --data-dir "$DATA_DIR" --asset-dir "$ASSET_DIR"
python - <<'PY'
import jax
print("[cosmos-rws] backend:", jax.default_backend())
print("[cosmos-rws] devices:", jax.devices())
if jax.default_backend() != "gpu":
    raise SystemExit("[cosmos-rws][error] JAX did not select the H100")
PY

INITIAL_ARGS=()
DATA_PARALLEL=single
N_GPUS=$(nvidia-smi --list-gpus | wc -l)
if (( N_GPUS > 1 )); then DATA_PARALLEL=pmap; fi
if [[ -n "$PREVIOUS" ]]; then
  test -e "$PREVIOUS/DONE"
  PREVIOUS_VALIDATE_ARGS=()
  if [[ "$STAGE" == "n5k" ]]; then
    PREVIOUS_VALIDATE_ARGS+=(--allow-inference-fail)
  fi
  python scripts/validate_cosmos2020_reproduction.py \
    --config "$CONFIG" --data-dir "$DATA_DIR" --asset-dir "$ASSET_DIR" \
    --run-dir "$PREVIOUS" "${PREVIOUS_VALIDATE_ARGS[@]}"
  INITIAL_ARGS=(--initial-checkpoint "$PREVIOUS/train/checkpoints/best.eqx")
fi

echo "[cosmos-rws] stage=$STAGE dataset=$DATASET epochs=$EPOCHS"
echo "[cosmos-rws] n_gpus=$N_GPUS data_parallel=$DATA_PARALLEL"
nvidia-smi
python -m euclid_dsps.cli --config "$CONFIG" amortized-train-cosmos \
  --runtime gpu --dataset "$DATASET" --out "$TRAIN_DIR" \
  --batch-size 256 --jax-batch-size 256 --epochs "$EPOCHS" --n-samples 1 \
  --validation-every 1 --data-parallel "$DATA_PARALLEL" --no-progress \
  "${INITIAL_ARGS[@]}"

python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-cosmos \
  --dataset "$DATASET" --checkpoint "$TRAIN_DIR/checkpoints/best.eqx" \
  --feature-stats "$TRAIN_DIR/feature_stats.json" --out "$INFER_DIR" \
  --limit "$INFER_LIMIT" --batch-size 256 --jax-batch-size 256 \
  --posterior-samples 64 --prior-samples 4096 \
  --decoder-sample-chunk-size 1 --prior-predictive-batch-size 128 \
  --selection-mode random --selection-seed 260727 \
  --shard-outputs --no-residual-samples
python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-inference --out "$INFER_DIR" --limit "$INFER_LIMIT"
python scripts/compare_popcosmos_a24.py \
  --dataset "$DATASET" --inference "$INFER_DIR" \
  --a24-summaries "$ASSET_DIR/zenodo/summaries.txt" \
  --out "$RUN_DIR/a24_comparison"
CURRENT_VALIDATE_ARGS=()
if [[ "$STAGE" == "smoke" ]]; then
  CURRENT_VALIDATE_ARGS+=(--allow-inference-fail)
fi
python scripts/validate_cosmos2020_reproduction.py \
  --config "$CONFIG" --data-dir "$DATA_DIR" --asset-dir "$ASSET_DIR" \
  --run-dir "$RUN_DIR" "${CURRENT_VALIDATE_ARGS[@]}"
touch "$RUN_DIR/DONE"
