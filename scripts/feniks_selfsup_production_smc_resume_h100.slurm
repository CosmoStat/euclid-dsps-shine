#!/bin/bash
#SBATCH --job-name=feniks_t2smc_resume
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

CONFIG="configs/experiments/feniks_selfsup_learned_smcwake_mix2_k4_t2.yaml"
LABEL="selfsup_smcwake_mix2_k4_t2"
START_EPOCH="${START_EPOCH:-40}"
END_EPOCH="${END_EPOCH:-40}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_selfsup_learned_t2_jaxcosmo_v1}"
TASK_DIR="$ROOT_DIR/$LABEL"
SOURCE_TRAIN="${SOURCE_TRAIN:-$TASK_DIR/train_failed_epoch40}"
INITIAL_CHECKPOINT="$SOURCE_TRAIN/checkpoints/last.eqx"
TRAIN_DIR="$TASK_DIR/train"
INFER_DIR="$TASK_DIR/inference"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MALLOC_TRIM_THRESHOLD_=65536
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}-smc-resume"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$TASK_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

for path in "$CONFIG" "$CATALOG_DIR/train.parquet" "$CATALOG_DIR/test.parquet" \
  "$INITIAL_CHECKPOINT" "${INITIAL_CHECKPOINT}.json" \
  "$SOURCE_TRAIN/feature_stats.json"; do
  test -s "$path" || { echo "[smc-resume][error] missing: $path"; exit 2; }
done
test ! -e "$TRAIN_DIR" || { echo "[smc-resume][error] exists: $TRAIN_DIR"; exit 2; }
test ! -e "$INFER_DIR" || { echo "[smc-resume][error] exists: $INFER_DIR"; exit 2; }
test ! -f "$TASK_DIR/DONE" || { echo "[smc-resume][error] task already complete"; exit 2; }

python - "$INITIAL_CHECKPOINT.json" "$((START_EPOCH - 1))" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
epoch = int(json.load(open(path, encoding="utf-8"))["epoch"])
if epoch != expected:
    raise SystemExit(
        f"[smc-resume][error] checkpoint epoch {epoch}, expected {expected}"
    )
print(f"[smc-resume] checkpoint contract: epoch={epoch}")
PY

python - <<'PY'
import jax

print("[smc-resume] backend:", jax.default_backend())
print("[smc-resume] devices:", jax.devices())
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 4:
    raise SystemExit("[smc-resume][error] expected exactly four local GPUs")
PY

echo "[smc-resume] warm restart epochs=${START_EPOCH}-${END_EPOCH}"
echo "[smc-resume] optimizer_state_resumed=0 validation_every=0"
nvidia-smi

python -m euclid_dsps.cli --config "$CONFIG" amortized-train-diffsky \
  --dataset "$CATALOG_DIR/train.parquet" \
  --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --start-epoch "$START_EPOCH" \
  --out "$TRAIN_DIR" \
  --batch-size 1024 \
  --jax-batch-size 128 \
  --epochs "$END_EPOCH" \
  --n-samples 1 \
  --data-parallel pmap \
  --validation-every 0 \
  --no-progress

python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-diffsky \
  --dataset "$CATALOG_DIR/test.parquet" \
  --checkpoint "$TRAIN_DIR/checkpoints/best.eqx" \
  --feature-stats "$TRAIN_DIR/feature_stats.json" \
  --out "$INFER_DIR" \
  --limit 5000 \
  --batch-size 512 \
  --jax-batch-size 512 \
  --posterior-samples 128 \
  --prior-samples 16384 \
  --decoder-sample-chunk-size 1 \
  --prior-predictive-batch-size 256 \
  --seed 260722 \
  --selection-mode stratified_redshift \
  --stratified-strategy proportional \
  --selection-seed 260722 \
  --shard-outputs \
  --no-residual-samples

python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-inference --out "$INFER_DIR" --limit 5000

python scripts/summarize_feniks_selfsup_production_task.py \
  --label "$LABEL" --config "$CONFIG" --train "$TRAIN_DIR" \
  --inference "$INFER_DIR" --out "$TASK_DIR/metrics.json"

cat >"$TASK_DIR/recovery_provenance.json" <<EOF
{
  "source_train": "$SOURCE_TRAIN",
  "initial_checkpoint": "$INITIAL_CHECKPOINT",
  "start_epoch": $START_EPOCH,
  "end_epoch": $END_EPOCH,
  "optimizer_state_resumed": false,
  "online_validation_enabled": false,
  "reason": "epoch-40 validation failed while allocating pinned host memory"
}
EOF
touch "$TASK_DIR/DONE"
