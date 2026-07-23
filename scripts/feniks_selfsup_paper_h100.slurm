#!/bin/bash
#SBATCH --job-name=feniks_paper
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=16:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3%4
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

CONFIGS=(
  configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml
  configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed3.yaml
  configs/experiments/feniks_selfsup_paper_fixed_prior_rws_k8_t2.yaml
  configs/experiments/feniks_selfsup_paper_avi_joint_t2.yaml
)
LABELS=(
  rws_k8_t2_seed2
  rws_k8_t2_seed3
  fixed_prior_rws_k8_t2
  avi_joint_t2
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CONFIG="${CONFIGS[$TASK_ID]}"
LABEL="${LABELS[$TASK_ID]}"
SMOKE="${SMOKE:-0}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
FULL_ROOT_DIR="${FULL_ROOT_DIR:-outputs/runs/feniks_selfsup_paper_v1}"
if [[ "$SMOKE" == "1" ]]; then
  ROOT_DIR="${FULL_ROOT_DIR}_smoke_${SLURM_ARRAY_JOB_ID}"
else
  ROOT_DIR="$FULL_ROOT_DIR"
fi
TASK_DIR="$ROOT_DIR/$LABEL"
TRAIN_DIR="$TASK_DIR/train"
INFER_DIR="$TASK_DIR/inference"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MALLOC_TRIM_THRESHOLD_=65536
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$TASK_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

for path in "$CONFIG" "$CATALOG_DIR/train.parquet" "$CATALOG_DIR/test.parquet" \
  "$CATALOG_DIR/amortized_catalog_contract.json" "$REFERENCE_CHECKPOINT" \
  "${REFERENCE_CHECKPOINT}.json"; do
  test -s "$path" || { echo "[paper][error] missing: $path"; exit 2; }
done
validate_cmd=(
  python scripts/validate_feniks_selfsup_paper_inputs.py
  --catalog-dir "$CATALOG_DIR"
  --reference-checkpoint "$REFERENCE_CHECKPOINT"
)
for item in "${CONFIGS[@]}"; do validate_cmd+=(--config "$item"); done
"${validate_cmd[@]}"
test ! -e "$TRAIN_DIR" || { echo "[paper][error] exists: $TRAIN_DIR"; exit 2; }
test ! -e "$INFER_DIR" || { echo "[paper][error] exists: $INFER_DIR"; exit 2; }

python - <<'PY'
import jax
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
plt.close(plt.figure(figsize=(1, 1)))
print("[paper] backend:", jax.default_backend())
print("[paper] devices:", jax.devices())
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 4:
    raise SystemExit("[paper][error] expected exactly four local H100 devices")
PY

EPOCHS=120
LIMIT=""
POSTERIOR_SAMPLES=128
PRIOR_SAMPLES=16384
INFERENCE_LIMIT=5000
VALIDATION_EVERY=8
JAX_BATCH_SIZE=256
EXTRA_TRAIN_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
  EPOCHS=2
  LIMIT=256
  POSTERIOR_SAMPLES=8
  PRIOR_SAMPLES=512
  INFERENCE_LIMIT=8
  VALIDATION_EVERY=1
  JAX_BATCH_SIZE=128
  if (( TASK_ID < 3 )); then
    EXTRA_TRAIN_ARGS+=(--wake-start-encoder-epoch 2 --wake-every-encoder-epochs 1)
  fi
fi

echo "[paper] task=$TASK_ID label=$LABEL config=$CONFIG smoke=$SMOKE"
nvidia-smi
train_cmd=(
  python -m euclid_dsps.cli --config "$CONFIG"
  amortized-train-diffsky
  --dataset "$CATALOG_DIR/train.parquet"
  --out "$TRAIN_DIR"
  --batch-size 1024
  --jax-batch-size "$JAX_BATCH_SIZE"
  --epochs "$EPOCHS"
  --n-samples 1
  --data-parallel pmap
  --validation-every "$VALIDATION_EVERY"
  --no-progress
  "${EXTRA_TRAIN_ARGS[@]}"
)
if [[ -n "$LIMIT" ]]; then train_cmd+=(--limit "$LIMIT"); fi
printf '[paper] train command:'; printf ' %q' "${train_cmd[@]}"; echo
"${train_cmd[@]}"

infer_cmd=(
  python -m euclid_dsps.cli --config "$CONFIG"
  amortized-infer-diffsky
  --dataset "$CATALOG_DIR/test.parquet"
  --checkpoint "$TRAIN_DIR/checkpoints/best.eqx"
  --feature-stats "$TRAIN_DIR/feature_stats.json"
  --out "$INFER_DIR"
  --limit "$INFERENCE_LIMIT"
  --batch-size 512
  --jax-batch-size 512
  --posterior-samples "$POSTERIOR_SAMPLES"
  --prior-samples "$PRIOR_SAMPLES"
  --decoder-sample-chunk-size 1
  --prior-predictive-batch-size 256
  --seed "$((260730 + TASK_ID))"
  --selection-mode stratified_redshift
  --stratified-strategy proportional
  --selection-seed 260722
  --shard-outputs
  --no-residual-samples
)
printf '[paper] inference command:'; printf ' %q' "${infer_cmd[@]}"; echo
"${infer_cmd[@]}"

python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-inference --out "$INFER_DIR" --limit "$INFERENCE_LIMIT"
python scripts/summarize_feniks_selfsup_paper.py \
  --label "$LABEL" --config "$CONFIG" --train "$TRAIN_DIR" \
  --inference "$INFER_DIR" --out "$TASK_DIR/metrics.json"
touch "$TASK_DIR/DONE"
