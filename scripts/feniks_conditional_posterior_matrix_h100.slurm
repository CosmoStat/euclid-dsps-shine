#!/bin/bash
#SBATCH --job-name=feniks_cpost
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-7%8
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

CONFIGS=(
  configs/experiments/feniks_avi_gaussian_x.yaml
  configs/experiments/feniks_avi_gaussian_u.yaml
  configs/experiments/feniks_avi_realnvp.yaml
  configs/experiments/feniks_avi_rqspline.yaml
  configs/experiments/feniks_npe_gaussian_x.yaml
  configs/experiments/feniks_npe_gaussian_u.yaml
  configs/experiments/feniks_npe_realnvp.yaml
  configs/experiments/feniks_npe_rqspline.yaml
)
LABELS=(
  avi_gaussian_x
  avi_gaussian_u
  avi_realnvp
  avi_rqspline
  npe_gaussian_x
  npe_gaussian_u
  npe_realnvp
  npe_rqspline
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CONFIG="${CONFIGS[$TASK_ID]}"
LABEL="${LABELS[$TASK_ID]}"
SMOKE="${SMOKE:-0}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_DIR="${SOURCE_DIR:-Data/diffsky/synthetic/feniks_260617_dsps_closure_18band_grouped}"
SPLINE_DIR="${SPLINE_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1}"
CATALOG_DIR="${CATALOG_DIR:-${SPLINE_DIR}/amortized}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
ROOT_NAME="feniks_conditional_posterior_jaxcosmo_v1"
if [[ "$SMOKE" == "1" ]]; then
  ROOT_NAME="${ROOT_NAME}_smoke_${SLURM_ARRAY_JOB_ID}"
fi
ROOT_DIR="${ROOT_DIR:-outputs/runs/${ROOT_NAME}}"
TASK_DIR="${ROOT_DIR}/${LABEL}"
TRAIN_DIR="${TASK_DIR}/train"
INFER_DIR="${TASK_DIR}/inference"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$TASK_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

for path in "$CONFIG" "$CATALOG_DIR/train.parquet" "$CATALOG_DIR/test.parquet" \
  "$CATALOG_DIR/amortized_catalog_contract.json" "$PRIOR_CHECKPOINT"; do
  test -s "$path" || { echo "[matrix][error] missing: $path"; exit 2; }
done
python scripts/validate_feniks_conditional_posterior_inputs.py \
  --catalog-dir "$CATALOG_DIR" --prior-checkpoint "$PRIOR_CHECKPOINT"
test ! -e "$TRAIN_DIR" || { echo "[matrix][error] exists: $TRAIN_DIR"; exit 2; }
test ! -e "$INFER_DIR" || { echo "[matrix][error] exists: $INFER_DIR"; exit 2; }

python - <<'PY'
import jax
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(1, 1))
plt.close(fig)
print("[matrix] backend:", jax.default_backend())
print("[matrix] devices:", jax.devices())
if len(jax.local_devices()) != 4:
    raise SystemExit("[matrix][error] expected exactly four local H100 devices")
PY

if [[ "$LABEL" == npe_* ]]; then
  EPOCHS=200
  BATCH_SIZE=8192
  JAX_BATCH_SIZE=8192
else
  EPOCHS=120
  BATCH_SIZE=1024
  JAX_BATCH_SIZE=256
fi
LIMIT=""
POSTERIOR_SAMPLES=128
INFERENCE_LIMIT=5000
if [[ "$SMOKE" == "1" ]]; then
  EPOCHS=1
  BATCH_SIZE=256
  JAX_BATCH_SIZE=256
  LIMIT=512
  POSTERIOR_SAMPLES=4
  INFERENCE_LIMIT=8
fi

echo "[matrix] task=$TASK_ID label=$LABEL config=$CONFIG smoke=$SMOKE"
echo "[matrix] train epochs=$EPOCHS batch=$BATCH_SIZE jax_batch=$JAX_BATCH_SIZE"
nvidia-smi

train_cmd=(
  python -m euclid_dsps.cli --config "$CONFIG"
  amortized-train-diffsky
  --dataset "$CATALOG_DIR/train.parquet"
  --prior-checkpoint "$PRIOR_CHECKPOINT"
  --out "$TRAIN_DIR"
  --batch-size "$BATCH_SIZE"
  --jax-batch-size "$JAX_BATCH_SIZE"
  --epochs "$EPOCHS"
  --n-samples 1
  --data-parallel pmap
  --validation-every 1
  --no-progress
)
if [[ -n "$LIMIT" ]]; then train_cmd+=(--limit "$LIMIT"); fi
printf '[matrix] train command:'; printf ' %q' "${train_cmd[@]}"; echo
"${train_cmd[@]}"

infer_cmd=(
  python -m euclid_dsps.cli --config "$CONFIG"
  amortized-infer-diffsky
  --dataset "$CATALOG_DIR/test.parquet"
  --checkpoint "$TRAIN_DIR/checkpoints/best.eqx"
  --feature-stats "$TRAIN_DIR/feature_stats.json"
  --prior-checkpoint "$PRIOR_CHECKPOINT"
  --out "$INFER_DIR"
  --limit "$INFERENCE_LIMIT"
  --batch-size 512
  --jax-batch-size 512
  --posterior-samples "$POSTERIOR_SAMPLES"
  --prior-samples 8192
  --decoder-sample-chunk-size 1
  --prior-predictive-batch-size 256
  --seed 260717
  --selection-mode sequential
  --selection-seed 260717
  --shard-outputs
  --no-posterior-predictive
  --no-residual-samples
)
printf '[matrix] inference command:'; printf ' %q' "${infer_cmd[@]}"; echo
"${infer_cmd[@]}"

python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-inference --out "$INFER_DIR" --limit "$INFERENCE_LIMIT"

python scripts/summarize_feniks_conditional_posterior_task.py \
  --label "$LABEL" --config "$CONFIG" --train "$TRAIN_DIR" \
  --inference "$INFER_DIR" --out "$TASK_DIR/metrics.json"
touch "$TASK_DIR/DONE"

exec 9>"$ROOT_DIR/.aggregate.lock"
flock -x 9
python scripts/summarize_feniks_conditional_posterior_task.py \
  --aggregate-root "$ROOT_DIR" --expected 8
