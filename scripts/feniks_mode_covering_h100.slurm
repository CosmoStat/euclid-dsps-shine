#!/bin/bash
#SBATCH --job-name=feniks_mode
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3%4
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

CONFIGS=(
  configs/experiments/feniks_mode_common15d_vem4_elbo_k1.yaml
  configs/experiments/feniks_mode_frozen_ref_elbo_k2_antithetic.yaml
  configs/experiments/feniks_mode_frozen_ref_periodic_wake_k4.yaml
  configs/experiments/feniks_mode_common15d_vem4_periodic_wake_k4.yaml
)
LABELS=(
  common15d_vem4_elbo_k1
  frozen_ref_elbo_k2_antithetic
  frozen_ref_periodic_wake_k4
  common15d_vem4_periodic_wake_k4
)
FULL_EPOCHS=(150 120 120 150)
TRAIN_SAMPLES=(1 2 1 1)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CONFIG="${CONFIGS[$TASK_ID]}"
LABEL="${LABELS[$TASK_ID]}"
EPOCHS="${FULL_EPOCHS[$TASK_ID]}"
N_SAMPLES="${TRAIN_SAMPLES[$TASK_ID]}"
SMOKE="${SMOKE:-0}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SPLINE_DIR="${SPLINE_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1}"
CATALOG_DIR="${CATALOG_DIR:-${SPLINE_DIR}/amortized}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
DEFAULT_FULL_ROOT="outputs/runs/feniks_mode_covering_common15d_jaxcosmo_v1"
FULL_ROOT_DIR="${FULL_ROOT_DIR:-$DEFAULT_FULL_ROOT}"
if [[ "$SMOKE" == "1" ]]; then
  TASK_ROOT_DIR="${FULL_ROOT_DIR}_smoke_${SLURM_ARRAY_JOB_ID}"
else
  TASK_ROOT_DIR="$FULL_ROOT_DIR"
fi
ROOT_DIR="$TASK_ROOT_DIR"
TASK_DIR="${ROOT_DIR}/${LABEL}"
TRAIN_DIR="${TASK_DIR}/train"
INFER_DIR="${TASK_DIR}/inference"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MALLOC_TRIM_THRESHOLD_=65536
export MPLBACKEND=Agg
export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$TASK_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

for path in "$CONFIG" "$CATALOG_DIR/train.parquet" "$CATALOG_DIR/test.parquet" \
  "$CATALOG_DIR/amortized_catalog_contract.json" "$REFERENCE_CHECKPOINT" \
  "${REFERENCE_CHECKPOINT}.json"; do
  test -s "$path" || { echo "[mode-covering][error] missing: $path"; exit 2; }
done
python scripts/validate_feniks_mode_covering_inputs.py \
  --catalog-dir "$CATALOG_DIR" \
  --reference-checkpoint "$REFERENCE_CHECKPOINT" \
  --config "${CONFIGS[0]}" --config "${CONFIGS[1]}" \
  --config "${CONFIGS[2]}" --config "${CONFIGS[3]}"
test ! -e "$TRAIN_DIR" || { echo "[mode-covering][error] exists: $TRAIN_DIR"; exit 2; }
test ! -e "$INFER_DIR" || { echo "[mode-covering][error] exists: $INFER_DIR"; exit 2; }

python - <<'PY'
import jax
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(1, 1))
plt.close(fig)
print("[mode-covering] backend:", jax.default_backend())
print("[mode-covering] devices:", jax.devices())
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 4:
    raise SystemExit("[mode-covering][error] expected exactly four local GPU devices")
PY

LIMIT=""
POSTERIOR_SAMPLES=128
INFERENCE_LIMIT=5000
JAX_BATCH_SIZE=512
VALIDATION_EVERY=4
EXTRA_TRAIN_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
  EPOCHS=2
  LIMIT=512
  POSTERIOR_SAMPLES=4
  INFERENCE_LIMIT=8
  JAX_BATCH_SIZE=256
  VALIDATION_EVERY=1
  if [[ "$LABEL" == common15d_vem4_* ]]; then
    EXTRA_TRAIN_ARGS+=(--prior-update-every-epochs 1)
  fi
  if [[ "$LABEL" == *periodic_wake* ]]; then
    EXTRA_TRAIN_ARGS+=(--wake-start-encoder-epoch 1 --wake-every-encoder-epochs 1)
  fi
fi

echo "[mode-covering] task=$TASK_ID label=$LABEL config=$CONFIG smoke=$SMOKE"
echo "[mode-covering] epochs=$EPOCHS n_samples=$N_SAMPLES catalog_batch=1024 jax_batch=$JAX_BATCH_SIZE"
nvidia-smi

train_cmd=(
  python -m euclid_dsps.cli --config "$CONFIG"
  amortized-train-diffsky
  --dataset "$CATALOG_DIR/train.parquet"
  --out "$TRAIN_DIR"
  --batch-size 1024
  --jax-batch-size "$JAX_BATCH_SIZE"
  --epochs "$EPOCHS"
  --n-samples "$N_SAMPLES"
  --data-parallel pmap
  --validation-every "$VALIDATION_EVERY"
  --no-progress
  "${EXTRA_TRAIN_ARGS[@]}"
)
if [[ -n "$LIMIT" ]]; then train_cmd+=(--limit "$LIMIT"); fi
printf '[mode-covering] train command:'; printf ' %q' "${train_cmd[@]}"; echo
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
  --prior-samples 8192
  --decoder-sample-chunk-size 1
  --prior-predictive-batch-size 256
  --seed 260720
  --selection-mode sequential
  --selection-seed 260720
  --shard-outputs
  --no-posterior-predictive
  --no-residual-samples
)
printf '[mode-covering] inference command:'; printf ' %q' "${infer_cmd[@]}"; echo
"${infer_cmd[@]}"

python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-inference --out "$INFER_DIR" --limit "$INFERENCE_LIMIT"

python scripts/summarize_feniks_mode_covering_task.py \
  --label "$LABEL" --config "$CONFIG" --train "$TRAIN_DIR" \
  --inference "$INFER_DIR" --out "$TASK_DIR/metrics.json"
touch "$TASK_DIR/DONE"

exec 9>"$ROOT_DIR/.aggregate.lock"
flock -x 9
python scripts/summarize_feniks_mode_covering_task.py \
  --aggregate-root "$ROOT_DIR" --expected 4
