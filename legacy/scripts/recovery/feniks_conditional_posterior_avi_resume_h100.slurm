#!/bin/bash
#SBATCH --job-name=feniks_avi_resume
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3%4
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

CONFIGS=(
  configs/experiments/feniks_avi_gaussian_x.yaml
  configs/experiments/feniks_avi_gaussian_u.yaml
  configs/experiments/feniks_avi_realnvp.yaml
  configs/experiments/feniks_avi_rqspline.yaml
)
LABELS=(avi_gaussian_x avi_gaussian_u avi_realnvp avi_rqspline)
START_EPOCHS=(92 92 92 91)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CONFIG="${CONFIGS[$TASK_ID]}"
LABEL="${LABELS[$TASK_ID]}"
START_EPOCH="${START_EPOCHS[$TASK_ID]}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SPLINE_DIR="${SPLINE_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1}"
CATALOG_DIR="${CATALOG_DIR:-${SPLINE_DIR}/amortized}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/runs/feniks_conditional_posterior_jaxcosmo_v1}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_conditional_posterior_jaxcosmo_v1_completed_v2}"
SOURCE_TRAIN="${SOURCE_ROOT}/${LABEL}/train"
INITIAL_CHECKPOINT="${SOURCE_TRAIN}/checkpoints/last.eqx"
TASK_DIR="${ROOT_DIR}/${LABEL}"
TRAIN_DIR="${TASK_DIR}/train"
INFER_DIR="${TASK_DIR}/inference"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$TASK_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

for path in "$CONFIG" "$CATALOG_DIR/train.parquet" "$CATALOG_DIR/test.parquet" \
  "$CATALOG_DIR/amortized_catalog_contract.json" "$PRIOR_CHECKPOINT" \
  "$INITIAL_CHECKPOINT" "$INITIAL_CHECKPOINT.json" "$SOURCE_TRAIN/feature_stats.json"; do
  test -s "$path" || { echo "[resume][error] missing: $path"; exit 2; }
done
test ! -e "$TRAIN_DIR" || { echo "[resume][error] exists: $TRAIN_DIR"; exit 2; }
test ! -e "$INFER_DIR" || { echo "[resume][error] exists: $INFER_DIR"; exit 2; }

python scripts/validate_feniks_conditional_posterior_inputs.py \
  --catalog-dir "$CATALOG_DIR" --prior-checkpoint "$PRIOR_CHECKPOINT"
python - "$INITIAL_CHECKPOINT.json" "$((START_EPOCH - 1))" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
epoch = int(json.load(open(path, encoding="utf-8"))["epoch"])
if epoch != expected:
    raise SystemExit(
        f"[resume][error] checkpoint epoch mismatch: got {epoch}, expected {expected}"
    )
print(f"[resume] checkpoint contract: epoch={epoch}")
PY
python - <<'PY'
import jax
import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(1, 1))
plt.close(fig)
print("[resume] backend:", jax.default_backend())
print("[resume] devices:", jax.devices())
if len(jax.local_devices()) != 4:
    raise SystemExit("[resume][error] expected exactly four local H100 devices")
PY

echo "[resume] task=$TASK_ID label=$LABEL start_epoch=$START_EPOCH"
nvidia-smi

train_cmd=(
  python -m euclid_dsps.cli --config "$CONFIG"
  amortized-train-diffsky
  --dataset "$CATALOG_DIR/train.parquet"
  --prior-checkpoint "$PRIOR_CHECKPOINT"
  --initial-checkpoint "$INITIAL_CHECKPOINT"
  --start-epoch "$START_EPOCH"
  --out "$TRAIN_DIR"
  --batch-size 1024
  --jax-batch-size 256
  --epochs 120
  --n-samples 1
  --data-parallel pmap
  --validation-every 4
  --no-progress
)
printf '[resume] train command:'; printf ' %q' "${train_cmd[@]}"; echo
"${train_cmd[@]}"

infer_cmd=(
  python -m euclid_dsps.cli --config "$CONFIG"
  amortized-infer-diffsky
  --dataset "$CATALOG_DIR/test.parquet"
  --checkpoint "$TRAIN_DIR/checkpoints/best.eqx"
  --feature-stats "$TRAIN_DIR/feature_stats.json"
  --prior-checkpoint "$PRIOR_CHECKPOINT"
  --out "$INFER_DIR"
  --limit 5000
  --batch-size 512
  --jax-batch-size 512
  --posterior-samples 128
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
printf '[resume] inference command:'; printf ' %q' "${infer_cmd[@]}"; echo
"${infer_cmd[@]}"

python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-inference --out "$INFER_DIR" --limit 5000
python scripts/summarize_feniks_conditional_posterior_task.py \
  --label "$LABEL" --config "$CONFIG" --train "$TRAIN_DIR" \
  --inference "$INFER_DIR" --out "$TASK_DIR/metrics.json"
touch "$TASK_DIR/DONE"

exec 9>"$ROOT_DIR/.aggregate.lock"
flock -x 9
python scripts/summarize_feniks_conditional_posterior_task.py \
  --aggregate-root "$ROOT_DIR" --expected 8
