#!/bin/bash
#SBATCH --job-name=diffsky_nn_matrix
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail

if [[ -z "${REPO_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/euclid_dsps" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
  elif [[ -n "${WORK:-}" ]]; then
    REPO_DIR="$WORK/euclid-dsps-shine"
  else
    REPO_DIR="$PWD"
  fi
fi
if [[ -z "${MINICONDA_PATH:-}" ]]; then
  if [[ -n "${WORK:-}" && -d "$WORK/miniconda3" ]]; then
    MINICONDA_PATH="$WORK/miniconda3"
  else
    MINICONDA_PATH="$(dirname "$REPO_DIR")/miniconda3"
  fi
fi

CONDA_ENV="${CONDA_ENV:-shine}"
ROWSET_DIR="${ROWSET_DIR:-outputs/rowsets/diffsky_hltds_nokl_reference}"
ROWSET_FILE="${ROWSET_FILE:-${ROWSET_DIR}/balanced20k.txt}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/diffsky_nn_matrix}"
LOG_DIR="${LOG_DIR:-outputs/logs}"

BATCH_SIZE="${BATCH_SIZE:-128}"
JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-128}"
SEED="${SEED:-42}"
DECODER_SAMPLE_CHUNK_SIZE="${DECODER_SAMPLE_CHUNK_SIZE:-1}"
PRIOR_PREDICTIVE_BATCH_SIZE="${PRIOR_PREDICTIVE_BATCH_SIZE:-512}"
SHARD_OUTPUTS="${SHARD_OUTPUTS:-1}"

case "${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}" in
  0)
    EXPERIMENT="nokl_det"
    CONFIG="configs/experiments/diffsky_hltds_autoencoder_nokl_deterministic_h100.yaml"
    EPOCHS="${EPOCHS:-12}"
    N_SAMPLES="${N_SAMPLES:-1}"
    POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-1}"
    PRIOR_SAMPLES="${PRIOR_SAMPLES:-512}"
    ;;
  1)
    EXPERIMENT="nokl_stoch"
    CONFIG="configs/experiments/diffsky_hltds_autoencoder_nokl_h100.yaml"
    EPOCHS="${EPOCHS:-12}"
    N_SAMPLES="${N_SAMPLES:-1}"
    POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-64}"
    PRIOR_SAMPLES="${PRIOR_SAMPLES:-512}"
    ;;
  2)
    EXPERIMENT="kl_fixed"
    CONFIG="configs/experiments/diffsky_hltds_joint_realnvp_kl_fixed_h100.yaml"
    EPOCHS="${EPOCHS:-20}"
    N_SAMPLES="${N_SAMPLES:-4}"
    POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-128}"
    PRIOR_SAMPLES="${PRIOR_SAMPLES:-8192}"
    ;;
  3)
    EXPERIMENT="kl_annealed"
    CONFIG="configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_h100.yaml"
    EPOCHS="${EPOCHS:-20}"
    N_SAMPLES="${N_SAMPLES:-4}"
    POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-128}"
    PRIOR_SAMPLES="${PRIOR_SAMPLES:-8192}"
    ;;
  *)
    echo "[matrix][error] unsupported task id: ${SLURM_ARRAY_TASK_ID:-${TASK_ID:-unset}}"
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${EXPERIMENT}_balanced20k_seed${SEED}}"
TRAIN_OUT="${TRAIN_OUT:-${RUN_ROOT}/${RUN_NAME}}"
INFER_OUT="${INFER_OUT:-${RUN_ROOT}/${RUN_NAME}_infer}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/diffsky_nn_${RUN_NAME}_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}.log}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export REPO_DIR
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "$REPO_DIR"
mkdir -p "$LOG_DIR" "$(dirname "$TRAIN_OUT")" "$(dirname "$INFER_OUT")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[matrix] start: $(date --iso-8601=seconds)"
echo "[matrix] experiment: $EXPERIMENT"
echo "[matrix] config: $CONFIG"
echo "[matrix] rowset: $ROWSET_FILE"
echo "[matrix] train_out: $TRAIN_OUT"
echo "[matrix] infer_out: $INFER_OUT"
echo "[matrix] epochs: $EPOCHS"
echo "[matrix] n_samples: $N_SAMPLES"
echo "[matrix] posterior_samples: $POSTERIOR_SAMPLES"
echo "[matrix] prior_samples: $PRIOR_SAMPLES"
echo "[matrix] batch_size: $BATCH_SIZE"
echo "[matrix] jax_batch_size: $JAX_BATCH_SIZE"
echo "[matrix] seed: $SEED"

if [[ ! -s "$ROWSET_FILE" ]]; then
  cat <<MSG
[matrix][error] Missing rowset file: $ROWSET_FILE
[matrix][hint] Build rowsets first:
[matrix][hint]   sbatch scripts/diffsky_nn_build_rowsets_h100.slurm
MSG
  exit 2
fi

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python - <<'PY'
import os
from pathlib import Path

import euclid_dsps
import jax

repo = Path(os.environ["REPO_DIR"]).resolve()
pkg = Path(euclid_dsps.__file__).resolve()
if repo not in pkg.parents:
    raise SystemExit(f"[matrix][error] euclid_dsps imported from {pkg}, not {repo}")
print("[matrix] euclid_dsps:", pkg)
print("[matrix] jax_backend:", jax.default_backend())
print("[matrix] jax_devices:", jax.devices())
PY
nvidia-smi || true
git status --short --branch
git rev-parse HEAD

train_cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  amortized-train-diffsky
  --out "$TRAIN_OUT"
  --row-indices-file "$ROWSET_FILE"
  --batch-size "$BATCH_SIZE"
  --jax-batch-size "$JAX_BATCH_SIZE"
  --epochs "$EPOCHS"
  --n-samples "$N_SAMPLES"
  --seed "$SEED"
)

echo "[matrix] train command:"
printf ' %q' "${train_cmd[@]}"
echo
"${train_cmd[@]}"

CHECKPOINT="${CHECKPOINT:-${TRAIN_OUT}/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-${TRAIN_OUT}/feature_stats.json}"
for path in "$CHECKPOINT" "$FEATURE_STATS"; do
  if [[ ! -s "$path" ]]; then
    echo "[matrix][missing] $path"
    exit 2
  fi
done

infer_cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  amortized-infer-diffsky
  --checkpoint "$CHECKPOINT"
  --feature-stats "$FEATURE_STATS"
  --out "$INFER_OUT"
  --row-indices-file "$ROWSET_FILE"
  --batch-size "$BATCH_SIZE"
  --jax-batch-size "$JAX_BATCH_SIZE"
  --posterior-samples "$POSTERIOR_SAMPLES"
  --prior-samples "$PRIOR_SAMPLES"
  --decoder-sample-chunk-size "$DECODER_SAMPLE_CHUNK_SIZE"
  --prior-predictive-batch-size "$PRIOR_PREDICTIVE_BATCH_SIZE"
  --seed "$SEED"
  --no-posterior-predictive
  --no-residual-samples
)

if [[ "$SHARD_OUTPUTS" == "1" || "$SHARD_OUTPUTS" == "true" ]]; then
  infer_cmd+=(--shard-outputs)
fi

echo "[matrix] infer command:"
printf ' %q' "${infer_cmd[@]}"
echo
"${infer_cmd[@]}"

echo "[matrix] training summary: $TRAIN_OUT/training_summary.json"
echo "[matrix] inference summary: $INFER_OUT/inference_summary.json"
echo "[matrix] end: $(date --iso-8601=seconds)"
