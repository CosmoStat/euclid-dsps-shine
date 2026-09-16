#!/bin/bash
#SBATCH --job-name=feniks_qarch
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=16:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-5%6
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

CONFIGS=(
  configs/experiments/feniks_architecture_20k_current_realnvp.yaml
  configs/experiments/feniks_architecture_20k_set_realnvp.yaml
  configs/experiments/feniks_architecture_20k_set_autoregressive_spline.yaml
)
LABELS=(current_mlp_realnvp set_transformer_realnvp set_transformer_autoregressive_spline)
SEEDS=(260820 260821)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CANDIDATE_INDEX=$((TASK_ID % 3))
SEED_INDEX=$((TASK_ID / 3))
(( CANDIDATE_INDEX < 3 && SEED_INDEX < ${#SEEDS[@]} )) || {
  echo "[feniks-architecture][error] invalid task: $TASK_ID" >&2; exit 2;
}
CONFIG="${CONFIGS[$CANDIDATE_INDEX]}"
LABEL="${LABELS[$CANDIDATE_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
ARCHITECTURE_ROOT="${ARCHITECTURE_ROOT:?ARCHITECTURE_ROOT is required}"
MANIFEST_ROOT="${MANIFEST_ROOT:?MANIFEST_ROOT is required}"
SMOKE="${SMOKE:-0}"
OUT="$ARCHITECTURE_ROOT/$LABEL/seed_${SEED}"
TRAIN="$OUT/train"
PROPOSAL="$OUT/blind_iw_proposal"
IMPORTANCE="$OUT/blind_iw_k2048"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$OUT"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

TRAIN_CATALOG="$CATALOG_DIR/train.parquet"
TEST_CATALOG="$CATALOG_DIR/test.parquet"
TRAIN_INDICES="$MANIFEST_ROOT/train_indices.npy"
VALIDATION_INDICES="$MANIFEST_ROOT/validation_indices.npy"
PROBE_INDICES="$MANIFEST_ROOT/blind_iw_probe_indices.npy"
for path in "$CONFIG" "$TRAIN_CATALOG" "$TEST_CATALOG" \
  "$REFERENCE_CHECKPOINT" "${REFERENCE_CHECKPOINT}.json" \
  "$MANIFEST_ROOT/manifest.json" "$TRAIN_INDICES" "$VALIDATION_INDICES" \
  "$PROBE_INDICES"; do
  test -s "$path" || { echo "[feniks-architecture][error] missing: $path" >&2; exit 2; }
done
test ! -e "$OUT/DONE" || { echo "[feniks-architecture][error] already complete: $OUT" >&2; exit 2; }
test ! -e "$TRAIN" || { echo "[feniks-architecture][error] output exists: $TRAIN" >&2; exit 2; }

python - <<'PY'
import jax
print("[feniks-architecture] backend:", jax.default_backend(), flush=True)
print("[feniks-architecture] devices:", jax.devices(), flush=True)
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 4:
    raise SystemExit("expected exactly four local H100 devices")
PY

EPOCHS=80
POSTERIOR_SAMPLES=2048
JAX_BATCH_SIZE=256
PROBE_LIMIT=256
VALIDATION_EVERY=5
if [[ "$SMOKE" == "1" ]]; then
  EPOCHS=2
  POSTERIOR_SAMPLES=32
  JAX_BATCH_SIZE=128
  PROBE_LIMIT=8
  VALIDATION_EVERY=1
fi

echo "[feniks-architecture] task=$TASK_ID candidate=$LABEL seed=$SEED smoke=$SMOKE"
echo "[feniks-architecture] train_manifest=$TRAIN_INDICES validation_manifest=$VALIDATION_INDICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m euclid_dsps.cli --config "$CONFIG" amortized-train-diffsky \
  --dataset "$TRAIN_CATALOG" \
  --prior-checkpoint "$REFERENCE_CHECKPOINT" \
  --train-indices-file "$TRAIN_INDICES" \
  --validation-indices-file "$VALIDATION_INDICES" \
  --out "$TRAIN" \
  --batch-size 1024 \
  --jax-batch-size "$JAX_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --n-samples 1 \
  --seed "$SEED" \
  --data-parallel pmap \
  --validation-every "$VALIDATION_EVERY" \
  --no-progress

python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-diffsky \
  --dataset "$TEST_CATALOG" \
  --checkpoint "$TRAIN/checkpoints/best.eqx" \
  --prior-checkpoint "$REFERENCE_CHECKPOINT" \
  --feature-stats "$TRAIN/feature_stats.json" \
  --row-indices-file "$PROBE_INDICES" \
  --selection-mode sequential \
  --limit "$PROBE_LIMIT" \
  --batch-size 64 \
  --jax-batch-size 32 \
  --posterior-samples "$POSTERIOR_SAMPLES" \
  --prior-samples 256 \
  --decoder-sample-chunk-size 1 \
  --prior-predictive-batch-size 64 \
  --seed "$((SEED + 1000))" \
  --out "$PROPOSAL" \
  --shard-outputs \
  --no-posterior-predictive \
  --no-residual-samples

python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-inference --out "$PROPOSAL" --limit "$PROBE_LIMIT"

python scripts/importance_correct_posterior.py \
  --posterior "$PROPOSAL" \
  --out "$IMPORTANCE" \
  --resample-count "$([[ "$SMOKE" == "1" ]] && echo 16 || echo 256)" \
  --seed "$((SEED + 2000))" \
  --min-median-ess-fraction 0.05 \
  --max-fraction-pareto-k-gt-0p7 0.20

python - "$LABEL" "$SEED" "$CONFIG" "$TRAIN" "$IMPORTANCE" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

label, seed, config, train_arg, importance_arg, out_arg = sys.argv[1:]
train = Path(train_arg)
importance = Path(importance_arg)
out = Path(out_arg)
training = json.loads((train / "training_summary.json").read_text())
iw = json.loads((importance / "importance_summary.json").read_text())
payload = {
    "status": "complete",
    "candidate": label,
    "seed": int(seed),
    "config": config,
    "training_objects": int(training["train_rows"]),
    "validation_objects": int(training["validation_rows"]),
    "best_validation_objective": float(training["best_loss"]),
    "best_checkpoint_epoch": int(training["best_checkpoint_epoch"]),
    "training_elapsed_seconds": float(training["elapsed_time_s"]),
    "ordinary_iw_support": iw["support_gate"]["status"],
    "median_raw_ess_fraction": float(iw["median_raw_ess_fraction"]),
    "fraction_pareto_k_gt_0p7": float(iw["fraction_pareto_k_gt_0p7"]),
    "fraction_pareto_k_gt_1": float(iw["fraction_pareto_k_gt_1"]),
    "probe_objects": int(iw["n_objects"]),
    "joint_draws": int(iw["n_joint_draws"]),
    "truth_used_for_training_or_selection": False,
}
(out / "candidate_summary.json").write_text(json.dumps(payload, indent=2))
PY

touch "$OUT/DONE"
echo "[feniks-architecture] complete candidate=$LABEL seed=$SEED -> $OUT"
