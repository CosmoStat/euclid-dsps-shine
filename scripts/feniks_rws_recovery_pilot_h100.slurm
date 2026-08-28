#!/bin/bash
#SBATCH --job-name=feniks_rws_recovery
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3%4

set -Eeuo pipefail

TRAIN_CONFIGS=(
  configs/experiments/feniks_sc_drws_r29_historical.yaml
  configs/experiments/feniks_sc_drws_r29_current.yaml
)
CANDIDATES=(historical_4x128 current_residual_6x256)
SEEDS=(260826 260827)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CANDIDATE_INDEX=$((TASK_ID % 2))
SEED_INDEX=$((TASK_ID / 2))
TRAIN_CONFIG="${TRAIN_CONFIGS[$CANDIDATE_INDEX]}"
EVAL_CONFIG="$TRAIN_CONFIG"
CANDIDATE="${CANDIDATES[$CANDIDATE_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:?CATALOG_DIR is required}"
RECOVERY_ROOT="${RECOVERY_ROOT:?RECOVERY_ROOT is required}"
MANIFEST_ROOT="${MANIFEST_ROOT:?MANIFEST_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
SMOKE="${SMOKE:-0}"
OUT="$RECOVERY_ROOT/$CANDIDATE/seed_$SEED"
TRAIN="$OUT/train"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda
export EUCLID_DSPS_JAX_PLATFORMS=cuda
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1
export EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"

cd "$REPO_DIR"
mkdir -p "$MPLCONFIGDIR" "$OUT" "$CACHE_ROOT/jax"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

TRAIN_CATALOG="$CATALOG_DIR/train.parquet"
TEST_CATALOG="$CATALOG_DIR/test.parquet"
TRAIN_INDICES="$MANIFEST_ROOT/pilot_train_indices.npy"
VALIDATION_INDICES="$MANIFEST_ROOT/validation_indices.npy"
PILOT_INDICES="$MANIFEST_ROOT/pilot_indices.npy"
for path in "$TRAIN_CONFIG" "$EVAL_CONFIG" "$TRAIN_CATALOG" "$TEST_CATALOG" \
  "$MANIFEST_ROOT/manifest.json" "$TRAIN_INDICES" "$VALIDATION_INDICES" \
  "$PILOT_INDICES"; do
  test -s "$path" || { echo "[rws-recovery][error] missing: $path" >&2; exit 2; }
done
test ! -e "$OUT/DONE" || { echo "[rws-recovery][error] already complete: $OUT"; exit 2; }
RESUME_ARGS=()
TRAINING_COMPLETE=0
if [[ -s "$TRAIN/training_receipt.json" ]] && python - "$TRAIN" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipt = json.loads((root / "training_receipt.json").read_text())
required = (
    root / "feature_stats.json",
    root / "checkpoints/final/raw_model.eqx",
    root / "checkpoints/final/ema_model.eqx",
    root / "checkpoints/final/raw_q.eqx",
    root / "checkpoints/final/ema_q.eqx",
    root / "checkpoints/final/prior.eqx",
)
valid = (
    receipt.get("status") == "TRAINING_COMPLETE_PENDING_SUPPORT_SELECTION"
    and receipt.get("truth_used_for_training_validation_or_checkpoint_selection") is False
    and int(receipt.get("wake_updates", 0)) > 0
    and all(path.is_file() and path.stat().st_size > 0 for path in required)
)
raise SystemExit(0 if valid else 1)
PY
then
  TRAINING_COMPLETE=1
  echo "[sc-drws] reusing completed truth-free training: $TRAIN"
elif [[ -s "$TRAIN/states/latest.eqx" ]]; then
  RESUME_ARGS+=(--resume-state "$TRAIN/states/latest.eqx")
  echo "[sc-drws] resuming $TRAIN/states/latest.eqx"
elif [[ -e "$TRAIN" ]]; then
  echo "[rws-recovery][error] train output exists without resumable state: $TRAIN" >&2
  exit 2
fi

python - <<'PY'
import jax
print("[rws-recovery] backend:", jax.default_backend(), flush=True)
print("[rws-recovery] devices:", jax.devices(), flush=True)
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 4:
    raise SystemExit("expected exactly four local H100 devices")
PY

SUPPORT_SAMPLES=2048
PREDICTIVE_SAMPLES=64
if [[ "$SMOKE" == "1" ]]; then
  SUPPORT_SAMPLES=32
  PREDICTIVE_SAMPLES=4
fi
OBJECTS=$(python -c 'import numpy as np,sys; print(len(np.load(sys.argv[1])))' "$PILOT_INDICES")
echo "[sc-drws] candidate=$CANDIDATE seed=$SEED smoke=$SMOKE"
echo "[rws-recovery] selected_train=$(python -c 'import numpy as np,sys; print(len(np.load(sys.argv[1])))' "$TRAIN_INDICES") pilot=$OBJECTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TRAIN_ARGS=(
  --config "$TRAIN_CONFIG"
  --out "$TRAIN"
  --train-indices-file "$TRAIN_INDICES" \
  --validation-indices-file "$VALIDATION_INDICES" \
  --manifest-file "$MANIFEST_ROOT/manifest.json"
  --seed "$SEED"
  "${RESUME_ARGS[@]}"
)
[[ "$SMOKE" == "1" ]] && TRAIN_ARGS+=(--smoke)
if [[ "$TRAINING_COMPLETE" == "0" ]]; then
  python scripts/train_feniks_sc_drws.py "${TRAIN_ARGS[@]}"
fi

inference_artifact_complete() {
  python - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2]).resolve()
samples = int(sys.argv[3])
objects = int(sys.argv[4])
predictive = sys.argv[5] == "1"
summary_path = root / "inference_summary.json"
if not summary_path.is_file():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text())
valid = (
    summary.get("complete") is True
    and Path(str(summary.get("checkpoint", ""))).resolve() == checkpoint
    and int(summary.get("posterior_samples", -1)) == samples
    and int(summary.get("n_processed", -1)) == objects
    and summary.get("write_posterior_predictive") is predictive
    and summary.get("truth_snapshot_enabled") is False
    and summary.get("truth_diagnostics_enabled") is False
    and summary.get("truth_used_for_inference_or_checkpoint_selection") is False
    and int(summary.get("truth_snapshot_rows", -1)) == 0
    and not (root / "inference_truth.parquet").exists()
)
raise SystemExit(0 if valid else 1)
PY
}

importance_artifact_complete() {
  python - "$1" "$2" "$3" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
objects = int(sys.argv[2])
draws = int(sys.argv[3])
summary_path = root / "importance_summary.json"
done_path = root / "DONE"
if not summary_path.is_file() or not done_path.is_file():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text())
done = json.loads(done_path.read_text())
valid = (
    summary.get("status") == "complete"
    and done.get("status") == "complete"
    and int(summary.get("n_objects", -1)) == objects
    and int(summary.get("n_joint_draws", -1)) == draws
)
raise SystemExit(0 if valid else 1)
PY
}

for VARIANT in raw ema; do
  VARIANT_OFFSET=0
  [[ "$VARIANT" == "ema" ]] && VARIANT_OFFSET=1
  CHECKPOINT="$TRAIN/checkpoints/final/${VARIANT}_model.eqx"
  SUPPORT="$OUT/pilot_${VARIANT}_exact_gaussian_k2048"
  IMPORTANCE="$OUT/pilot_${VARIANT}_exact_gaussian_iw"
  PREDICTIVE="$OUT/pilot_${VARIANT}_exact_gaussian_predictive"
  COMMON_INFER=(
  --dataset "$TEST_CATALOG"
  --checkpoint "$CHECKPOINT"
  --feature-stats "$TRAIN/feature_stats.json"
  --row-indices-file "$PILOT_INDICES"
  --selection-mode sequential
  --limit "$OBJECTS"
  --batch-size 64
  --jax-batch-size 32
  --prior-samples 64
  --decoder-sample-chunk-size 1
  --prior-predictive-batch-size 64
  --shard-outputs
  --no-residual-samples
  )

  if inference_artifact_complete \
    "$SUPPORT" "$CHECKPOINT" "$SUPPORT_SAMPLES" "$OBJECTS" 0 \
    && importance_artifact_complete \
      "$IMPORTANCE" "$OBJECTS" "$((OBJECTS * SUPPORT_SAMPLES))"; then
    echo "[sc-drws] reusing complete ${VARIANT} ordinary-IW artifact: $IMPORTANCE"
  else
    if inference_artifact_complete \
      "$SUPPORT" "$CHECKPOINT" "$SUPPORT_SAMPLES" "$OBJECTS" 0; then
      echo "[sc-drws] reusing complete truth-free ${VARIANT} support inference: $SUPPORT"
    else
      python -m euclid_dsps.cli --config "$EVAL_CONFIG" amortized-infer-diffsky \
        "${COMMON_INFER[@]}" --posterior-samples "$SUPPORT_SAMPLES" \
        --seed "$((SEED + 1000 + VARIANT_OFFSET))" --out "$SUPPORT" --no-posterior-predictive
      python -m euclid_dsps.cli --config "$EVAL_CONFIG" \
        amortized-finalize-inference --out "$SUPPORT" --limit "$OBJECTS"
    fi

    python scripts/importance_correct_posterior.py \
      --posterior "$SUPPORT" --out "$IMPORTANCE" \
      --resample-count "$([[ "$SMOKE" == "1" ]] && echo 16 || echo 256)" \
      --seed "$((SEED + 2000))" --min-median-ess-fraction 0.05 \
      --max-fraction-pareto-k-gt-0p7 0.20
  fi

  if inference_artifact_complete \
    "$PREDICTIVE" "$CHECKPOINT" "$PREDICTIVE_SAMPLES" "$OBJECTS" 1; then
    echo "[sc-drws] reusing complete truth-free ${VARIANT} predictive inference: $PREDICTIVE"
  else
    python -m euclid_dsps.cli --config "$EVAL_CONFIG" amortized-infer-diffsky \
      "${COMMON_INFER[@]}" --posterior-samples "$PREDICTIVE_SAMPLES" \
      --seed "$((SEED + 3000))" --out "$PREDICTIVE"
    python -m euclid_dsps.cli --config "$EVAL_CONFIG" \
      amortized-finalize-inference --out "$PREDICTIVE" --limit "$OBJECTS"
  fi
done

SUMMARY_ARGS=(
  --candidate "$CANDIDATE"
  --seed "$SEED"
  --phase pilot
  --config "$EVAL_CONFIG"
  --train "$TRAIN"
  --importance "$OUT/pilot_raw_exact_gaussian_iw"
  --predictive "$OUT/pilot_raw_exact_gaussian_predictive"
  --ema-importance "$OUT/pilot_ema_exact_gaussian_iw"
  --ema-predictive "$OUT/pilot_ema_exact_gaussian_predictive"
  --out "$OUT"
)
[[ "$SMOKE" == "1" ]] && SUMMARY_ARGS+=(--smoke)
python scripts/evaluate_feniks_rws_recovery.py summarize "${SUMMARY_ARGS[@]}"
if [[ "$SMOKE" == "1" ]]; then
  python - "$OUT/pilot_summary.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
if value.get("status") != "SMOKE_PASS":
    raise SystemExit("SC-DRWS smoke gate failed")
PY
fi
touch "$OUT/DONE"
echo "[rws-recovery] complete -> $OUT"
