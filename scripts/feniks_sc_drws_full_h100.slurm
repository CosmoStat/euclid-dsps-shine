#!/bin/bash
#SBATCH --job-name=feniks_sc_drws_full
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-1%2

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:?CATALOG_DIR is required}"
RECOVERY_ROOT="${RECOVERY_ROOT:?RECOVERY_ROOT is required}"
MANIFEST_ROOT="${MANIFEST_ROOT:?MANIFEST_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
SEEDS=(260826 260827)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}"
SELECTED=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate"])' "$RECOVERY_ROOT/RWS_RECOVERY_PASS.json")
case "$SELECTED" in
  historical_4x128) CONFIG=configs/experiments/feniks_sc_drws_r27p5_historical.yaml ;;
  current_residual_6x256) CONFIG=configs/experiments/feniks_sc_drws_r27p5_current.yaml ;;
  *) echo "[sc-drws-full][error] invalid candidate: $SELECTED" >&2; exit 2 ;;
esac
OUT="$RECOVERY_ROOT/full/$SELECTED/seed_$SEED"
TRAIN="$OUT/train"
TRAIN_INDICES="$MANIFEST_ROOT/full_train_indices.npy"
VALIDATION_INDICES="$MANIFEST_ROOT/validation_indices.npy"
EVAL_INDICES="$MANIFEST_ROOT/final_validation_indices.npy"
TRAIN_CATALOG="$CATALOG_DIR/train.parquet"
TEST_CATALOG="$CATALOG_DIR/test.parquet"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
cd "$REPO_DIR"
mkdir -p "$MPLCONFIGDIR" "$CACHE_ROOT/jax" "$OUT"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
for path in "$RECOVERY_ROOT/RWS_RECOVERY_PASS.json" "$CONFIG" \
  "$MANIFEST_ROOT/manifest.json" "$TRAIN_INDICES" "$VALIDATION_INDICES" \
  "$EVAL_INDICES" "$TRAIN_CATALOG" "$TEST_CATALOG"; do
  test -s "$path" || { echo "[sc-drws-full][error] missing: $path" >&2; exit 2; }
done
test ! -e "$OUT/DONE" || { echo "already complete: $OUT"; exit 0; }
RESUME_ARGS=()
if [[ -s "$TRAIN/states/latest.eqx" ]]; then
  RESUME_ARGS+=(--resume-state "$TRAIN/states/latest.eqx")
  echo "[sc-drws-full] resuming $TRAIN/states/latest.eqx"
elif [[ -e "$TRAIN" ]]; then
  echo "[sc-drws-full][error] train directory exists without resumable state" >&2
  exit 2
fi
python scripts/train_feniks_sc_drws.py --config "$CONFIG" --out "$TRAIN" \
  --train-indices-file "$TRAIN_INDICES" \
  --validation-indices-file "$VALIDATION_INDICES" \
  --manifest-file "$MANIFEST_ROOT/manifest.json" --seed "$SEED" \
  --require-full-dataset "${RESUME_ARGS[@]}"

OBJECTS=$(python -c 'import numpy as np,sys; print(len(np.load(sys.argv[1])))' "$EVAL_INDICES")
for VARIANT in raw ema; do
  VARIANT_OFFSET=0
  [[ "$VARIANT" == "ema" ]] && VARIANT_OFFSET=1
  SUPPORT="$OUT/full_${VARIANT}_exact_gaussian_k2048"
  IMPORTANCE="$OUT/full_${VARIANT}_exact_gaussian_iw"
  PREDICTIVE="$OUT/full_${VARIANT}_exact_gaussian_predictive"
  COMMON=(--dataset "$TEST_CATALOG" --checkpoint "$TRAIN/checkpoints/final/${VARIANT}_model.eqx"
    --feature-stats "$TRAIN/feature_stats.json" --row-indices-file "$EVAL_INDICES"
    --selection-mode sequential --limit "$OBJECTS" --batch-size 64 --jax-batch-size 32
    --prior-samples 64 --decoder-sample-chunk-size 1 --prior-predictive-batch-size 64
    --shard-outputs --no-residual-samples)
  python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-diffsky \
    "${COMMON[@]}" --posterior-samples 2048 --seed "$((SEED + 7000 + VARIANT_OFFSET))" \
    --out "$SUPPORT" --no-posterior-predictive
  python -m euclid_dsps.cli --config "$CONFIG" amortized-finalize-inference \
    --out "$SUPPORT" --limit "$OBJECTS"
  python scripts/importance_correct_posterior.py --posterior "$SUPPORT" \
    --out "$IMPORTANCE" --resample-count 256 --seed "$((SEED + 8000))" \
    --min-median-ess-fraction 0.05 --max-fraction-pareto-k-gt-0p7 0.20
  python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-diffsky \
    "${COMMON[@]}" --posterior-samples 64 --seed "$((SEED + 9000 + VARIANT_OFFSET))" \
    --out "$PREDICTIVE"
  python -m euclid_dsps.cli --config "$CONFIG" amortized-finalize-inference \
    --out "$PREDICTIVE" --limit "$OBJECTS"
done
python scripts/evaluate_feniks_rws_recovery.py summarize --candidate "$SELECTED" \
  --seed "$SEED" --phase full --config "$CONFIG" --train "$TRAIN" \
  --importance "$OUT/full_raw_exact_gaussian_iw" \
  --predictive "$OUT/full_raw_exact_gaussian_predictive" \
  --ema-importance "$OUT/full_ema_exact_gaussian_iw" \
  --ema-predictive "$OUT/full_ema_exact_gaussian_predictive" --out "$OUT"
touch "$OUT/DONE"
