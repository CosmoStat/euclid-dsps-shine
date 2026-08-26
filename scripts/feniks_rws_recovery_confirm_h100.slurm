#!/bin/bash
#SBATCH --job-name=feniks_rws_confirm
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
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
SELECTED=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate"])' "$RECOVERY_ROOT/PILOT_PASS.json")
case "$SELECTED" in
  historical_4x128) EVAL_CONFIG=configs/experiments/feniks_rws_recovery_r25_historical_gaussian_eval.yaml ;;
  current_residual_6x256) EVAL_CONFIG=configs/experiments/feniks_rws_recovery_r25_current_gaussian_eval.yaml ;;
  *) echo "[rws-confirm][error] invalid selected candidate: $SELECTED" >&2; exit 2 ;;
esac
OUT="$RECOVERY_ROOT/$SELECTED/seed_$SEED"
TRAIN="$OUT/train"
SUPPORT="$OUT/confirmation_exact_gaussian_k2048"
IMPORTANCE="$OUT/confirmation_exact_gaussian_iw"
PREDICTIVE="$OUT/confirmation_exact_gaussian_predictive"
INDICES="$MANIFEST_ROOT/confirmation_indices.npy"
TEST_CATALOG="$CATALOG_DIR/test.parquet"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
cd "$REPO_DIR"
mkdir -p "$MPLCONFIGDIR" "$CACHE_ROOT/jax"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
for path in "$RECOVERY_ROOT/PILOT_PASS.json" "$EVAL_CONFIG" "$TEST_CATALOG" \
  "$INDICES" "$TRAIN/checkpoints/best.eqx" "$TRAIN/feature_stats.json"; do
  test -s "$path" || { echo "[rws-confirm][error] missing: $path" >&2; exit 2; }
done
test ! -e "$OUT/confirmation_summary.json" || { echo "confirmation already exists"; exit 2; }
OBJECTS=$(python -c 'import numpy as np,sys; print(len(np.load(sys.argv[1])))' "$INDICES")
COMMON_INFER=(
  --dataset "$TEST_CATALOG" --checkpoint "$TRAIN/checkpoints/best.eqx"
  --feature-stats "$TRAIN/feature_stats.json" --row-indices-file "$INDICES"
  --selection-mode sequential --limit "$OBJECTS" --batch-size 64
  --jax-batch-size 32 --prior-samples 64 --decoder-sample-chunk-size 1
  --prior-predictive-batch-size 64 --shard-outputs --no-residual-samples
)
echo "[rws-confirm] candidate=$SELECTED seed=$SEED objects=$OBJECTS"
python -m euclid_dsps.cli --config "$EVAL_CONFIG" amortized-infer-diffsky \
  "${COMMON_INFER[@]}" --posterior-samples 2048 --seed "$((SEED + 4000))" \
  --out "$SUPPORT" --no-posterior-predictive
python -m euclid_dsps.cli --config "$EVAL_CONFIG" amortized-finalize-inference \
  --out "$SUPPORT" --limit "$OBJECTS"
python scripts/importance_correct_posterior.py --posterior "$SUPPORT" \
  --out "$IMPORTANCE" --resample-count 256 --seed "$((SEED + 5000))" \
  --min-median-ess-fraction 0.05 --max-fraction-pareto-k-gt-0p7 0.20
python -m euclid_dsps.cli --config "$EVAL_CONFIG" amortized-infer-diffsky \
  "${COMMON_INFER[@]}" --posterior-samples 64 --seed "$((SEED + 6000))" \
  --out "$PREDICTIVE"
python -m euclid_dsps.cli --config "$EVAL_CONFIG" amortized-finalize-inference \
  --out "$PREDICTIVE" --limit "$OBJECTS"
python scripts/evaluate_feniks_rws_recovery.py summarize \
  --candidate "$SELECTED" --seed "$SEED" --phase confirmation \
  --config "$EVAL_CONFIG" --train "$TRAIN" --importance "$IMPORTANCE" \
  --predictive "$PREDICTIVE" --out "$OUT"
