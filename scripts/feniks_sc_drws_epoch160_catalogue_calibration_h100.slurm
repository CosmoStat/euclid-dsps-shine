#!/bin/bash
#SBATCH --job-name=feniks_sc_e160_cal
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_EVAL_ROOT="${SOURCE_EVAL_ROOT:?SOURCE_EVAL_ROOT is required}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:?CALIBRATION_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
NUM_REGIONS="${NUM_REGIONS:-100}"
NUM_BOOTSTRAP="${NUM_BOOTSTRAP:-1000}"

case "$TASK" in
  0) MODE=common32; DIAGNOSTIC=mira; DRAWS=32 ;;
  1) MODE=common32; DIAGNOSTIC=tarp; DRAWS=32 ;;
  2) MODE=q256; DIAGNOSTIC=mira; DRAWS=256 ;;
  3) MODE=q256; DIAGNOSTIC=tarp; DRAWS=256 ;;
  *) echo "[catalogue-calibration][error] invalid task: $TASK" >&2; exit 2 ;;
esac

TRUTH="$SOURCE_EVAL_ROOT/population/catalogue_selected_truth.parquet"
DESTINATION="$CALIBRATION_ROOT/$MODE/$DIAGNOSTIC"
STAGING="$CALIBRATION_ROOT/.staging/task_${TASK}-${SLURM_JOB_ID}"
RAW_Q32="$SOURCE_EVAL_ROOT/population/posterior_aggregate/raw_q.parquet"
RAW_IW32="$SOURCE_EVAL_ROOT/population/posterior_aggregate/raw_iw.parquet"
EMA_Q32="$SOURCE_EVAL_ROOT/population/posterior_aggregate/ema_q.parquet"
EMA_IW32="$SOURCE_EVAL_ROOT/population/posterior_aggregate/ema_iw.parquet"
RAW_Q256="$CALIBRATION_ROOT/banks/raw_q256"
EMA_Q256="$CALIBRATION_ROOT/banks/ema_q256"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=false XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1 EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"

cd "$REPO_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$CALIBRATION_ROOT/.staging" "$MPLCONFIGDIR" "$CACHE_ROOT/jax"

for path in "$TRUTH" "$CALIBRATION_ROOT/CATALOGUE_CALIBRATION_MANIFEST.json"; do
  test -s "$path" || { echo "[catalogue-calibration][error] missing: $path" >&2; exit 2; }
done
if [[ -e "$DESTINATION/DONE" ]]; then
  echo "[catalogue-calibration] already complete: $DESTINATION"
  exit 0
fi
test ! -e "$DESTINATION" || {
  echo "[catalogue-calibration][error] receipt-less output exists: $DESTINATION" >&2
  exit 2
}
test ! -e "$STAGING" || {
  echo "[catalogue-calibration][error] staging output exists: $STAGING" >&2
  exit 2
}

POSTERIORS=()
if [[ "$MODE" == common32 ]]; then
  for path in "$RAW_Q32" "$RAW_IW32" "$EMA_Q32" "$EMA_IW32"; do
    test -s "$path" || { echo "[catalogue-calibration][error] missing: $path" >&2; exit 2; }
  done
  POSTERIORS+=(
    --posterior "raw_q=$RAW_Q32"
    --posterior "raw_iw=$RAW_IW32"
    --posterior "ema_q=$EMA_Q32"
    --posterior "ema_iw=$EMA_IW32"
  )
else
  for path in "$RAW_Q256" "$EMA_Q256"; do
    test -d "$path" || { echo "[catalogue-calibration][error] missing: $path" >&2; exit 2; }
    compgen -G "$path/*.parquet" >/dev/null || {
      echo "[catalogue-calibration][error] empty posterior bank: $path" >&2
      exit 2
    }
  done
  POSTERIORS+=(--posterior "raw_q=$RAW_Q256" --posterior "ema_q=$EMA_Q256")
fi

python - <<'PY'
import jax

print("[catalogue-calibration] backend:", jax.default_backend())
print("[catalogue-calibration] devices:", jax.devices())
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
    raise SystemExit("expected exactly one local GPU")
PY

COMMAND=(
  python "scripts/evaluate_feniks_${DIAGNOSTIC}.py"
  --truth "$TRUTH"
  "${POSTERIORS[@]}"
  --out "$STAGING"
  --samples-per-object "$DRAWS"
  --num-bootstrap "$NUM_BOOTSTRAP"
  --drop-nonfinite-truth
  --seed "$((263000 + TASK))"
)
if [[ "$DIAGNOSTIC" == mira ]]; then
  COMMAND+=(--num-regions "$NUM_REGIONS")
fi

echo "[catalogue-calibration] mode=$MODE diagnostic=$DIAGNOSTIC draws=$DRAWS"
printf '[catalogue-calibration] command:'
printf ' %q' "${COMMAND[@]}"
echo
"${COMMAND[@]}"

test -s "$STAGING/${DIAGNOSTIC}_summary.json"
test -e "$STAGING/DONE"
mkdir -p "$(dirname "$DESTINATION")"
mv "$STAGING" "$DESTINATION"
echo "[catalogue-calibration] complete -> $DESTINATION"
