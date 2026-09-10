#!/bin/bash
#SBATCH --job-name=feniks_rws_guarded
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=20:00:00
set -Eeuo pipefail
cd "${REPO_DIR:?}"
module purge
module load arch/h100
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" MPLBACKEND=Agg
export MPLCONFIGDIR="$CACHE_ROOT/mpl/$SLURM_JOB_ID"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
OUT="$GUARDED_ROOT/seed_$TASK"
ARGS=(--require-full-dataset)
TRAIN_ROWS="$MANIFEST_ROOT/full_train_indices.npy"
VALIDATION_ROWS="$MANIFEST_ROOT/confirmation_indices.npy"
SELECTION_MANIFEST="$MANIFEST_ROOT/manifest.json"
if [[ "${GUARDED_SMOKE:-0}" == 1 ]]; then
  OUT="$GUARDED_ROOT/smoke"
  ARGS=(--smoke)
  TRAIN_ROWS="$GUARDED_ROOT/smoke_full_train_indices.npy"
  VALIDATION_ROWS="$GUARDED_ROOT/smoke_confirmation_indices.npy"
  SELECTION_MANIFEST="$GUARDED_ROOT/smoke_manifest.json"
elif [[ -s "$OUT/states/latest.eqx" ]]; then
  ARGS+=(--resume-state "$OUT/states/latest.eqx")
fi
if [[ "${GUARDED_SMOKE:-0}" != 1 ]]; then
  python -m scripts.check_feniks_global_smoke "$GUARDED_ROOT/smoke"
fi
python scripts/train_feniks_sc_drws.py \
  --config configs/experiments/feniks_sc_drws_r29_global_guarded.yaml \
  --out "$OUT" --seed "$((260826 + TASK))" \
  --train-indices-file "$TRAIN_ROWS" \
  --validation-indices-file "$VALIDATION_ROWS" \
  --validation-catalog "$CATALOG_DIR/test.parquet" \
  --manifest-file "$SELECTION_MANIFEST" "${ARGS[@]}"
if [[ "${GUARDED_SMOKE:-0}" == 1 ]]; then
  python -m scripts.check_feniks_global_smoke "$OUT"
fi
