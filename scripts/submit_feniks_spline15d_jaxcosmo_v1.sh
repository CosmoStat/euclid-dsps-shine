#!/bin/bash
set -Eeuo pipefail

SOURCE_DIR="Data/diffsky/synthetic/feniks_260617_dsps_closure_18band_grouped"
SPLINE_DIR="Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1"
CATALOG_DIR="${SPLINE_DIR}/amortized"
PRIOR_RUN="outputs/runs/feniks_spline15d_jaxcosmo_prior_v1"
AMORTIZED_RUN="outputs/runs/feniks_spline15d_jaxcosmo_amortized_v1"
INFERENCE_RUN="outputs/runs/feniks_spline15d_jaxcosmo_inference_v1"
PROJECTION_CONFIG="configs/feniks_spline15d_grouped_jaxcosmo_v1.yaml"
PRIOR_CONFIG="configs/prior_feniks_spline15d_jaxcosmo_v1.yaml"
AMORTIZED_CONFIG="configs/amortized_feniks_spline15d_jaxcosmo_v1_gpu.yaml"

for path in "$SOURCE_DIR/train.parquet" "$SOURCE_DIR/validation.parquet" \
  "$SOURCE_DIR/test.parquet"; do
  test -s "$path" || { echo "[submit][error] missing source: $path"; exit 2; }
done

for path in "$SPLINE_DIR" "$PRIOR_RUN" "$AMORTIZED_RUN" "$INFERENCE_RUN"; do
  test ! -e "$path" || {
    echo "[submit][error] refusing to reuse existing output: $path"; exit 2;
  }
done

python - <<'PY'
import jax_cosmo
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

print("[submit] jax_cosmo:", jax_cosmo.__version__)
print("[submit] spline:", InterpolatedUnivariateSpline.__name__)
PY

submit() {
  local raw
  raw=$(sbatch --parsable "$@")
  printf '%s' "${raw%%;*}"
}

projection_job=$(submit \
  --export="ALL,CONFIG=${PROJECTION_CONFIG},SOURCE_DATASET_DIR=${SOURCE_DIR},SPLINE_DATASET_DIR=${SPLINE_DIR}" \
  scripts/feniks_spline15d_project_h100.slurm)

prior_job=$(submit \
  --dependency="afterok:${projection_job}" \
  --time=12:00:00 \
  --export="ALL,SPLINE_DATASET_DIR=${SPLINE_DIR},OUT_DIR=${PRIOR_RUN},PRIOR_CONFIG=${PRIOR_CONFIG}" \
  scripts/feniks_spline15d_v6_positive_support_h100.slurm)

amortized_job=$(submit \
  --dependency="afterok:${prior_job}" \
  --gres=gpu:4 \
  --export="ALL,CONFIG=${AMORTIZED_CONFIG},SOURCE_DIR=${SOURCE_DIR},SPLINE_DIR=${SPLINE_DIR},CATALOG_DIR=${CATALOG_DIR},PRIOR_CHECKPOINT=${PRIOR_RUN}/checkpoints/best.eqx,OUT_DIR=${AMORTIZED_RUN},DATA_PARALLEL=pmap,BATCH_SIZE=1024,JAX_BATCH_SIZE=256,EPOCHS=120,N_SAMPLES=1,VALIDATION_EVERY=1" \
  scripts/feniks_spline15d_amortized_h100.slurm)

inference_job=$(submit \
  --dependency="afterok:${amortized_job}" \
  --export="ALL,CONFIG=${AMORTIZED_CONFIG},DATASET=${CATALOG_DIR}/test.parquet,TRAIN_RUN=${AMORTIZED_RUN},CHECKPOINT=${AMORTIZED_RUN}/checkpoints/best.eqx,FEATURE_STATS=${AMORTIZED_RUN}/feature_stats.json,FLOW_PRIOR_CHECKPOINT=${PRIOR_RUN}/checkpoints/best.eqx,RUN_NAME=feniks_spline15d_jaxcosmo_inference_v1,OUT_DIR=${INFERENCE_RUN}" \
  scripts/diffsky_amortized_infer_h100.slurm)

printf 'projection_job=%s\n' "$projection_job"
printf 'prior_job=%s dependency=afterok:%s\n' "$prior_job" "$projection_job"
printf 'amortized_job=%s dependency=afterok:%s\n' "$amortized_job" "$prior_job"
printf 'inference_job=%s dependency=afterok:%s\n' "$inference_job" "$amortized_job"
printf 'monitor: squeue -j %s,%s,%s,%s\n' \
  "$projection_job" "$prior_job" "$amortized_job" "$inference_job"
