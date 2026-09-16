#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
PROPOSAL_REFRESH_OUT="${PROPOSAL_REFRESH_OUT:?Set PROPOSAL_REFRESH_OUT to the completed updated-prior refresh}"
UPDATED_CHECKPOINT="${UPDATED_CHECKPOINT:-$PROPOSAL_REFRESH_OUT/encoder_refresh/checkpoints/best.eqx}"
UPDATED_FEATURE_STATS="${UPDATED_FEATURE_STATS:-$PROPOSAL_REFRESH_OUT/encoder_refresh/feature_stats.json}"
IMPORTANCE_DIAGNOSTICS="${IMPORTANCE_DIAGNOSTICS:-$PROPOSAL_REFRESH_OUT/moderate_k2048_importance/importance_diagnostics.parquet}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:-$PROPOSAL_REFRESH_OUT/proposal_diagnostic_${RUN_TAG}}"
N_SHARDS="${N_SHARDS:-8}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-16}"
WORST_OBJECTS="${WORST_OBJECTS:-32}"
CONTROL_OBJECTS="${CONTROL_OBJECTS:-32}"
SMC_SEEDS_CSV="260817,260818"
SMC_VARIANTS_CSV="floor_0p05"
export SMC_VARIANTS_CSV SMC_SEEDS_CSV

cd "$REPO_DIR"
for path in "$UPDATED_CHECKPOINT" "${UPDATED_CHECKPOINT}.json" "$UPDATED_FEATURE_STATS" "$IMPORTANCE_DIAGNOSTICS"; do
  test -s "$path" || { echo "[proposal-expressivity-submit][error] missing: $path" >&2; exit 2; }
done
test ! -e "$DIAGNOSTIC_ROOT" || {
  echo "[proposal-expressivity-submit][error] output exists: $DIAGNOSTIC_ROOT" >&2
  exit 2
}
mkdir -p outputs/logs "$DIAGNOSTIC_ROOT"

CONFIG=configs/experiments/popcosmos_native15d_rws_floor05.yaml
DATASET=Data/cosmos2020/prepared/farmer_a24_full.parquet
python scripts/build_popcosmos_proposal_diagnostic_panel.py \
  --importance-diagnostics "$IMPORTANCE_DIAGNOSTICS" \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --feature-stats "$UPDATED_FEATURE_STATS" \
  --out "$DIAGNOSTIC_ROOT/cohorts" \
  --worst-objects "$WORST_OBJECTS" \
  --control-objects "$CONTROL_OBJECTS" \
  --healthy-pareto-k-max "${HEALTHY_PARETO_K_MAX:-0.5}" \
  --n-shards "$N_SHARDS" \
  --validation-fraction "${VALIDATION_FRACTION:-0.5}" \
  --seed "${SEED:-260819}"

LIMIT=$((WORST_OBJECTS + CONTROL_OBJECTS))
array_tasks=$((2 * N_SHARDS))
array_max=$((array_tasks - 1))
smc_raw=$(sbatch --parsable --array="0-${array_max}%${ARRAY_CONCURRENCY}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",SOURCE_ROOT="$PROPOSAL_REFRESH_OUT",SMC_CHECKPOINT="$UPDATED_CHECKPOINT",SMC_FEATURE_STATS="$UPDATED_FEATURE_STATS",OUTPUT_ROOT="$DIAGNOSTIC_ROOT",LIMIT="$LIMIT",N_SHARDS="$N_SHARDS",PARTICLES="${PARTICLES:-1024}",OBJECT_BATCH_SIZE="${SMC_OBJECT_BATCH_SIZE:-4}",TARGET_ESS_FRACTION="${TARGET_ESS_FRACTION:-0.5}",MAX_STAGES="${MAX_STAGES:-64}",MALA_STEPS="${MALA_STEPS:-2}",MALA_STEP_SIZE="${MALA_STEP_SIZE:-0.005}",MALA_PARTICLE_CHUNK_SIZE="${MALA_PARTICLE_CHUNK_SIZE:-64}" \
  scripts/popcosmos_posthoc_smc_h100.slurm)
SMC_DIAGNOSTIC_JOB="${smc_raw%%;*}"

final_raw=$(sbatch --parsable --dependency="afterok:${SMC_DIAGNOSTIC_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",OUTPUT_ROOT="$DIAGNOSTIC_ROOT",N_SHARDS="$N_SHARDS",LIMIT="$LIMIT" \
  scripts/popcosmos_posthoc_smc_finalize.slurm)
SMC_DIAGNOSTIC_FINALIZER_JOB="${final_raw%%;*}"

expr_raw=$(sbatch --parsable --dependency="afterok:${SMC_DIAGNOSTIC_FINALIZER_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",DIAGNOSTIC_ROOT="$DIAGNOSTIC_ROOT",UPDATED_CHECKPOINT="$UPDATED_CHECKPOINT",UPDATED_FEATURE_STATS="$UPDATED_FEATURE_STATS",EPOCHS="${EPOCHS:-40}",PROPOSAL_SAMPLES="${PROPOSAL_SAMPLES:-2048}",DECODER_SAMPLE_CHUNK_SIZE="${DECODER_SAMPLE_CHUNK_SIZE:-1}",EXPERTS="${EXPERTS:-2}",OBJECT_BATCH_SIZE="${OBJECT_BATCH_SIZE:-8}",LEARNING_RATE="${LEARNING_RATE:-2e-5}",WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}",MIXTURE_MEAN_OFFSET="${MIXTURE_MEAN_OFFSET:-0.05}",GEOMETRY_DRAWS="${GEOMETRY_DRAWS:-256}",GEOMETRY_PROJECTIONS="${GEOMETRY_PROJECTIONS:-64}",SEED="${SEED:-260819}" \
  scripts/popcosmos_proposal_expressivity_h100.slurm)
PROPOSAL_EXPRESSIVITY_JOB="${expr_raw%%;*}"

env_file=outputs/logs/popcosmos_proposal_expressivity_latest.env
printf 'export SMC_DIAGNOSTIC_JOB=%q\nexport SMC_DIAGNOSTIC_FINALIZER_JOB=%q\nexport PROPOSAL_EXPRESSIVITY_JOB=%q\nexport DIAGNOSTIC_ROOT=%q\nexport PROPOSAL_EXPRESSIVITY_OUT=%q\nexport PROPOSAL_REFRESH_OUT=%q\nexport UPDATED_CHECKPOINT=%q\nexport UPDATED_FEATURE_STATS=%q\nexport IMPORTANCE_DIAGNOSTICS=%q\n' \
  "$SMC_DIAGNOSTIC_JOB" "$SMC_DIAGNOSTIC_FINALIZER_JOB" \
  "$PROPOSAL_EXPRESSIVITY_JOB" "$DIAGNOSTIC_ROOT" \
  "$DIAGNOSTIC_ROOT/proposal_expressivity" "$PROPOSAL_REFRESH_OUT" \
  "$UPDATED_CHECKPOINT" "$UPDATED_FEATURE_STATS" "$IMPORTANCE_DIAGNOSTICS" \
  > "$env_file"

echo "smc_diagnostic_job=$SMC_DIAGNOSTIC_JOB"
echo "smc_diagnostic_finalizer_job=$SMC_DIAGNOSTIC_FINALIZER_JOB"
echo "proposal_expressivity_job=$PROPOSAL_EXPRESSIVITY_JOB"
echo "diagnostic_root=$DIAGNOSTIC_ROOT"
echo "array_tasks=$array_tasks concurrency=$ARRAY_CONCURRENCY"
echo "latest_env=$env_file"
