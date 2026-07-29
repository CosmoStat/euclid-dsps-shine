#!/bin/bash
set -Eeuo pipefail

CONFIG="${CONFIG:-configs/experiments/popcosmos_a24_rws_joint.yaml}"
DATASET="${DATASET:-Data/cosmos2020/prepared/farmer_a24_full.parquet}"
MODEL_STAGE="${MODEL_STAGE:-n40k}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/popcosmos_a24_rws_v1/$MODEL_STAGE}"
OUT="${OUT:-outputs/runs/popcosmos_a24_mclmc_v1}"
MODE="${MODE:-pilot}"
ARRAY="0-5%2"
if [[ "$MODE" == "smoke" ]]; then ARRAY="0-1%2"; fi

test -e "$MODEL_ROOT/DONE"
test ! -e "$OUT" || {
  echo "[cosmos-mclmc-submit][error] refusing to reuse $OUT"; exit 2;
}
mkdir -p "$OUT" outputs/logs
python scripts/prepare_cosmos2020_mclmc_cohort.py \
  --dataset "$DATASET" --out "$OUT/cohort_input.csv"
EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu \
python scripts/run_feniks_exact_posterior_benchmark.py prepare-cohort \
  --config "$CONFIG" --dataset "$DATASET" \
  --checkpoint "$MODEL_ROOT/train/checkpoints/best.eqx" \
  --feature-stats "$MODEL_ROOT/train/feature_stats.json" \
  --out "$OUT" --cohort-file "$OUT/cohort_input.csv" --mode "$MODE"
job_raw=$(sbatch --parsable --array="$ARRAY" \
  --export=ALL,CONFIG="$CONFIG",DATASET="$DATASET",MODEL_STAGE="$MODEL_STAGE",MODEL_ROOT="$MODEL_ROOT",OUT="$OUT",MODE="$MODE" \
  scripts/cosmos2020_mclmc_h100.slurm)
job="${job_raw%%;*}"
printf 'mclmc_array=%s\nmonitor: squeue -j %s\n' "$job" "$job"
printf 'verify: sacct -j %s --format=JobID,State,Elapsed,AllocTRES,ExitCode\n' "$job"
