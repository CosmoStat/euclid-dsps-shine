#!/bin/bash
set -Eeuo pipefail

CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_selfsup_rws_jaxcosmo_v1}"
CONFIGS=(
  configs/experiments/feniks_selfsup_fixed_ref_rws_k4_gaussian.yaml
  configs/experiments/feniks_selfsup_learned_rws_k4_gaussian.yaml
  configs/experiments/feniks_selfsup_learned_rws_sleep3_wake1_k4.yaml
  configs/experiments/feniks_selfsup_learned_rws_sleep3_wake1_k8.yaml
)

mkdir -p outputs/logs
validate_cmd=(
  python scripts/validate_feniks_selfsup_rws_inputs.py
  --catalog-dir "$CATALOG_DIR"
  --reference-checkpoint "$REFERENCE_CHECKPOINT"
)
for config in "${CONFIGS[@]}"; do validate_cmd+=(--config "$config"); done
JAX_PLATFORMS=cpu "${validate_cmd[@]}"
test ! -e "$ROOT_DIR" || {
  echo "[submit][error] refusing to reuse existing output: $ROOT_DIR"; exit 2;
}

smoke_raw=$(sbatch --parsable --time=00:10:00 \
  --export=ALL,SMOKE=1,FULL_ROOT_DIR="$ROOT_DIR" \
  scripts/feniks_selfsup_rws_h100.slurm)
smoke_job="${smoke_raw%%;*}"
full_raw=$(sbatch --parsable --dependency="afterok:${smoke_job}" \
  --export=ALL,SMOKE=0,FULL_ROOT_DIR="$ROOT_DIR" \
  scripts/feniks_selfsup_rws_h100.slurm)
full_job="${full_raw%%;*}"

stamp=$(date +%Y%m%d_%H%M%S)
submit_log="outputs/logs/submit_selfsup_rws_${stamp}.log"
{
  printf 'smoke=%s full=%s\n' "$smoke_job" "$full_job"
  printf 'root=%s\n' "$ROOT_DIR"
} | tee "$submit_log"
printf 'smoke_array=%s tasks=0-3 max_gpus=16 time=00:10:00\n' "$smoke_job"
printf 'full_array=%s tasks=0-3 max_gpus=16 dependency=afterok:%s\n' \
  "$full_job" "$smoke_job"
printf 'monitor: squeue -j %s,%s\n' "$smoke_job" "$full_job"
printf 'accounting: sacct -X -j %s,%s --format=JobID%%22,JobName%%24,State%%18,ExitCode,Elapsed,AllocTRES%%55\n' \
  "$smoke_job" "$full_job"
printf 'submission_log=%s\n' "$submit_log"
