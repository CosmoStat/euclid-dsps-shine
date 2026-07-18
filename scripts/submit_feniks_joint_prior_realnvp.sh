#!/bin/bash
set -Eeuo pipefail

CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_joint_prior_realnvp_jaxcosmo_v1}"

mkdir -p outputs/logs
python scripts/validate_feniks_conditional_posterior_inputs.py \
  --catalog-dir "$CATALOG_DIR" --prior-checkpoint "$PRIOR_CHECKPOINT"
test ! -e "$ROOT_DIR" || {
  echo "[submit][error] refusing to reuse existing output: $ROOT_DIR"; exit 2;
}

smoke_raw=$(sbatch --parsable --time=00:10:00 --export=ALL,SMOKE=1 \
  scripts/feniks_joint_prior_realnvp_h100.slurm)
smoke_job="${smoke_raw%%;*}"
full_raw=$(sbatch --parsable --dependency="afterok:${smoke_job}" --export=ALL,SMOKE=0 \
  scripts/feniks_joint_prior_realnvp_h100.slurm)
full_job="${full_raw%%;*}"

printf 'smoke_array=%s tasks=0-5 max_gpus=24 time=00:10:00\n' "$smoke_job"
printf 'full_array=%s tasks=0-5 max_gpus=24 dependency=afterok:%s\n' "$full_job" "$smoke_job"
printf 'monitor: squeue -j %s,%s\n' "$smoke_job" "$full_job"
printf 'accounting: sacct -X -j %s,%s --format=JobID%%22,JobName%%24,State%%18,ExitCode,Elapsed,AllocTRES%%55\n' "$smoke_job" "$full_job"
