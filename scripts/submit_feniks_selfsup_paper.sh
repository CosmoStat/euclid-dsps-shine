#!/bin/bash
set -Eeuo pipefail

CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_selfsup_paper_v1}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
CONFIGS=(
  configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml
  configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed3.yaml
  configs/experiments/feniks_selfsup_paper_fixed_prior_rws_k8_t2.yaml
  configs/experiments/feniks_selfsup_paper_avi_joint_t2.yaml
)

mkdir -p "$LOG_DIR"
validate_cmd=(
  python scripts/validate_feniks_selfsup_paper_inputs.py
  --catalog-dir "$CATALOG_DIR"
  --reference-checkpoint "$REFERENCE_CHECKPOINT"
)
for config in "${CONFIGS[@]}"; do validate_cmd+=(--config "$config"); done
JAX_PLATFORMS=cpu "${validate_cmd[@]}"
test ! -e "$ROOT_DIR" || {
  echo "[paper-submit][error] refusing to reuse existing output: $ROOT_DIR"; exit 2;
}

smoke_train_raw=$(sbatch --parsable --time=00:15:00 \
  --export=ALL,SMOKE=1,FULL_ROOT_DIR="$ROOT_DIR" \
  scripts/feniks_selfsup_paper_h100.slurm)
smoke_train="${smoke_train_raw%%;*}"
SMOKE_ROOT="${ROOT_DIR}_smoke_${smoke_train}"
smoke_lens_raw=$(sbatch --parsable --time=00:15:00 \
  --dependency="afterok:${smoke_train}" \
  --export=ALL,SMOKE=1,ROOT_DIR="$SMOKE_ROOT" \
  scripts/feniks_selfsup_paper_jlens_h100.slurm)
smoke_lens="${smoke_lens_raw%%;*}"
smoke_finalize_raw=$(sbatch --parsable --time=00:15:00 \
  --dependency="afterok:${smoke_lens}" \
  --export=ALL,ROOT_DIR="$SMOKE_ROOT" \
  scripts/feniks_selfsup_paper_finalize_h100.slurm)
smoke_finalize="${smoke_finalize_raw%%;*}"

full_train_raw=$(sbatch --parsable --dependency="afterok:${smoke_finalize}" \
  --export=ALL,SMOKE=0,FULL_ROOT_DIR="$ROOT_DIR" \
  scripts/feniks_selfsup_paper_h100.slurm)
full_train="${full_train_raw%%;*}"
full_lens_raw=$(sbatch --parsable --dependency="afterok:${full_train}" \
  --export=ALL,SMOKE=0,ROOT_DIR="$ROOT_DIR" \
  scripts/feniks_selfsup_paper_jlens_h100.slurm)
full_lens="${full_lens_raw%%;*}"
full_finalize_raw=$(sbatch --parsable --dependency="afterok:${full_lens}" \
  --export=ALL,ROOT_DIR="$ROOT_DIR" \
  scripts/feniks_selfsup_paper_finalize_h100.slurm)
full_finalize="${full_finalize_raw%%;*}"

stamp=$(date +%Y%m%d_%H%M%S)
submit_log="$LOG_DIR/submit_selfsup_paper_${stamp}.log"
{
  printf 'smoke_train=%s smoke_lens=%s smoke_finalize=%s\n' \
    "$smoke_train" "$smoke_lens" "$smoke_finalize"
  printf 'full_train=%s full_lens=%s full_finalize=%s\n' \
    "$full_train" "$full_lens" "$full_finalize"
  printf 'smoke_root=%s\nroot=%s\n' "$SMOKE_ROOT" "$ROOT_DIR"
} | tee "$submit_log"
printf 'peak_training_gpus=16 peak_jlens_gpus=16\n'
printf 'monitor: squeue -j %s,%s,%s,%s,%s,%s\n' \
  "$smoke_train" "$smoke_lens" "$smoke_finalize" \
  "$full_train" "$full_lens" "$full_finalize"
printf 'submission_log=%s\n' "$submit_log"
