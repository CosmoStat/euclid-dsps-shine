#!/bin/bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_selfsup_learned_t2_jaxcosmo_v1}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
SMC_LABEL="selfsup_smcwake_mix2_k4_t2"
SMC_DEPENDENCY_JOB_ID="${SMC_DEPENDENCY_JOB_ID:-}"
RWS_LABELS=(selfsup_rws_k8_t2 selfsup_rws_mix2_k8_t2)

for label in "${RWS_LABELS[@]}"; do
  test -f "$ROOT_DIR/$label/DONE" || {
    echo "[individual-submit][error] completed RWS task missing: $label"
    exit 2
  }
done
mkdir -p "$LOG_DIR"

if [[ -f "$ROOT_DIR/$SMC_LABEL/DONE" ]]; then
  smc_dependency=()
elif [[ -n "$SMC_DEPENDENCY_JOB_ID" ]]; then
  smc_dependency=(--dependency="afterok:${SMC_DEPENDENCY_JOB_ID}")
else
  mapfile -t active_smc_jobs < <(
    squeue --me --noheader --states=PENDING,RUNNING --format='%A|%j' |
      awk -F'|' '$2 == "feniks_t2smc_resume" {print $1}' |
      sort -u
  )
  if (( ${#active_smc_jobs[@]} != 1 )); then
    echo "[individual-submit][error] SMC is incomplete and its dependency is ambiguous."
    echo "Set SMC_DEPENDENCY_JOB_ID to the job that writes $ROOT_DIR/$SMC_LABEL/DONE."
    exit 2
  fi
  SMC_DEPENDENCY_JOB_ID="${active_smc_jobs[0]}"
  smc_dependency=(--dependency="afterok:${SMC_DEPENDENCY_JOB_ID}")
fi

rws_raw=$(sbatch --parsable \
  --array=0-1%2 \
  --export="ALL,ROOT_DIR=$ROOT_DIR" \
  scripts/feniks_individual_posteriors_h100.slurm)
rws_job="${rws_raw%%;*}"

smc_raw=$(sbatch --parsable \
  --array=2 \
  "${smc_dependency[@]}" \
  --export="ALL,ROOT_DIR=$ROOT_DIR" \
  scripts/feniks_individual_posteriors_h100.slurm)
smc_job="${smc_raw%%;*}"

stamp=$(date +%Y%m%d_%H%M%S)
submission_log="$LOG_DIR/submit_feniks_individual_posteriors_${stamp}.log"
{
  printf 'root=%s\n' "$ROOT_DIR"
  printf 'rws_array=%s\n' "$rws_job"
  printf 'smc_task=%s dependency=%s\n' \
    "$smc_job" "${SMC_DEPENDENCY_JOB_ID:-already_complete}"
} | tee "$submission_log"
printf 'monitor: squeue -j %s,%s\n' "$rws_job" "$smc_job"
printf 'submission_log=%s\n' "$submission_log"
