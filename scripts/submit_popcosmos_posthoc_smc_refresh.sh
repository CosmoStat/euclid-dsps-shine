#!/bin/bash
set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the completed full_cont120 run}"
SMC_ROOT="${SMC_ROOT:?Set SMC_ROOT to a completed adaptive-SMC pilot}"
OUT="${OUT:-$SMC_ROOT/proposal_refresh}"
cd "$REPO_DIR"
mkdir -p outputs/logs
raw=$(sbatch --parsable \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",SOURCE_ROOT="$SOURCE_ROOT",SMC_ROOT="$SMC_ROOT",OUT="$OUT",REFRESH_EPOCHS="${REFRESH_EPOCHS:-20}",PROBE_SAMPLES="${PROBE_SAMPLES:-2048}",MIN_MEDIAN_ESS_FRACTION="${MIN_MEDIAN_ESS_FRACTION:-0.05}",MAX_FRACTION_PARETO_K_GT_0P7="${MAX_FRACTION_PARETO_K_GT_0P7:-0.2}" \
  scripts/popcosmos_posthoc_smc_refresh_h100.slurm)
job="${raw%%;*}"
env_file=outputs/logs/popcosmos_posthoc_smc_refresh_latest.env
printf 'export SMC_REFRESH_JOB=%q\nexport SMC_REFRESH_OUT=%q\nexport SMC_ROOT=%q\nexport SOURCE_ROOT=%q\n' \
  "$job" "$OUT" "$SMC_ROOT" "$SOURCE_ROOT" > "$env_file"
echo "smc_refresh_job=$job"
echo "smc_refresh_out=$OUT"
echo "monitor: squeue -j $job"
echo "log: outputs/logs/cosmos_qfix-${job}.out"
echo "latest_env=$env_file"
