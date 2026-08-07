#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/runs/posthoc_calibration_${RUN_TAG}/importance_probes}"
BUDGETS_CSV="${BUDGETS_CSV:-128,512,2048}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
LIMIT="${LIMIT:-256}"

cd "$REPO_DIR"
mkdir -p outputs/logs
IFS=',' read -r -a budgets <<< "$BUDGETS_CSV"
n_tasks=$((2 * ${#budgets[@]}))
last_task=$((n_tasks - 1))

raw=$(sbatch --parsable --array="0-${last_task}%${ARRAY_CONCURRENCY}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",OUTPUT_ROOT="$OUTPUT_ROOT",BUDGETS_CSV="$BUDGETS_CSV",LIMIT="$LIMIT" \
  scripts/posthoc_importance_probe_h100.slurm)
job="${raw%%;*}"

env_file="outputs/logs/posthoc_importance_latest.env"
printf 'export IMPORTANCE_JOB=%q\nexport IMPORTANCE_OUTPUT_ROOT=%q\nexport BUDGETS_CSV=%q\nexport LIMIT=%q\n' \
  "$job" "$OUTPUT_ROOT" "$BUDGETS_CSV" "$LIMIT" > "$env_file"
echo "importance_job=$job"
echo "importance_output_root=$OUTPUT_ROOT"
echo "monitor: squeue -j $job"
echo "logs: outputs/logs/posthoc_iw-${job}_<taskid>.out"
echo "latest_env=$env_file"
