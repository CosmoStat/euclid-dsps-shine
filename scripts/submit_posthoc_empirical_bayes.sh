#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/runs/posthoc_calibration_${RUN_TAG}/empirical_bayes}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"

cd "$REPO_DIR"
mkdir -p outputs/logs
raw=$(sbatch --parsable --array="0-1%${ARRAY_CONCURRENCY}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",OUTPUT_ROOT="$OUTPUT_ROOT",TRAIN_LIMIT="${TRAIN_LIMIT:-5000}",EVAL_LIMIT="${EVAL_LIMIT:-500}",PROPOSAL_SAMPLES="${PROPOSAL_SAMPLES:-512}",EVAL_SAMPLES="${EVAL_SAMPLES:-512}",EM_ITERATIONS="${EM_ITERATIONS:-3}",MSTEP_EPOCHS="${MSTEP_EPOCHS:-5}",ALLOW_LOW_ESS="${ALLOW_LOW_ESS:-0}" \
  scripts/posthoc_empirical_bayes_h100.slurm)
job="${raw%%;*}"

env_file="outputs/logs/posthoc_empirical_bayes_latest.env"
printf 'export EMPIRICAL_BAYES_JOB=%q\nexport EMPIRICAL_BAYES_OUTPUT_ROOT=%q\n' \
  "$job" "$OUTPUT_ROOT" > "$env_file"
echo "empirical_bayes_job=$job"
echo "empirical_bayes_output_root=$OUTPUT_ROOT"
echo "monitor: squeue -j $job"
echo "logs: outputs/logs/posthoc_em-${job}_<taskid>.out"
echo "latest_env=$env_file"
