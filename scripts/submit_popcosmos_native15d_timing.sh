#!/bin/bash
"${BASH_VERSION:+true}" 2>/dev/null || exec bash "$0" "$@"
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
ROOT_BASE="${ROOT_BASE:?Set ROOT_BASE to the completed native15d array root}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_BASE/publication_checks/timing_h100_${RUN_TAG}}"

cd "$REPO_DIR"
mkdir -p outputs/logs

raw=$(sbatch --parsable --array=0-1%2 \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",ROOT_BASE="$ROOT_BASE",OUTPUT_ROOT="$OUTPUT_ROOT",LIMIT="${LIMIT:-128}",POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-128}",DECODER_SAMPLE_CHUNK_SIZE="${DECODER_SAMPLE_CHUNK_SIZE:-1}",REPEATS="${REPEATS:-5}" \
  scripts/popcosmos_native15d_timing_h100.slurm)
job="${raw%%;*}"

env_file="outputs/logs/popcosmos_native15d_timing_latest.env"
printf 'export TIMING_JOB=%q\nexport TIMING_OUTPUT_ROOT=%q\nexport ROOT_BASE=%q\n' \
  "$job" "$OUTPUT_ROOT" "$ROOT_BASE" > "$env_file"
echo "timing_job=$job"
echo "timing_output_root=$OUTPUT_ROOT"
echo "monitor: squeue -j $job"
echo "logs: outputs/logs/cosmos15_time-${job}_<taskid>.out"
echo "latest_env=$env_file"
