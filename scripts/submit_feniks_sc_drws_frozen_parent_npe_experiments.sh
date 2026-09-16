#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
BENCHMARK_ENV="${1:-outputs/logs/feniks_sc_drws_population_projection_benchmark_latest.env}"
BASELINE_ENV="${BASELINE_ENV:-outputs/logs/feniks_sc_drws_full_test_posterior_latest.env}"
NPE_ENV="${NPE_ENV:-outputs/logs/feniks_sc_drws_frozen_parent_npe_latest.env}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
export REPO_DIR BASELINE_ENV NPE_ENV

# Stage 2 is submitted first but runs concurrently with the stage-3 NPE arms.
bash scripts/submit_feniks_sc_drws_full_test_posterior.sh "$BENCHMARK_ENV"
bash scripts/submit_feniks_sc_drws_frozen_parent_npe.sh \
  "$BENCHMARK_ENV" "$BASELINE_ENV"

echo "both experiments submitted"
echo "monitor: bash scripts/monitor_feniks_sc_drws_frozen_parent_npe.sh $NPE_ENV 30"
