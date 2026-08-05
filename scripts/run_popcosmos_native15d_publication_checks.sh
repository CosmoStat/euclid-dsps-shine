#!/bin/bash
"${BASH_VERSION:+true}" 2>/dev/null || exec bash "$0" "$@"
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
ROOT_BASE="${ROOT_BASE:?Set ROOT_BASE to the completed native15d array root}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CHECK_ROOT="${CHECK_ROOT:-$ROOT_BASE/publication_checks/checks_${RUN_TAG}}"
DATASET="${DATASET:-Data/cosmos2020/prepared/farmer_a24_full.parquet}"
POPCOSMOS="${POPCOSMOS:-Data/cosmos2020/assets/zenodo/summaries.txt}"

cd "$REPO_DIR"
mkdir -p outputs/logs "$CHECK_ROOT"

RWS26="$ROOT_BASE/bands26/full_cont120/inference/redshift_predictions.parquet"
RWS24="$ROOT_BASE/bands24_no_irac/full_cont120/inference/redshift_predictions.parquet"
for path in "$RWS26" "$RWS24" "$POPCOSMOS" "$DATASET"; do
  test -s "$path"
done

python scripts/compare_popcosmos_redshift_only.py \
  --rws26 "$RWS26" \
  --rws24 "$RWS24" \
  --popcosmos "$POPCOSMOS" \
  --out "$CHECK_ROOT/redshift_only_comparison" \
  --expected-evaluation 5000 \
  --expected-specz 1395 \
  --bootstrap "${BOOTSTRAP:-10000}" \
  --bootstrap-seed "${BOOTSTRAP_SEED:-260805}"

python scripts/audit_popcosmos_spectroscopic_cohort.py \
  --popcosmos "$POPCOSMOS" \
  --evaluation "$RWS26" \
  --prepared-full "$DATASET" \
  --out "$CHECK_ROOT/spectroscopy_cohort_audit" \
  --expected-published 12014 \
  --expected-xray 501 \
  --expected-fallback 1395

OUTPUT_ROOT="$CHECK_ROOT/timing_h100" \
ROOT_BASE="$ROOT_BASE" \
RUN_TAG="$RUN_TAG" \
bash scripts/submit_popcosmos_native15d_timing.sh

source outputs/logs/popcosmos_native15d_timing_latest.env
env_file="outputs/logs/popcosmos_publication_checks_latest.env"
printf 'export ROOT_BASE=%q\nexport CHECK_ROOT=%q\nexport COMPARISON_OUT=%q\nexport SPECZ_AUDIT_OUT=%q\nexport TIMING_JOB=%q\nexport TIMING_OUTPUT_ROOT=%q\n' \
  "$ROOT_BASE" \
  "$CHECK_ROOT" \
  "$CHECK_ROOT/redshift_only_comparison" \
  "$CHECK_ROOT/spectroscopy_cohort_audit" \
  "$TIMING_JOB" \
  "$TIMING_OUTPUT_ROOT" > "$env_file"

echo "publication_checks=$CHECK_ROOT"
echo "timing_job=$TIMING_JOB"
echo "monitor: squeue -j $TIMING_JOB"
echo "latest_env=$env_file"
