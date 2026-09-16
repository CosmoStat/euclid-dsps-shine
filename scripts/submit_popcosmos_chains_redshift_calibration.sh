#!/bin/bash
# Submit the Pop-COSMOS chain download and offline redshift calibration.

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536}"
ASSET_ROOT="${ASSET_ROOT:-${SCRATCH:?Set SCRATCH or ASSET_ROOT}/popcosmos_v1p1_chains}"
POSTERIOR_OUT="${POSTERIOR_OUT:-$ASSET_ROOT/redshift_posterior_1395}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/popcosmos_chain_redshift_calibration_v1}"
PAIRED_SPECZ="${PAIRED_SPECZ:-$RUN_ROOT/publication_checks/checks_20260805_143042/redshift_only_comparison/paired_public_specz_objects.parquet}"
SKIP_PREPOST="${SKIP_PREPOST:-0}"

cd "$REPO_DIR"
mkdir -p outputs/logs
test -s "$PAIRED_SPECZ" || {
  echo "[popcosmos-chain-submit][error] missing cohort: $PAIRED_SPECZ"
  exit 2
}
if [[ -e "$OUT_DIR" ]]; then
  echo "[popcosmos-chain-submit][error] output already exists: $OUT_DIR"
  exit 2
fi

common_export="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,RUN_ROOT=$RUN_ROOT,ASSET_ROOT=$ASSET_ROOT,POSTERIOR_OUT=$POSTERIOR_OUT,PAIRED_SPECZ=$PAIRED_SPECZ,OUT_DIR=$OUT_DIR"

if [[ "$SKIP_PREPOST" == "1" ]]; then
  test -s "$ASSET_ROOT/PREPOST_COMPLETE.json" || {
    echo "[popcosmos-chain-submit][error] SKIP_PREPOST=1 but marker is missing"
    exit 2
  }
  prepost_job="reused"
  evaluation_raw=$(sbatch --parsable \
    --export="$common_export" \
    scripts/popcosmos_chains_redshift_calibration_h100.slurm)
else
  prepost_raw=$(sbatch --parsable \
    --export="$common_export" \
    scripts/popcosmos_chains_prepost.slurm)
  prepost_job="${prepost_raw%%;*}"
  evaluation_raw=$(sbatch --parsable \
    --dependency="afterok:$prepost_job" \
    --export="$common_export" \
    scripts/popcosmos_chains_redshift_calibration_h100.slurm)
fi
evaluation_job="${evaluation_raw%%;*}"

run_tag=$(date +%Y%m%d_%H%M%S)
submission_log="outputs/logs/submit_popcosmos_chains_${run_tag}.log"
cat > "$submission_log" <<EOF
prepost=$prepost_job
evaluation=$evaluation_job
asset_root=$ASSET_ROOT
posterior_out=$POSTERIOR_OUT
output=$OUT_DIR
paired_specz=$PAIRED_SPECZ
EOF
cat > outputs/logs/popcosmos_chains_latest.env <<EOF
export PREPOST_JOB=$prepost_job
export EVALUATION_JOB=$evaluation_job
export ASSET_ROOT=$(printf '%q' "$ASSET_ROOT")
export POSTERIOR_OUT=$(printf '%q' "$POSTERIOR_OUT")
export CALIBRATION_OUT=$(printf '%q' "$OUT_DIR")
export PAIRED_SPECZ=$(printf '%q' "$PAIRED_SPECZ")
export SUBMISSION_LOG=$(printf '%q' "$submission_log")
EOF

cat "$submission_log"
echo "latest_env=outputs/logs/popcosmos_chains_latest.env"
if [[ "$prepost_job" == "reused" ]]; then
  squeue -j "$evaluation_job"
else
  squeue -j "$prepost_job,$evaluation_job"
fi
