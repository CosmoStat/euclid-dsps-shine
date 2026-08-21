#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
TRAIN_ENV="${TRAIN_ENV:-outputs/logs/feniks_parentprior_sleepnpe_recovery_latest.env}"
source "$TRAIN_ENV"

CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_parentprior_sleepnpe_defensivewake_selection_r25.yaml}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
SOURCE_DATASET="${SOURCE_DATASET:-$CATALOG_DIR/test.parquet}"
CHECKPOINT="${CHECKPOINT:-$TRAIN_ROOT/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$TRAIN_ROOT/feature_stats.json}"
OBSERVED_COHORT="${OBSERVED_COHORT:-$MANIFEST_ROOT/exact_cohort.csv}"
DIAGNOSTIC_MODE="${DIAGNOSTIC_MODE:-smoke}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

case "$DIAGNOSTIC_MODE" in
  smoke)
    N_OBSERVED="${N_OBSERVED:-2}"
    N_SLEEP="${N_SLEEP:-2}"
    ROOT_SUFFIX="encoder_diagnostic_exact_smoke_4"
    ;;
  full)
    N_OBSERVED="${N_OBSERVED:-16}"
    N_SLEEP="${N_SLEEP:-16}"
    ROOT_SUFFIX="encoder_diagnostic_exact_32"
    ;;
  *)
    echo "[feniks-qdiag-submit][error] DIAGNOSTIC_MODE must be smoke or full" >&2
    exit 2
    ;;
esac
N_GALAXIES=$((N_OBSERVED + N_SLEEP))
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:-$RUN_ROOT/$ROOT_SUFFIX}"
INPUT_ROOT="$DIAGNOSTIC_ROOT/input"
EXACT_ROOT="$DIAGNOSTIC_ROOT/exact"
DATASET="$INPUT_ROOT/diagnostic_dataset.parquet"

cd "$REPO_DIR"
mkdir -p outputs/logs
for path in "$CONFIG" "$SOURCE_DATASET" "$CHECKPOINT" \
  "${CHECKPOINT}.json" "$FEATURE_STATS" "$OBSERVED_COHORT" \
  "$TRAIN_ROOT/training_summary.json" "$TRAIN_ROOT/training_log.csv"; do
  test -s "$path" || {
    echo "[feniks-qdiag-submit][error] missing: $path" >&2
    exit 2
  }
done
test ! -e "$DIAGNOSTIC_ROOT" || {
  echo "[feniks-qdiag-submit][error] output exists: $DIAGNOSTIC_ROOT" >&2
  exit 2
}

python - "$TRAIN_ROOT/training_summary.json" "$TRAIN_ROOT/training_log.csv" <<'PY'
import json
import sys
import pandas as pd

summary = json.load(open(sys.argv[1]))
log = pd.read_csv(sys.argv[2])
if summary.get("best_checkpoint_metric") != "validation_sleep_nll":
    raise SystemExit("checkpoint was not selected by validation_sleep_nll")
if int(summary.get("epochs", 0)) < 80:
    raise SystemExit("training did not reach epoch 80")
sleep = log.loc[log["update_phase"].eq("encoder_sleep")]
if sleep.empty or not bool((sleep["update_applied"] > 0).any()):
    raise SystemExit("no applied encoder sleep updates")
print("[feniks-qdiag-submit] accepted diagnostic checkpoint; prior PASS is not required")
PY

export REPO_DIR MINICONDA_PATH CONDA_ENV CONFIG SOURCE_DATASET CHECKPOINT
export FEATURE_STATS OBSERVED_COHORT INPUT_ROOT EXACT_ROOT DATASET
export N_OBSERVED N_SLEEP N_GALAXIES DIAGNOSTIC_MODE MAX_CONCURRENT
export ENCODER_DIAGNOSTIC_ONLY=1 DIAGNOSTIC_SEED="${DIAGNOSTIC_SEED:-260821}"

prep_raw=$(sbatch --parsable scripts/feniks_encoder_diagnostic_prepare_h100.slurm)
DIAGNOSTIC_PREP_JOB="${prep_raw%%;*}"
exact_raw=$(sbatch --parsable --dependency="afterok:${DIAGNOSTIC_PREP_JOB}" \
  --array="0-$((N_GALAXIES - 1))%${MAX_CONCURRENT}" \
  scripts/feniks_parentprior_exact_h100.slurm)
DIAGNOSTIC_EXACT_JOB="${exact_raw%%;*}"
final_raw=$(sbatch --parsable --dependency="afterok:${DIAGNOSTIC_EXACT_JOB}" \
  scripts/feniks_encoder_diagnostic_finalize.slurm)
DIAGNOSTIC_FINALIZER_JOB="${final_raw%%;*}"

latest=outputs/logs/feniks_encoder_diagnostic_exact_latest.env
printf 'export DIAGNOSTIC_PREP_JOB=%q\nexport DIAGNOSTIC_EXACT_JOB=%q\nexport DIAGNOSTIC_FINALIZER_JOB=%q\nexport DIAGNOSTIC_ROOT=%q\nexport EXACT_ROOT=%q\nexport INPUT_ROOT=%q\nexport N_GALAXIES=%q\nexport DIAGNOSTIC_MODE=%q\n' \
  "$DIAGNOSTIC_PREP_JOB" "$DIAGNOSTIC_EXACT_JOB" \
  "$DIAGNOSTIC_FINALIZER_JOB" "$DIAGNOSTIC_ROOT" "$EXACT_ROOT" \
  "$INPUT_ROOT" "$N_GALAXIES" "$DIAGNOSTIC_MODE" > "$latest"

echo "diagnostic_mode=$DIAGNOSTIC_MODE"
echo "diagnostic_prep_job=$DIAGNOSTIC_PREP_JOB"
echo "diagnostic_exact_job=$DIAGNOSTIC_EXACT_JOB"
echo "diagnostic_finalizer_job=$DIAGNOSTIC_FINALIZER_JOB"
echo "objects=$N_GALAXIES observed=$N_OBSERVED sleep_synthetic=$N_SLEEP"
echo "exact_root=$EXACT_ROOT"
echo "latest_env=$latest"
