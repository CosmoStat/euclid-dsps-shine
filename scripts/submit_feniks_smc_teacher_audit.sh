#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
source "${CURRICULUM_ENV:-outputs/logs/feniks_adaptive_smc_q_curriculum_latest.env}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_adaptive_smcwake_parentprior_selection_r25.yaml}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
AUDIT_TAG="${AUDIT_TAG:-feniks_smc_teacher_audit_$(date +%Y%m%d_%H%M%S)}"
AUDIT_ROOT="${AUDIT_ROOT:-outputs/runs/$AUDIT_TAG}"
BOOTSTRAP_ROOT="$AUDIT_ROOT/bootstrap"
DISTILLED_ROOT="$AUDIT_ROOT/distilled"
AUDIT_SUMMARY="$AUDIT_ROOT/summary"
BOOTSTRAP_CHECKPOINT="$SOURCE_SMOKE_ROOT/train/checkpoints/bootstrap.eqx"
DISTILLED_CHECKPOINT="$CURRICULUM_ROOT/checkpoints/q_exact_curriculum.eqx"
FEATURE_STATS="$CURRICULUM_ROOT/feature_stats.json"
COHORT_FILE="$SOURCE_SMOKE_ROOT/manifests/exact_cohort.csv"
DATASET="$CATALOG_DIR/test.parquet"

cd "$REPO_DIR"
for path in "$CONFIG" "$DATASET" "$FEATURE_STATS" "$COHORT_FILE" \
  "$BOOTSTRAP_CHECKPOINT" "$DISTILLED_CHECKPOINT"; do
  test -s "$path" || { echo "[teacher-audit][error] missing: $path" >&2; exit 2; }
done
test -e "$CURRICULUM_ROOT/DONE" || {
  echo "[teacher-audit][error] curriculum is incomplete" >&2; exit 2;
}
python - "$CURRICULUM_ROOT/curriculum_receipt.json" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1]))
expected = {
    "q_updates_applied": 64,
    "prior_updates_applied": 0,
    "truth_used_for_training_or_selection": False,
}
actual = {name: receipt.get(name) for name in expected}
if actual != expected:
    raise SystemExit(f"incompatible curriculum receipt: {actual} != {expected}")
if not receipt.get("selection_gradient_ready", False):
    raise SystemExit("selection gradient is not ready")
PY
test ! -e "$AUDIT_ROOT" || {
  echo "[teacher-audit][error] immutable root exists: $AUDIT_ROOT" >&2; exit 2;
}
mkdir -p outputs/logs "$AUDIT_ROOT"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
for spec in "$BOOTSTRAP_CHECKPOINT:$BOOTSTRAP_ROOT" \
  "$DISTILLED_CHECKPOINT:$DISTILLED_ROOT"; do
  checkpoint="${spec%%:*}"
  root="${spec#*:}"
  JAX_PLATFORMS=cpu python scripts/run_feniks_exact_posterior_benchmark.py \
    prepare-cohort --config "$CONFIG" --dataset "$DATASET" \
    --checkpoint "$checkpoint" --feature-stats "$FEATURE_STATS" \
    --out "$root" --mode full --cohort-file "$COHORT_FILE" --ignore-truth
done

count=$(python - "$BOOTSTRAP_ROOT/cohort.csv" <<'PY'
import pandas as pd
import sys
print(len(pd.read_csv(sys.argv[1])))
PY
)
[[ "$count" == 8 ]] || {
  echo "[teacher-audit][error] expected 8 objects, got $count" >&2; exit 2;
}

export REPO_DIR MINICONDA_PATH CONDA_ENV CONFIG DATASET FEATURE_STATS
export BOOTSTRAP_CHECKPOINT DISTILLED_CHECKPOINT BOOTSTRAP_ROOT DISTILLED_ROOT
export AUDIT_SUMMARY
raw=$(sbatch --parsable --array="0-7%4" scripts/feniks_smc_teacher_audit_h100.slurm)
AUDIT_JOB="${raw%%;*}"
final_raw=$(sbatch --parsable --dependency="afterok:${AUDIT_JOB}" \
  scripts/feniks_smc_teacher_audit_finalize.slurm)
AUDIT_FINALIZER_JOB="${final_raw%%;*}"

latest=outputs/logs/feniks_smc_teacher_audit_latest.env
printf 'export AUDIT_JOB=%q\nexport AUDIT_FINALIZER_JOB=%q\nexport AUDIT_ROOT=%q\nexport BOOTSTRAP_ROOT=%q\nexport DISTILLED_ROOT=%q\nexport AUDIT_SUMMARY=%q\n' \
  "$AUDIT_JOB" "$AUDIT_FINALIZER_JOB" "$AUDIT_ROOT" "$BOOTSTRAP_ROOT" \
  "$DISTILLED_ROOT" "$AUDIT_SUMMARY" > "$latest"
echo "audit_job=$AUDIT_JOB"
echo "audit_finalizer_job=$AUDIT_FINALIZER_JOB"
echo "audit_root=$AUDIT_ROOT"
echo "array=0-7%4 one_galaxy_per_H100"
echo "big_job_not_submitted=1"
echo "latest_env=$latest"
