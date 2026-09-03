#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
RECOVERY_ROOT="${RECOVERY_ROOT:?Source the epoch-160 evaluation environment first}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
SOURCE_BANK_VEM_ROOT="${SOURCE_BANK_VEM_ROOT:-$RECOVERY_ROOT/population_vem_epoch160_v2}"
CALIBRATION_VEM_ROOT="${CALIBRATION_VEM_ROOT:-$RECOVERY_ROOT/population_vem_epoch160_v1}"
PROJECTION_ROOT="${PROJECTION_ROOT:-$RECOVERY_ROOT/population_projection_epoch160_v1}"
PROJECTION_LOG_ROOT="${PROJECTION_LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$PROJECTION_ROOT")}"
CONFIG="configs/experiments/feniks_sc_drws_r29_current_production.yaml"
TRUTH_CONFIG="configs/experiments/feniks_sc_drws_r29_truth_closure.yaml"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[population-projection][error] tracked source changes are not committed" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
for path in \
  "$SOURCE_BANK_VEM_ROOT/RUN_MANIFEST.json" \
  "$SOURCE_BANK_VEM_ROOT/POPULATION_VEM_COMPLETE.json" \
  "$SOURCE_BANK_VEM_ROOT/banks/q_fit/bank_manifest.json" \
  "$SOURCE_BANK_VEM_ROOT/banks/q_validation/bank_manifest.json" \
  "$CALIBRATION_VEM_ROOT/RUN_MANIFEST.json" \
  "$CALIBRATION_VEM_ROOT/POPULATION_VEM_COMPLETE.json" \
  "$CALIBRATION_VEM_ROOT/q_refresh/Q_REFRESH_COMPLETE.json" \
  "$CALIBRATION_VEM_ROOT/evaluation/selected_test_truth.parquet" \
  "$CALIBRATION_VEM_ROOT/evaluation/mira/mira_summary.json" \
  "$CALIBRATION_VEM_ROOT/evaluation/tarp/tarp_summary.json" \
  "$CONFIG" "$TRUTH_CONFIG"; do
  test -s "$path" || {
    echo "[population-projection][error] missing: $path" >&2
    exit 2
  }
done
if [[ -s "$PROJECTION_ROOT/SUBMISSION.json" ]]; then
  echo "[population-projection][error] immutable run already submitted: $PROJECTION_ROOT" >&2
  echo "Source its saved environment and monitor it instead." >&2
  exit 2
fi
mkdir -p "$PROJECTION_LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs

python scripts/prepare_feniks_sc_drws_population_projection.py \
  --config "$CONFIG" --truth-config "$TRUTH_CONFIG" \
  --source-bank-vem-root "$SOURCE_BANK_VEM_ROOT" \
  --calibration-vem-root "$CALIBRATION_VEM_ROOT" \
  --out "$PROJECTION_ROOT"

JOB_REPO_DIR="${PROJECTION_CODE_ROOT:-$CACHE_ROOT/code/population-projection-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")"
if [[ -e "$JOB_REPO_DIR" ]]; then
  EXISTING_COMMIT="$(git -C "$JOB_REPO_DIR" rev-parse HEAD)"
  if [[ "$EXISTING_COMMIT" != "$CODE_COMMIT" ]]; then
    echo "[population-projection][error] code snapshot has wrong commit" >&2
    exit 2
  fi
else
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi
test -e "$JOB_REPO_DIR/Data/diffsky"

EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,PROJECTION_ROOT=$PROJECTION_ROOT,CACHE_ROOT=$CACHE_ROOT"
BETA_RAW=$(sbatch --parsable --array=0-19%20 \
  --output="$PROJECTION_LOG_ROOT/beta-%A_%a.out" \
  --error="$PROJECTION_LOG_ROOT/beta-%A_%a.err" \
  --export="$EXPORTS" \
  scripts/feniks_sc_drws_population_projection_beta_h100.slurm)
BETA_JOB="${BETA_RAW%%;*}"
GATE_RAW=$(sbatch --parsable --dependency="afterok:$BETA_JOB" \
  --output="$PROJECTION_LOG_ROOT/beta-gate-%j.out" \
  --error="$PROJECTION_LOG_ROOT/beta-gate-%j.err" \
  --export="$EXPORTS" \
  scripts/feniks_sc_drws_population_projection_beta_finalize.slurm)
GATE_JOB="${GATE_RAW%%;*}"
FIT_RAW=$(sbatch --parsable --dependency="afterok:$GATE_JOB" \
  --output="$PROJECTION_LOG_ROOT/fit-%j.out" \
  --error="$PROJECTION_LOG_ROOT/fit-%j.err" \
  --export="$EXPORTS" \
  scripts/feniks_sc_drws_population_projection_fit_h100.slurm)
FIT_JOB="${FIT_RAW%%;*}"
EVALUATION_RAW=$(sbatch --parsable --dependency="afterok:$FIT_JOB" \
  --output="$PROJECTION_LOG_ROOT/evaluation-%j.out" \
  --error="$PROJECTION_LOG_ROOT/evaluation-%j.err" \
  --export="$EXPORTS" \
  scripts/feniks_sc_drws_population_projection_evaluate_h100.slurm)
EVALUATION_JOB="${EVALUATION_RAW%%;*}"
ALL_JOBS="$BETA_JOB,$GATE_JOB,$FIT_JOB,$EVALUATION_JOB"
LATEST="outputs/logs/feniks_sc_drws_population_projection_latest.env"

printf 'export BETA_JOB=%q\nexport GATE_JOB=%q\nexport FIT_JOB=%q\nexport EVALUATION_JOB=%q\nexport ALL_JOBS=%q\nexport PROJECTION_ROOT=%q\nexport PROJECTION_LOG_ROOT=%q\nexport SOURCE_BANK_VEM_ROOT=%q\nexport CALIBRATION_VEM_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\n' \
  "$BETA_JOB" "$GATE_JOB" "$FIT_JOB" "$EVALUATION_JOB" "$ALL_JOBS" \
  "$PROJECTION_ROOT" "$PROJECTION_LOG_ROOT" "$SOURCE_BANK_VEM_ROOT" \
  "$CALIBRATION_VEM_ROOT" "$RECOVERY_ROOT" "$JOB_REPO_DIR" "$CODE_COMMIT" \
  > "$LATEST"

python - "$PROJECTION_ROOT" "$BETA_JOB" "$GATE_JOB" "$FIT_JOB" \
  "$EVALUATION_JOB" "$ALL_JOBS" "$JOB_REPO_DIR" "$CODE_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "status": "SUBMITTED",
    "beta_job": sys.argv[2],
    "beta_gate_job": sys.argv[3],
    "fit_job": sys.argv[4],
    "evaluation_job": sys.argv[5],
    "all_jobs": sys.argv[6],
    "code_snapshot": sys.argv[7],
    "code_commit": sys.argv[8],
}
(root / "SUBMISSION.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "beta_job=$BETA_JOB (20 one-H100 tasks; existing q banks, no new q inference)"
echo "beta_gate_job=$GATE_JOB"
echo "fit_job=$FIT_JOB (4 H100; selected and parent weighted MLE)"
echo "evaluation_job=$EVALUATION_JOB (1 H100; PIT/coverage + distributions)"
echo "root=$PROJECTION_ROOT"
echo "latest_env=$LATEST"
echo "monitor: bash scripts/monitor_feniks_sc_drws_population_projection.sh 30"
