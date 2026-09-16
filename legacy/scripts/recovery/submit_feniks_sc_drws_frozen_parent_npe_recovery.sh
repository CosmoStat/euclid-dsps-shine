#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ENV_FILE="${1:-outputs/logs/feniks_sc_drws_frozen_parent_npe_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
RECOVER_SCRATCH_ARM="${RECOVER_SCRATCH_ARM:-0}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
ENV_FILE="$(realpath "$ENV_FILE")"
test -s "$ENV_FILE" || { echo "[frozen-npe-recovery][error] missing: $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"

if [[ "$RECOVER_SCRATCH_ARM" != 1 ]]; then
  echo "[frozen-npe-recovery][error] set RECOVER_SCRATCH_ARM=1" >&2
  exit 2
fi
test -s "$NPE_ROOT/RUN_MANIFEST.json"
test ! -s "$NPE_ROOT/arms/scratch_encoder/ARM_COMPLETE.json" || {
  echo "[frozen-npe-recovery][error] scratch arm is already complete" >&2
  exit 2
}
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[frozen-npe-recovery][error] tracked source changes are not committed" >&2
  exit 2
fi

ORIGINAL_ARM_JOB="${ORIGINAL_ARM_JOB:-$ARM_JOB}"
OLD_GATE_JOB="$GATE_JOB"
OLD_SUBMIT_EVALUATION_JOB="$SUBMIT_EVALUATION_JOB"
CODE_COMMIT="$(git rev-parse HEAD)"
MANIFEST_COMMIT="$(python - "$NPE_ROOT/RUN_MANIFEST.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['code_commit'])
PY
)"
git merge-base --is-ancestor "$MANIFEST_COMMIT" "$CODE_COMMIT" || {
  echo "[frozen-npe-recovery][error] recovery commit does not descend from manifest" >&2
  exit 2
}

JOB_REPO_DIR="${FROZEN_NPE_RECOVERY_CODE_ROOT:-$CACHE_ROOT/code/frozen-npe-recovery-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")" "$NPE_LOG_ROOT"
if [[ -e "$JOB_REPO_DIR" ]]; then
  test "$(git -C "$JOB_REPO_DIR" rev-parse HEAD)" = "$CODE_COMMIT" || {
    echo "[frozen-npe-recovery][error] code snapshot has wrong commit" >&2
    exit 2
  }
else
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  mkdir -p "$JOB_REPO_DIR/Data"
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi

scancel "$OLD_GATE_JOB" "$OLD_SUBMIT_EVALUATION_JOB" 2>/dev/null || true
EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,NPE_ROOT=$NPE_ROOT,CACHE_ROOT=$CACHE_ROOT,BENCHMARK_ENV=$BENCHMARK_ENV,BASELINE_ENV=$BASELINE_ENV,NPE_RECOVERY_COMMIT=$CODE_COMMIT,NPE_STAGE4_CODE_COMMIT=$CODE_COMMIT"
SCRATCH_RAW=$(sbatch --parsable --array=1-1%1 \
  --output="$NPE_LOG_ROOT/arm-recovery-%A_%a.out" \
  --error="$NPE_LOG_ROOT/arm-recovery-%A_%a.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_frozen_parent_npe_train_h100.slurm")
SCRATCH_RECOVERY_JOB="${SCRATCH_RAW%%;*}"
GATE_RAW=$(sbatch --parsable \
  --dependency="afterok:${ORIGINAL_ARM_JOB}_0:$SCRATCH_RECOVERY_JOB" \
  --output="$NPE_LOG_ROOT/gate-recovery-%j.out" \
  --error="$NPE_LOG_ROOT/gate-recovery-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_frozen_parent_npe_gate.slurm")
GATE_JOB="${GATE_RAW%%;*}"
SUBMIT_RAW=$(sbatch --parsable --dependency="afterok:$GATE_JOB" \
  --output="$NPE_LOG_ROOT/submit-evaluation-recovery-%j.out" \
  --error="$NPE_LOG_ROOT/submit-evaluation-recovery-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_frozen_parent_npe_submit_evaluation.slurm")
SUBMIT_EVALUATION_JOB="${SUBMIT_RAW%%;*}"
ALL_JOBS="$ORIGINAL_ARM_JOB,$SCRATCH_RECOVERY_JOB,$GATE_JOB,$SUBMIT_EVALUATION_JOB"

printf 'export ORIGINAL_ARM_JOB=%q\nexport ARM_JOB=%q\nexport SCRATCH_RECOVERY_JOB=%q\nexport GATE_JOB=%q\nexport SUBMIT_EVALUATION_JOB=%q\nexport ALL_JOBS=%q\nexport NPE_ROOT=%q\nexport NPE_LOG_ROOT=%q\nexport BASELINE_ENV=%q\nexport BENCHMARK_ENV=%q\nexport SOURCE_BENCHMARK_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport CACHE_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\nexport NPE_RECOVERY_COMMIT=%q\n' \
  "$ORIGINAL_ARM_JOB" "$ORIGINAL_ARM_JOB" "$SCRATCH_RECOVERY_JOB" \
  "$GATE_JOB" "$SUBMIT_EVALUATION_JOB" "$ALL_JOBS" "$NPE_ROOT" \
  "$NPE_LOG_ROOT" "$BASELINE_ENV" "$BENCHMARK_ENV" \
  "$SOURCE_BENCHMARK_ROOT" "$RECOVERY_ROOT" "$CACHE_ROOT" \
  "$JOB_REPO_DIR" "$CODE_COMMIT" "$CODE_COMMIT" > "$ENV_FILE"

python - "$NPE_ROOT/SCRATCH_ARM_RECOVERY_SUBMISSION.json" "$MANIFEST_COMMIT" \
  "$CODE_COMMIT" "${ORIGINAL_ARM_JOB}_0" "$SCRATCH_RECOVERY_JOB" \
  "$GATE_JOB" "$SUBMIT_EVALUATION_JOB" <<'PY'
import json,sys
from pathlib import Path
payload={
    'status':'SUBMITTED',
    'scope':'scratch_encoder_cli_adapter_only',
    'manifest_code_commit':sys.argv[2],
    'recovery_code_commit':sys.argv[3],
    'reused_warm_task':sys.argv[4],
    'scratch_recovery_job':sys.argv[5],
    'replacement_gate_job':sys.argv[6],
    'replacement_stage4_submitter_job':sys.argv[7],
    'baseline_jobs_reused':True,
    'truth_used':False,
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY

echo "reused_warm_task=${ORIGINAL_ARM_JOB}_0"
echo "scratch_recovery_job=$SCRATCH_RECOVERY_JOB"
echo "replacement_gate_job=$GATE_JOB"
echo "replacement_stage4_submitter_job=$SUBMIT_EVALUATION_JOB"
echo "monitor: bash scripts/monitor_feniks_sc_drws_frozen_parent_npe.sh $ENV_FILE 30"
