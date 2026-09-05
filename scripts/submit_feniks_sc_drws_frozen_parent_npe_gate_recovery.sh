#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ENV_FILE="${1:-outputs/logs/feniks_sc_drws_frozen_parent_npe_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
RECOVER_GATE_ONLY="${RECOVER_GATE_ONLY:-0}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH or CACHE_ROOT}/feniks_sc_drws_runtime}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
ENV_FILE="$(realpath "$ENV_FILE")"
test -s "$ENV_FILE" || { echo "[npe-gate-recovery][error] missing: $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"
if [[ "$RECOVER_GATE_ONLY" != 1 ]]; then
  echo "[npe-gate-recovery][error] set RECOVER_GATE_ONLY=1" >&2
  exit 2
fi
for receipt in \
  "$NPE_ROOT/arms/warm_start/ARM_COMPLETE.json" \
  "$NPE_ROOT/arms/scratch_encoder/ARM_COMPLETE.json"; do
  test -s "$receipt" || { echo "[npe-gate-recovery][error] missing: $receipt" >&2; exit 2; }
  test "$(python - "$receipt" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('status'))
PY
)" = PASS || { echo "[npe-gate-recovery][error] non-PASS arm: $receipt" >&2; exit 2; }
done
test ! -s "$NPE_ROOT/NPE_WINNER_FROZEN.json" || {
  echo "[npe-gate-recovery][error] winner already frozen" >&2
  exit 2
}
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[npe-gate-recovery][error] tracked source changes are not committed" >&2
  exit 2
fi

FAILED_GATE_JOB="$GATE_JOB"
OLD_SUBMIT_EVALUATION_JOB="$SUBMIT_EVALUATION_JOB"
CODE_COMMIT="$(git rev-parse HEAD)"
MANIFEST="$NPE_ROOT/RUN_MANIFEST.json"
MANIFEST_COMMIT="$(python - "$MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['code_commit'])
PY
)"
git merge-base --is-ancestor "$MANIFEST_COMMIT" "$CODE_COMMIT" || {
  echo "[npe-gate-recovery][error] recovery commit does not descend from manifest" >&2
  exit 2
}
readarray -t ARM_COMMITS < <(python - "$NPE_ROOT" "$MANIFEST_COMMIT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); fallback=sys.argv[2]
for arm in ('warm_start','scratch_encoder'):
    receipt=json.load(open(root/'arms'/arm/'ARM_COMPLETE.json'))
    print(receipt.get('runtime_code_commit', fallback))
PY
)
for arm_commit in "${ARM_COMMITS[@]}"; do
  git merge-base --is-ancestor "$MANIFEST_COMMIT" "$arm_commit" || {
    echo "[npe-gate-recovery][error] arm commit is not descended from manifest: $arm_commit" >&2
    exit 2
  }
  git merge-base --is-ancestor "$arm_commit" "$CODE_COMMIT" || {
    echo "[npe-gate-recovery][error] finalizer does not descend from arm: $arm_commit" >&2
    exit 2
  }
done

JOB_REPO_DIR="${FROZEN_NPE_GATE_CODE_ROOT:-$CACHE_ROOT/code/frozen-npe-gate-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")" "$NPE_LOG_ROOT"
if [[ -e "$JOB_REPO_DIR" ]]; then
  test "$(git -C "$JOB_REPO_DIR" rev-parse HEAD)" = "$CODE_COMMIT" || exit 2
else
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  mkdir -p "$JOB_REPO_DIR/Data"
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi

AUTHORIZATION="$NPE_ROOT/GATE_RECOVERY_AUTHORIZATION_${FAILED_GATE_JOB}.json"
python - "$AUTHORIZATION" "$NPE_ROOT" "$MANIFEST" "$MANIFEST_COMMIT" \
  "$CODE_COMMIT" "$FAILED_GATE_JOB" <<'PY'
import hashlib,json,sys
from pathlib import Path

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

path=Path(sys.argv[1]); root=Path(sys.argv[2]).resolve()
payload={
    'status':'AUTHORIZED',
    'scope':'gate_finalizer_only_no_git_binary',
    'reason':'CPU runtime has no git executable after module purge',
    'npe_root':str(root),
    'manifest_sha256':sha256(sys.argv[3]),
    'manifest_code_commit':sys.argv[4],
    'finalizer_code_commit':sys.argv[5],
    'failed_gate_job':sys.argv[6],
    'warm_receipt_sha256':sha256(root/'arms/warm_start/ARM_COMPLETE.json'),
    'scratch_receipt_sha256':sha256(root/'arms/scratch_encoder/ARM_COMPLETE.json'),
    'warm_runtime_code_commit':json.load(open(root/'arms/warm_start/ARM_COMPLETE.json')).get('runtime_code_commit',sys.argv[4]),
    'scratch_runtime_code_commit':json.load(open(root/'arms/scratch_encoder/ARM_COMPLETE.json')).get('runtime_code_commit',sys.argv[4]),
    'training_reused':True,
    'baseline_reused':True,
    'truth_used':False,
}
if path.exists() and json.load(open(path)) != payload:
    raise SystemExit('existing gate recovery authorization differs')
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY

scancel "$OLD_SUBMIT_EVALUATION_JOB" 2>/dev/null || true
EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,NPE_ROOT=$NPE_ROOT,CACHE_ROOT=$CACHE_ROOT,BENCHMARK_ENV=$BENCHMARK_ENV,BASELINE_ENV=$BASELINE_ENV,NPE_GATE_RECOVERY_RECEIPT=$AUTHORIZATION,NPE_FAILED_GATE_JOB=$FAILED_GATE_JOB,NPE_STAGE4_CODE_COMMIT=$CODE_COMMIT"
GATE_RAW=$(sbatch --parsable \
  --output="$NPE_LOG_ROOT/gate-finalizer-recovery-%j.out" \
  --error="$NPE_LOG_ROOT/gate-finalizer-recovery-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_frozen_parent_npe_gate.slurm")
GATE_JOB="${GATE_RAW%%;*}"
SUBMIT_RAW=$(sbatch --parsable --dependency="afterok:$GATE_JOB" \
  --output="$NPE_LOG_ROOT/submit-evaluation-final-recovery-%j.out" \
  --error="$NPE_LOG_ROOT/submit-evaluation-final-recovery-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_frozen_parent_npe_submit_evaluation.slurm")
SUBMIT_EVALUATION_JOB="${SUBMIT_RAW%%;*}"
ORIGINAL_ARM_JOB="${ORIGINAL_ARM_JOB:-$ARM_JOB}"
SCRATCH_RECOVERY_JOB="${SCRATCH_RECOVERY_JOB:-}"
ALL_JOBS="$ORIGINAL_ARM_JOB${SCRATCH_RECOVERY_JOB:+,$SCRATCH_RECOVERY_JOB},$GATE_JOB,$SUBMIT_EVALUATION_JOB"

printf 'export ORIGINAL_ARM_JOB=%q\nexport ARM_JOB=%q\nexport SCRATCH_RECOVERY_JOB=%q\nexport GATE_JOB=%q\nexport SUBMIT_EVALUATION_JOB=%q\nexport ALL_JOBS=%q\nexport NPE_ROOT=%q\nexport NPE_LOG_ROOT=%q\nexport BASELINE_ENV=%q\nexport BENCHMARK_ENV=%q\nexport SOURCE_BENCHMARK_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport CACHE_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\nexport NPE_GATE_RECOVERY_RECEIPT=%q\nexport NPE_FAILED_GATE_JOB=%q\n' \
  "$ORIGINAL_ARM_JOB" "$ORIGINAL_ARM_JOB" "$SCRATCH_RECOVERY_JOB" \
  "$GATE_JOB" "$SUBMIT_EVALUATION_JOB" "$ALL_JOBS" "$NPE_ROOT" \
  "$NPE_LOG_ROOT" "$BASELINE_ENV" "$BENCHMARK_ENV" \
  "$SOURCE_BENCHMARK_ROOT" "$RECOVERY_ROOT" "$CACHE_ROOT" \
  "$JOB_REPO_DIR" "$CODE_COMMIT" "$AUTHORIZATION" "$FAILED_GATE_JOB" \
  > "$ENV_FILE"

echo "reused_arm_receipts=warm_start,scratch_encoder"
echo "failed_gate_job=$FAILED_GATE_JOB"
echo "replacement_gate_job=$GATE_JOB"
echo "replacement_stage4_submitter_job=$SUBMIT_EVALUATION_JOB"
echo "monitor: bash scripts/monitor_feniks_sc_drws_frozen_parent_npe.sh $ENV_FILE 30"
