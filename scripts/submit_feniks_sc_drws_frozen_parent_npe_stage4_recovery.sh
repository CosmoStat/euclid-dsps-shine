#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ENV_FILE="${1:-outputs/logs/feniks_sc_drws_frozen_parent_npe_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
RECOVER_STAGE4_ONLY="${RECOVER_STAGE4_ONLY:-0}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH or CACHE_ROOT}/feniks_sc_drws_runtime}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
ENV_FILE="$(realpath "$ENV_FILE")"
test -s "$ENV_FILE" || {
  echo "[npe-stage4-recovery][error] missing: $ENV_FILE" >&2
  exit 2
}
source "$ENV_FILE"
if [[ "$RECOVER_STAGE4_ONLY" != 1 ]]; then
  echo "[npe-stage4-recovery][error] set RECOVER_STAGE4_ONLY=1" >&2
  exit 2
fi
WINNER="$NPE_ROOT/NPE_WINNER_FROZEN.json"
test -s "$WINNER" || {
  echo "[npe-stage4-recovery][error] frozen winner is missing" >&2
  exit 2
}
test "$(python - "$WINNER" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('status'))
PY
)" = FROZEN || {
  echo "[npe-stage4-recovery][error] winner receipt is not FROZEN" >&2
  exit 2
}
for path in \
  "$NPE_ROOT/stage4_full.env" \
  "$NPE_ROOT/stage4_support.env" \
  "$NPE_ROOT/downstream.env" \
  "$NPE_ROOT/evaluation/full_test_k256/SUBMISSION.json" \
  "$NPE_ROOT/evaluation/support_k1024/SUBMISSION.json"; do
  test ! -s "$path" || {
    echo "[npe-stage4-recovery][error] partial stage-4 submission exists: $path" >&2
    exit 2
  }
done
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[npe-stage4-recovery][error] tracked source changes are not committed" >&2
  exit 2
fi

FAILED_SUBMITTER_JOB="$SUBMIT_EVALUATION_JOB"
CODE_COMMIT="$(git rev-parse HEAD)"
WINNER_FINALIZER_COMMIT="$(python - "$WINNER" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(x['runtime_provenance']['finalizer_code_commit'])
PY
)"
git merge-base --is-ancestor "$WINNER_FINALIZER_COMMIT" "$CODE_COMMIT" || {
  echo "[npe-stage4-recovery][error] stage-4 code does not descend from winner finalizer" >&2
  exit 2
}

JOB_REPO_DIR="${FROZEN_NPE_STAGE4_CODE_ROOT:-$CACHE_ROOT/code/frozen-npe-stage4-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")" "$NPE_LOG_ROOT"
if [[ -e "$JOB_REPO_DIR" ]]; then
  test "$(git -C "$JOB_REPO_DIR" rev-parse HEAD)" = "$CODE_COMMIT" || {
    echo "[npe-stage4-recovery][error] code snapshot has wrong commit" >&2
    exit 2
  }
else
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  mkdir -p "$JOB_REPO_DIR/Data"
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi

AUTHORIZATION="$NPE_ROOT/STAGE4_RECOVERY_AUTHORIZATION_${FAILED_SUBMITTER_JOB}.json"
python - "$AUTHORIZATION" "$NPE_ROOT" "$WINNER" "$WINNER_FINALIZER_COMMIT" \
  "$CODE_COMMIT" "$FAILED_SUBMITTER_JOB" <<'PY'
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
    'scope':'stage4_submitter_only_no_git_binary',
    'reason':'CPU runtime has no git executable after module purge',
    'npe_root':str(root),
    'winner_receipt_sha256':sha256(sys.argv[3]),
    'winner_finalizer_code_commit':sys.argv[4],
    'stage4_code_commit':sys.argv[5],
    'failed_submitter_job':sys.argv[6],
    'training_reused':True,
    'baseline_reused':True,
    'stage4_inference_reused':False,
    'truth_used':False,
}
if path.exists() and json.load(open(path)) != payload:
    raise SystemExit('existing stage-4 recovery authorization differs')
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY

scancel "$FAILED_SUBMITTER_JOB" 2>/dev/null || true
EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,NPE_ROOT=$NPE_ROOT,CACHE_ROOT=$CACHE_ROOT,BENCHMARK_ENV=$BENCHMARK_ENV,BASELINE_ENV=$BASELINE_ENV,NPE_STAGE4_CODE_COMMIT=$CODE_COMMIT,NPE_STAGE4_RECOVERY_RECEIPT=$AUTHORIZATION,NPE_FAILED_SUBMITTER_JOB=$FAILED_SUBMITTER_JOB"
SUBMIT_RAW=$(sbatch --parsable \
  --output="$NPE_LOG_ROOT/submit-evaluation-stage4-recovery-%j.out" \
  --error="$NPE_LOG_ROOT/submit-evaluation-stage4-recovery-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_frozen_parent_npe_submit_evaluation.slurm")
SUBMIT_EVALUATION_JOB="${SUBMIT_RAW%%;*}"
ORIGINAL_ARM_JOB="${ORIGINAL_ARM_JOB:-$ARM_JOB}"
SCRATCH_RECOVERY_JOB="${SCRATCH_RECOVERY_JOB:-}"
ALL_JOBS="$ORIGINAL_ARM_JOB${SCRATCH_RECOVERY_JOB:+,$SCRATCH_RECOVERY_JOB},$GATE_JOB,$SUBMIT_EVALUATION_JOB"

printf 'export ORIGINAL_ARM_JOB=%q\nexport ARM_JOB=%q\nexport SCRATCH_RECOVERY_JOB=%q\nexport GATE_JOB=%q\nexport SUBMIT_EVALUATION_JOB=%q\nexport ALL_JOBS=%q\nexport NPE_ROOT=%q\nexport NPE_LOG_ROOT=%q\nexport BASELINE_ENV=%q\nexport BENCHMARK_ENV=%q\nexport SOURCE_BENCHMARK_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport CACHE_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\nexport NPE_STAGE4_RECOVERY_RECEIPT=%q\nexport NPE_FAILED_SUBMITTER_JOB=%q\n' \
  "$ORIGINAL_ARM_JOB" "$ORIGINAL_ARM_JOB" "$SCRATCH_RECOVERY_JOB" \
  "$GATE_JOB" "$SUBMIT_EVALUATION_JOB" "$ALL_JOBS" "$NPE_ROOT" \
  "$NPE_LOG_ROOT" "$BASELINE_ENV" "$BENCHMARK_ENV" \
  "$SOURCE_BENCHMARK_ROOT" "$RECOVERY_ROOT" "$CACHE_ROOT" \
  "$JOB_REPO_DIR" "$CODE_COMMIT" "$AUTHORIZATION" \
  "$FAILED_SUBMITTER_JOB" > "$ENV_FILE"

python - "$NPE_ROOT/STAGE4_SUBMITTER_RECOVERY_SUBMISSION.json" \
  "$FAILED_SUBMITTER_JOB" "$SUBMIT_EVALUATION_JOB" "$CODE_COMMIT" \
  "$AUTHORIZATION" <<'PY'
import json,sys
from pathlib import Path
payload={
    'status':'SUBMITTED',
    'scope':'stage4_submitter_only_no_git_binary',
    'failed_submitter_job':sys.argv[2],
    'replacement_submitter_job':sys.argv[3],
    'runtime_code_commit':sys.argv[4],
    'authorization':sys.argv[5],
    'training_reused':True,
    'baseline_reused':True,
    'stage4_inference_reused':False,
    'truth_used':False,
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY

echo "frozen_winner_reused=warm_start"
echo "failed_submitter_job=$FAILED_SUBMITTER_JOB"
echo "replacement_stage4_submitter_job=$SUBMIT_EVALUATION_JOB"
echo "monitor: bash scripts/monitor_feniks_sc_drws_frozen_parent_npe.sh $ENV_FILE 30"
