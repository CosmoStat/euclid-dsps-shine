#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ENV_FILE="${1:-outputs/logs/feniks_sc_drws_population_projection_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
test -s "$ENV_FILE" || { echo "[projection-eval-recovery][error] missing $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"

OLD_EVALUATION_JOB="$EVALUATION_JOB"
FIT_RECEIPT="$PROJECTION_ROOT/PROJECTION_FIT_COMPLETE.json"
CODE_RECOVERY="$PROJECTION_ROOT/CODE_RECOVERY.json"
RECOVERY_SUBMISSION="$PROJECTION_ROOT/RECOVERY_SUBMISSION.json"
EVALUATION_RECOVERY="$PROJECTION_ROOT/EVALUATION_CODE_RECOVERY.json"
EVALUATION_SUBMISSION="$PROJECTION_ROOT/EVALUATION_RECOVERY_SUBMISSION.json"
OLD_ERROR="$(find "$PROJECTION_LOG_ROOT" -maxdepth 1 -type f \
  -name "*-${OLD_EVALUATION_JOB}.err" -print -quit)"

for path in "$FIT_RECEIPT" "$CODE_RECOVERY" "$RECOVERY_SUBMISSION" "$OLD_ERROR"; do
  test -s "$path" || { echo "[projection-eval-recovery][error] missing $path" >&2; exit 2; }
done
test ! -e "$PROJECTION_ROOT/POPULATION_PROJECTION_COMPLETE.json" || {
  echo "[projection-eval-recovery][error] projection evaluation is already complete" >&2
  exit 2
}
test ! -e "$EVALUATION_SUBMISSION" || {
  echo "[projection-eval-recovery][error] evaluation recovery was already submitted" >&2
  exit 2
}
grep -Fq "rank values must lie in [0, 1]" "$OLD_ERROR" || {
  echo "[projection-eval-recovery][error] old evaluation does not have the authorized CDF-roundoff failure" >&2
  exit 2
}
python - "$FIT_RECEIPT" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1]))
if receipt.get("status") != "COMPLETE" or receipt.get("truth_used") is not False:
    raise SystemExit("projection fit is not a complete truth-free receipt")
PY

if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[projection-eval-recovery][error] tracked source changes are not committed" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
PRIOR_RUNTIME_COMMIT="$(python - "$CODE_RECOVERY" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["runtime_code_commit"])
PY
)"
if [[ "$CODE_COMMIT" == "$PRIOR_RUNTIME_COMMIT" ]]; then
  echo "[projection-eval-recovery][error] pull the CDF-roundoff fix first" >&2
  exit 2
fi

JOB_REPO_DIR="${PROJECTION_EVALUATION_CODE_ROOT:-$CACHE_ROOT/code/population-projection-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")"
if [[ -e "$JOB_REPO_DIR" ]]; then
  test "$(git -C "$JOB_REPO_DIR" rev-parse HEAD)" = "$CODE_COMMIT" || {
    echo "[projection-eval-recovery][error] code snapshot has wrong commit" >&2
    exit 2
  }
else
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi
test -e "$JOB_REPO_DIR/Data/diffsky"

python - "$EVALUATION_RECOVERY" "$PROJECTION_ROOT" "$CODE_RECOVERY" \
  "$FIT_RECEIPT" "$OLD_EVALUATION_JOB" "$CODE_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

path = Path(sys.argv[1])
payload = {
    "status": "AUTHORIZED",
    "scope": "evaluation_only",
    "reason": "weighted empirical CDF exceeded one only by cumulative floating point roundoff",
    "projection_root": str(Path(sys.argv[2]).resolve()),
    "code_recovery_sha256": sha256(sys.argv[3]),
    "fit_receipt_sha256": sha256(sys.argv[4]),
    "failed_evaluation_job": sys.argv[5],
    "runtime_code_commit": sys.argv[6],
    "fit_reused": True,
    "beta_banks_reused": True,
    "truth_used": False,
}
if path.exists():
    if json.load(open(path)) != payload:
        raise SystemExit("existing evaluation-recovery receipt differs")
else:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
PY

scancel "$OLD_EVALUATION_JOB" 2>/dev/null || true
EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,PROJECTION_ROOT=$PROJECTION_ROOT,CACHE_ROOT=$CACHE_ROOT"
EVALUATION_RAW=$(sbatch --parsable \
  --output="$PROJECTION_LOG_ROOT/evaluation-recovery-%j.out" \
  --error="$PROJECTION_LOG_ROOT/evaluation-recovery-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_projection_evaluate_h100.slurm")
EVALUATION_JOB="${EVALUATION_RAW%%;*}"
ALL_JOBS="$FIT_JOB,$EVALUATION_JOB"
EVALUATION_ENV="outputs/logs/feniks_sc_drws_population_projection_evaluation_recovery_latest.env"
LATEST="outputs/logs/feniks_sc_drws_population_projection_latest.env"

printf 'export BETA_JOB=%q\nexport GATE_JOB=%q\nexport FIT_JOB=%q\nexport EVALUATION_JOB=%q\nexport ALL_JOBS=%q\nexport ORIGINAL_ALL_JOBS=%q\nexport PROJECTION_ROOT=%q\nexport PROJECTION_LOG_ROOT=%q\nexport SOURCE_BANK_VEM_ROOT=%q\nexport CALIBRATION_VEM_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\n' \
  "$BETA_JOB" "$GATE_JOB" "$FIT_JOB" "$EVALUATION_JOB" "$ALL_JOBS" \
  "${ORIGINAL_ALL_JOBS:-}" "$PROJECTION_ROOT" "$PROJECTION_LOG_ROOT" \
  "$SOURCE_BANK_VEM_ROOT" "$CALIBRATION_VEM_ROOT" "$RECOVERY_ROOT" \
  "$JOB_REPO_DIR" "$CODE_COMMIT" > "$EVALUATION_ENV"
cp "$EVALUATION_ENV" "$LATEST"

python - "$EVALUATION_SUBMISSION" "$OLD_EVALUATION_JOB" "$EVALUATION_JOB" \
  "$FIT_JOB" "$CODE_COMMIT" "$JOB_REPO_DIR" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "status": "SUBMITTED",
    "failed_evaluation_job": sys.argv[2],
    "recovery_evaluation_job": sys.argv[3],
    "reused_fit_job": sys.argv[4],
    "runtime_code_commit": sys.argv[5],
    "code_snapshot": sys.argv[6],
    "fit_reused": True,
    "beta_banks_reused": True,
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "reused_fit_job=$FIT_JOB"
echo "recovery_evaluation_job=$EVALUATION_JOB"
echo "latest_env=$LATEST"
echo "monitor: bash scripts/monitor_feniks_sc_drws_population_projection.sh 30"
