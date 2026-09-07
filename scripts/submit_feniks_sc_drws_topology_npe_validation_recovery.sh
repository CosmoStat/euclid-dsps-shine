#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ENV_FILE="${1:-outputs/logs/feniks_sc_drws_topology_npe_pilot_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH or CACHE_ROOT}/feniks_sc_drws_runtime}"
RECOVER_VALIDATION_TIMEOUTS="${RECOVER_VALIDATION_TIMEOUTS:-0}"
VALIDATION_TIME="${TOPOLOGY_VALIDATION_RECOVERY_TIME:-08:00:00}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
ENV_FILE="$(realpath "$ENV_FILE")"
test -s "$ENV_FILE" || {
  echo "[topology-npe-recovery][error] missing environment: $ENV_FILE" >&2
  exit 2
}
source "$ENV_FILE"
if [[ "$RECOVER_VALIDATION_TIMEOUTS" != 1 ]]; then
  echo "[topology-npe-recovery][error] set RECOVER_VALIDATION_TIMEOUTS=1" >&2
  exit 2
fi
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[topology-npe-recovery][error] tracked source changes must be committed" >&2
  exit 2
fi

FAILED_VALIDATION_A_JOB="$VALIDATION_A_JOB"
FAILED_VALIDATION_B_JOB="$VALIDATION_B_JOB"
FAILED_VALIDATION_C_JOB="$VALIDATION_C_JOB"
FAILED_FINAL_JOB="$FINAL_JOB"
MANIFEST="$PILOT_ROOT/RUN_MANIFEST.json"
FINAL_RECEIPT="$PILOT_ROOT/TOPOLOGY_NPE_PILOT_COMPLETE.json"
RECOVERY_SUBMITTER_COMMIT="$(git rev-parse HEAD)"

test -s "$MANIFEST" || {
  echo "[topology-npe-recovery][error] missing manifest: $MANIFEST" >&2
  exit 2
}
test ! -e "$FINAL_RECEIPT" || {
  echo "[topology-npe-recovery][error] final receipt already exists: $FINAL_RECEIPT" >&2
  exit 2
}
for arm in B C; do
  test -s "$PILOT_ROOT/arms/$arm/ARM_COMPLETE.json" || {
    echo "[topology-npe-recovery][error] arm $arm is not complete" >&2
    exit 2
  }
done
for arm in A B C; do
  test ! -e "$PILOT_ROOT/validation/$arm/VALIDATION_COMPLETE.json" || {
    echo "[topology-npe-recovery][error] validation $arm already complete" >&2
    exit 2
  }
done

python - "$MANIFEST" "$PILOT_ROOT" "$JOB_REPO_DIR" "$CODE_COMMIT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
job_repo = Path(sys.argv[3])
expected_commit = sys.argv[4]
if manifest["code_commit"] != expected_commit:
    raise SystemExit("pilot environment and manifest commits differ")
actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=job_repo, text=True
).strip()
if actual_commit != expected_commit:
    raise SystemExit("immutable pilot code snapshot has the wrong commit")
for arm in ("B", "C"):
    receipt = json.loads(
        (root / "arms" / arm / "ARM_COMPLETE.json").read_text(encoding="utf-8")
    )
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("truth_used_for_training_or_checkpoint_selection") is not False
        or receipt.get("prior_bitwise_unchanged") is not True
    ):
        raise SystemExit(f"arm {arm} is not a certified frozen-prior result")
PY

RECOVERY_RECEIPT="$PILOT_ROOT/VALIDATION_TIMEOUT_RECOVERY_SUBMISSION.json"
test ! -e "$RECOVERY_RECEIPT" || {
  echo "[topology-npe-recovery][error] recovery already submitted: $RECOVERY_RECEIPT" >&2
  exit 2
}

scancel "$FAILED_FINAL_JOB" 2>/dev/null || true
COMMON="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,PILOT_ROOT=$PILOT_ROOT,CACHE_ROOT=$CACHE_ROOT"
VALIDATION_A_RAW=$(sbatch --parsable --time="$VALIDATION_TIME" --array=0 \
  --output="$PILOT_LOG_ROOT/validation-recovery-%A_%a.out" \
  --error="$PILOT_LOG_ROOT/validation-recovery-%A_%a.err" \
  --export="$COMMON" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_validate_h100.slurm")
VALIDATION_A_JOB="${VALIDATION_A_RAW%%;*}"
VALIDATION_B_RAW=$(sbatch --parsable --time="$VALIDATION_TIME" --array=1 \
  --output="$PILOT_LOG_ROOT/validation-recovery-%A_%a.out" \
  --error="$PILOT_LOG_ROOT/validation-recovery-%A_%a.err" \
  --export="$COMMON" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_validate_h100.slurm")
VALIDATION_B_JOB="${VALIDATION_B_RAW%%;*}"
VALIDATION_C_RAW=$(sbatch --parsable --time="$VALIDATION_TIME" --array=2 \
  --output="$PILOT_LOG_ROOT/validation-recovery-%A_%a.out" \
  --error="$PILOT_LOG_ROOT/validation-recovery-%A_%a.err" \
  --export="$COMMON" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_validate_h100.slurm")
VALIDATION_C_JOB="${VALIDATION_C_RAW%%;*}"
VALIDATION_JOBS="$VALIDATION_A_JOB,$VALIDATION_B_JOB,$VALIDATION_C_JOB"
FINAL_RAW=$(sbatch --parsable \
  --dependency="afterok:$VALIDATION_A_JOB:$VALIDATION_B_JOB:$VALIDATION_C_JOB" \
  --output="$PILOT_LOG_ROOT/finalize-recovery-%j.out" \
  --error="$PILOT_LOG_ROOT/finalize-recovery-%j.err" \
  --export="$COMMON" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_finalize.slurm")
FINAL_JOB="${FINAL_RAW%%;*}"
ALL_JOBS="$B_JOB,$C_JOB,$VALIDATION_JOBS,$FINAL_JOB"

python - "$RECOVERY_RECEIPT" "$MANIFEST" "$PILOT_ROOT" \
  "$FAILED_VALIDATION_A_JOB" "$FAILED_VALIDATION_B_JOB" \
  "$FAILED_VALIDATION_C_JOB" "$FAILED_FINAL_JOB" "$VALIDATION_A_JOB" \
  "$VALIDATION_B_JOB" "$VALIDATION_C_JOB" "$FINAL_JOB" "$CODE_COMMIT" \
  "$RECOVERY_SUBMITTER_COMMIT" "$VALIDATION_TIME" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = Path(sys.argv[2])
root = Path(sys.argv[3])
payload = {
    "status": "SUBMITTED",
    "scope": "validation_timeouts_only",
    "pilot_root": str(root.resolve()),
    "run_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "failed_validation_jobs": {
        "A": sys.argv[4],
        "B": sys.argv[5],
        "C": sys.argv[6],
    },
    "failed_finalizer_job": sys.argv[7],
    "replacement_validation_jobs": {
        "A": sys.argv[8],
        "B": sys.argv[9],
        "C": sys.argv[10],
    },
    "replacement_finalizer_job": sys.argv[11],
    "pilot_runtime_code_commit": sys.argv[12],
    "recovery_submitter_commit": sys.argv[13],
    "validation_time_limit": sys.argv[14],
    "training_reused": True,
    "new_training_submitted": False,
    "truth_used": False,
    "scientific_promotion": False,
}
path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf 'export B_JOB=%q\nexport C_JOB=%q\nexport VALIDATION_A_JOB=%q\nexport VALIDATION_B_JOB=%q\nexport VALIDATION_C_JOB=%q\nexport VALIDATION_JOBS=%q\nexport FINAL_JOB=%q\nexport ALL_JOBS=%q\nexport PILOT_ROOT=%q\nexport PILOT_LOG_ROOT=%q\nexport SOURCE_NPE_ROOT=%q\nexport CACHE_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\nexport FAILED_VALIDATION_A_JOB=%q\nexport FAILED_VALIDATION_B_JOB=%q\nexport FAILED_VALIDATION_C_JOB=%q\nexport FAILED_FINAL_JOB=%q\nexport VALIDATION_TIMEOUT_RECOVERY_RECEIPT=%q\nexport RECOVERY_SUBMITTER_COMMIT=%q\n' \
  "$B_JOB" "$C_JOB" "$VALIDATION_A_JOB" "$VALIDATION_B_JOB" \
  "$VALIDATION_C_JOB" "$VALIDATION_JOBS" "$FINAL_JOB" "$ALL_JOBS" \
  "$PILOT_ROOT" "$PILOT_LOG_ROOT" "$SOURCE_NPE_ROOT" "$CACHE_ROOT" \
  "$JOB_REPO_DIR" "$CODE_COMMIT" "$FAILED_VALIDATION_A_JOB" \
  "$FAILED_VALIDATION_B_JOB" "$FAILED_VALIDATION_C_JOB" \
  "$FAILED_FINAL_JOB" "$RECOVERY_RECEIPT" "$RECOVERY_SUBMITTER_COMMIT" \
  > "$ENV_FILE"

echo "training_reused=B,C"
echo "validation_A_job=$VALIDATION_A_JOB"
echo "validation_B_job=$VALIDATION_B_JOB"
echo "validation_C_job=$VALIDATION_C_JOB"
echo "final_job=$FINAL_JOB"
echo "validation_time_limit=$VALIDATION_TIME"
echo "latest_env=$ENV_FILE"
echo "monitor: bash scripts/monitor_feniks_sc_drws_topology_npe_pilot.sh $ENV_FILE 30"
