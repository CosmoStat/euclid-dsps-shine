#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ENV_FILE="${1:-outputs/logs/feniks_sc_drws_topology_npe_pilot_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH or CACHE_ROOT}/feniks_sc_drws_runtime}"
RECOVER_TOPOLOGY_INTERNAL="${RECOVER_TOPOLOGY_INTERNAL:-0}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
ENV_FILE="$(realpath "$ENV_FILE")"
test -s "$ENV_FILE" || { echo "[topology-internal-recovery][error] missing environment: $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"
[[ "$RECOVER_TOPOLOGY_INTERNAL" == 1 ]] || {
  echo "[topology-internal-recovery][error] set RECOVER_TOPOLOGY_INTERNAL=1" >&2
  exit 2
}
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[topology-internal-recovery][error] tracked source changes must be committed" >&2
  exit 2
fi

MANIFEST="$PILOT_ROOT/RUN_MANIFEST.json"
test -s "$MANIFEST"
test ! -s "$PILOT_ROOT/TOPOLOGY_NPE_PILOT_COMPLETE.json" || {
  echo "[topology-internal-recovery][error] pilot already finalized" >&2
  exit 2
}
for arm in A B C; do
  for receipt in \
    "$PILOT_ROOT/validation/$arm/tracking_k256/summary/TRUTH_FREE_POSTERIOR_VALIDATION.json" \
    "$PILOT_ROOT/validation/$arm/support_k1024/summary/TRUTH_FREE_POSTERIOR_VALIDATION.json"; do
    test -s "$receipt" || { echo "[topology-internal-recovery][error] missing: $receipt" >&2; exit 2; }
  done
  test ! -s "$PILOT_ROOT/validation/$arm/VALIDATION_COMPLETE.json" || {
    echo "[topology-internal-recovery][error] validation $arm already finalized" >&2
    exit 2
  }
done

FAILED_VALIDATION_A_JOB="$VALIDATION_A_JOB"
FAILED_VALIDATION_B_JOB="$VALIDATION_B_JOB"
FAILED_VALIDATION_C_JOB="$VALIDATION_C_JOB"
FAILED_FINAL_JOB="$FINAL_JOB"
ORIGINAL_CODE_COMMIT="$CODE_COMMIT"
RECOVERY_CODE_COMMIT="$(git rev-parse HEAD)"
git merge-base --is-ancestor "$ORIGINAL_CODE_COMMIT" "$RECOVERY_CODE_COMMIT" || {
  echo "[topology-internal-recovery][error] recovery code does not descend from pilot code" >&2
  exit 2
}

JOB_REPO_DIR="${TOPOLOGY_INTERNAL_RECOVERY_CODE_ROOT:-$CACHE_ROOT/code/topology-internal-${RECOVERY_CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")" "$PILOT_LOG_ROOT"
if [[ -e "$JOB_REPO_DIR" ]]; then
  test "$(git -C "$JOB_REPO_DIR" rev-parse HEAD)" = "$RECOVERY_CODE_COMMIT"
else
  git worktree add --detach "$JOB_REPO_DIR" "$RECOVERY_CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  mkdir -p "$JOB_REPO_DIR/Data"
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi

AUTHORIZATION="$PILOT_ROOT/INTERNAL_VALIDATION_RECOVERY_AUTHORIZATION_${FAILED_VALIDATION_A_JOB}.json"
test ! -e "$AUTHORIZATION" || {
  echo "[topology-internal-recovery][error] recovery already submitted: $AUTHORIZATION" >&2
  exit 2
}
python - "$AUTHORIZATION" "$MANIFEST" "$ORIGINAL_CODE_COMMIT" "$RECOVERY_CODE_COMMIT" \
  "$FAILED_VALIDATION_A_JOB" "$FAILED_VALIDATION_B_JOB" "$FAILED_VALIDATION_C_JOB" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[2])
payload = {
    "status": "AUTHORIZED",
    "scope": "internal_validation_only",
    "reason": "load_context API mismatch after completed K256 and K1024 inference",
    "run_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "original_code_commit": sys.argv[3],
    "recovery_code_commit": sys.argv[4],
    "failed_validation_jobs": {"A": sys.argv[5], "B": sys.argv[6], "C": sys.argv[7]},
    "completed_inference_reused": True,
    "new_inference_submitted": False,
    "new_training_submitted": False,
    "truth_used": False,
    "scientific_promotion": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

scancel "$FAILED_FINAL_JOB" 2>/dev/null || true
COMMON="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,PILOT_ROOT=$PILOT_ROOT,CACHE_ROOT=$CACHE_ROOT,TOPOLOGY_INTERNAL_RECOVERY_AUTHORIZATION=$AUTHORIZATION"
VALIDATION_RAW=$(sbatch --parsable --array=0-2%3 \
  --output="$PILOT_LOG_ROOT/internal-recovery-%A_%a.out" \
  --error="$PILOT_LOG_ROOT/internal-recovery-%A_%a.err" \
  --export="$COMMON" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_internal_recovery_h100.slurm")
VALIDATION_JOB="${VALIDATION_RAW%%;*}"
FINAL_RAW=$(sbatch --parsable --dependency="afterok:$VALIDATION_JOB" \
  --output="$PILOT_LOG_ROOT/finalize-internal-recovery-%j.out" \
  --error="$PILOT_LOG_ROOT/finalize-internal-recovery-%j.err" \
  --export="$COMMON" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_finalize.slurm")
FINAL_JOB="${FINAL_RAW%%;*}"
VALIDATION_A_JOB="${VALIDATION_JOB}_0"
VALIDATION_B_JOB="${VALIDATION_JOB}_1"
VALIDATION_C_JOB="${VALIDATION_JOB}_2"
VALIDATION_JOBS="$VALIDATION_JOB"
ALL_JOBS="$B_JOB,$C_JOB,$VALIDATION_JOB,$FINAL_JOB"

printf 'export B_JOB=%q\nexport C_JOB=%q\nexport VALIDATION_A_JOB=%q\nexport VALIDATION_B_JOB=%q\nexport VALIDATION_C_JOB=%q\nexport VALIDATION_JOBS=%q\nexport FINAL_JOB=%q\nexport ALL_JOBS=%q\nexport PILOT_ROOT=%q\nexport PILOT_LOG_ROOT=%q\nexport SOURCE_NPE_ROOT=%q\nexport CACHE_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\nexport ORIGINAL_CODE_COMMIT=%q\nexport FAILED_VALIDATION_A_JOB=%q\nexport FAILED_VALIDATION_B_JOB=%q\nexport FAILED_VALIDATION_C_JOB=%q\nexport FAILED_FINAL_JOB=%q\nexport TOPOLOGY_INTERNAL_RECOVERY_AUTHORIZATION=%q\n' \
  "$B_JOB" "$C_JOB" "$VALIDATION_A_JOB" "$VALIDATION_B_JOB" "$VALIDATION_C_JOB" \
  "$VALIDATION_JOBS" "$FINAL_JOB" "$ALL_JOBS" "$PILOT_ROOT" "$PILOT_LOG_ROOT" \
  "$SOURCE_NPE_ROOT" "$CACHE_ROOT" "$JOB_REPO_DIR" "$RECOVERY_CODE_COMMIT" \
  "$ORIGINAL_CODE_COMMIT" "$FAILED_VALIDATION_A_JOB" "$FAILED_VALIDATION_B_JOB" \
  "$FAILED_VALIDATION_C_JOB" "$FAILED_FINAL_JOB" "$AUTHORIZATION" > "$ENV_FILE"

echo "completed inference reused: A/B/C K256 and K1024"
echo "internal_validation_job=$VALIDATION_JOB"
echo "final_job=$FINAL_JOB"
echo "latest_env=$ENV_FILE"
