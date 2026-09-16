#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
LATEST_ENV="${1:-outputs/logs/feniks_sc_drws_population_posterior_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -s "$LATEST_ENV" || {
  echo "[population-posterior-recovery][error] missing environment: $LATEST_ENV" >&2
  exit 2
}
source "$LATEST_ENV"
FAILED_FINAL_JOB="$FINAL_JOB"
MANIFEST="$POSTERIOR_ROOT/RUN_MANIFEST.json"
FINAL_RECEIPT="$POSTERIOR_ROOT/INDIVIDUAL_POSTERIOR_DIAGNOSTIC_COMPLETE.json"

test -s "$MANIFEST" || {
  echo "[population-posterior-recovery][error] missing run manifest: $MANIFEST" >&2
  exit 2
}
test ! -e "$FINAL_RECEIPT" || {
  echo "[population-posterior-recovery] final receipt already exists: $FINAL_RECEIPT"
  exit 0
}
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[population-posterior-recovery][error] tracked source changes are not committed" >&2
  exit 2
fi

readarray -t VALUES < <(python - "$MANIFEST" "$POSTERIOR_ROOT" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
records = manifest["cohort"]["shards"]
for record in records:
    shard = root / "shards" / f"shard_{int(record['shard']):05d}"
    done = shard / "DONE"
    receipt = shard / "SHARD_COMPLETE.json"
    if not done.is_file() or not receipt.is_file():
        raise SystemExit(f"incomplete inference shard: {shard}")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE" or payload.get("truth_used") is not False:
        raise SystemExit(f"invalid inference shard receipt: {receipt}")
print(manifest["code_commit"])
print(len(records))
PY
)
INFERENCE_CODE_COMMIT="${VALUES[0]}"
SHARDS="${VALUES[1]}"
CODE_COMMIT="$(git rev-parse HEAD)"
test "$CODE_COMMIT" != "$INFERENCE_CODE_COMMIT" || {
  echo "[population-posterior-recovery][error] checkout still has the failed finalizer code" >&2
  exit 2
}

JOB_REPO_DIR="${POPULATION_POSTERIOR_RECOVERY_CODE_ROOT:-$CACHE_ROOT/code/population-posterior-finalizer-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")" "$POSTERIOR_LOG_ROOT" "$CACHE_ROOT/jax"
if [[ -e "$JOB_REPO_DIR" ]]; then
  test "$(git -C "$JOB_REPO_DIR" rev-parse HEAD)" = "$CODE_COMMIT" || {
    echo "[population-posterior-recovery][error] code snapshot has wrong commit" >&2
    exit 2
  }
else
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi

RECOVERY_AUTH="$POSTERIOR_ROOT/FINALIZER_RECOVERY_${CODE_COMMIT:0:12}.json"
python - "$MANIFEST" "$RECOVERY_AUTH" "$INFERENCE_CODE_COMMIT" \
  "$CODE_COMMIT" "$FAILED_FINAL_JOB" "$SHARDS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
authorization = Path(sys.argv[2])
payload = {
    "status": "AUTHORIZED",
    "scope": "finalizer_only_nonfinite_json_recovery",
    "inference_code_commit": sys.argv[3],
    "finalizer_code_commit": sys.argv[4],
    "failed_finalizer_job": sys.argv[5],
    "completed_inference_shards": int(sys.argv[6]),
    "run_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "inference_shards_reused": True,
    "new_inference_submitted": False,
    "truth_boundary_unchanged": True,
    "reason": "strict JSON rejected a non-finite diagnostic scalar",
}
serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
if authorization.exists():
    existing = json.loads(authorization.read_text(encoding="utf-8"))
    if existing != payload:
        raise SystemExit(f"refusing to replace recovery authorization: {authorization}")
else:
    temporary = authorization.with_name(f".{authorization.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(authorization)
PY

EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,POSTERIOR_ROOT=$POSTERIOR_ROOT,CACHE_ROOT=$CACHE_ROOT,FINALIZER_RECOVERY_RECEIPT=$RECOVERY_AUTH"
FINAL_RAW=$(sbatch --parsable \
  --output="$POSTERIOR_LOG_ROOT/finalize-recovery-%j.out" \
  --error="$POSTERIOR_LOG_ROOT/finalize-recovery-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_posterior_finalize_h100.slurm")
FINAL_JOB="${FINAL_RAW%%;*}"
ALL_JOBS="$INFERENCE_JOB,$FINAL_JOB"
RECOVERY_SUBMISSION="$POSTERIOR_ROOT/FINALIZER_RECOVERY_SUBMISSION_${FINAL_JOB}.json"
python - "$RECOVERY_SUBMISSION" "$RECOVERY_AUTH" "$INFERENCE_JOB" \
  "$FAILED_FINAL_JOB" "$FINAL_JOB" "$CODE_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "status": "SUBMITTED",
    "authorization": sys.argv[2],
    "reused_inference_job": sys.argv[3],
    "failed_finalizer_job": sys.argv[4],
    "recovery_finalizer_job": sys.argv[5],
    "finalizer_code_commit": sys.argv[6],
    "new_inference_submitted": False,
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf 'export INFERENCE_JOB=%q\nexport FINAL_JOB=%q\nexport ALL_JOBS=%q\nexport POSTERIOR_ROOT=%q\nexport POSTERIOR_LOG_ROOT=%q\nexport SOURCE_BENCHMARK_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\nexport FINALIZER_RECOVERY_RECEIPT=%q\nexport FAILED_FINAL_JOB=%q\n' \
  "$INFERENCE_JOB" "$FINAL_JOB" "$ALL_JOBS" "$POSTERIOR_ROOT" \
  "$POSTERIOR_LOG_ROOT" "$SOURCE_BENCHMARK_ROOT" "$RECOVERY_ROOT" \
  "$JOB_REPO_DIR" "$CODE_COMMIT" "$RECOVERY_AUTH" "$FAILED_FINAL_JOB" \
  > "$LATEST_ENV"

echo "reused_inference_job=$INFERENCE_JOB ($SHARDS shards complete)"
echo "failed_finalizer_job=$FAILED_FINAL_JOB"
echo "recovery_finalizer_job=$FINAL_JOB"
echo "authorization=$RECOVERY_AUTH"
echo "latest_env=$LATEST_ENV"
echo "monitor: bash scripts/monitor_feniks_sc_drws_population_posterior.sh 30"
