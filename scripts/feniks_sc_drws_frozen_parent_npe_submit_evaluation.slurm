#!/bin/bash
#SBATCH --job-name=feniks_sc_npe_submit
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
NPE_ROOT="${NPE_ROOT:?NPE_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
BENCHMARK_ENV="${BENCHMARK_ENV:?BENCHMARK_ENV is required}"
BASELINE_ENV="${BASELINE_ENV:?BASELINE_ENV is required}"
NPE_STAGE4_CODE_COMMIT="${NPE_STAGE4_CODE_COMMIT:?NPE_STAGE4_CODE_COMMIT is required}"
NPE_STAGE4_RECOVERY_RECEIPT="${NPE_STAGE4_RECOVERY_RECEIPT:-}"
NPE_FAILED_SUBMITTER_JOB="${NPE_FAILED_SUBMITTER_JOB:-}"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1

cd "$REPO_DIR"
module purge
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
test -s "$NPE_ROOT/NPE_WINNER_FROZEN.json"
test -s "$BASELINE_ENV"
python - "$REPO_DIR" "$NPE_ROOT" "$NPE_STAGE4_CODE_COMMIT" \
  "$NPE_STAGE4_RECOVERY_RECEIPT" "$NPE_FAILED_SUBMITTER_JOB" <<'PY'
import json
import sys
from pathlib import Path

from euclid_dsps.amortized.population_vem import require_git_commit, sha256_file

repo = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
runtime_commit = sys.argv[3]
authorization_path = sys.argv[4]
failed_submitter = sys.argv[5]
require_git_commit(repo, runtime_commit)
if authorization_path:
    path = Path(authorization_path).resolve()
    authorization = json.loads(path.read_text(encoding="utf-8"))
    winner_path = root / "NPE_WINNER_FROZEN.json"
    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    expected = {
        "status": "AUTHORIZED",
        "scope": "stage4_submitter_only_no_git_binary",
        "npe_root": str(root),
        "winner_receipt_sha256": sha256_file(winner_path),
        "winner_finalizer_code_commit": winner["runtime_provenance"][
            "finalizer_code_commit"
        ],
        "stage4_code_commit": runtime_commit,
        "failed_submitter_job": failed_submitter,
        "training_reused": True,
        "baseline_reused": True,
        "stage4_inference_reused": False,
        "truth_used": False,
    }
    mismatches = {
        key: {"expected": value, "actual": authorization.get(key)}
        for key, value in expected.items()
        if authorization.get(key) != value
    }
    if mismatches:
        raise ValueError(f"invalid stage-4 recovery authorization: {mismatches}")
print(f"[frozen-npe-stage4] commit={runtime_commit} recovery={bool(authorization_path)}")
PY
export POPULATION_POSTERIOR_RUNTIME_COMMIT="$NPE_STAGE4_CODE_COMMIT"

FULL_ROOT="$NPE_ROOT/evaluation/full_test_k256"
SUPPORT_ROOT="$NPE_ROOT/evaluation/support_k1024"
FULL_ENV="$NPE_ROOT/stage4_full.env"
SUPPORT_ENV="$NPE_ROOT/stage4_support.env"

if [[ ! -s "$FULL_ENV" ]]; then
  POSTERIOR_ROOT="$FULL_ROOT" \
  POSTERIOR_LOG_ROOT="$CACHE_ROOT/slurm_logs/$(basename "$NPE_ROOT")-full" \
  POSTERIOR_ENV="$FULL_ENV" \
  POSTERIOR_MODEL_RECEIPT="$NPE_ROOT/NPE_WINNER_FROZEN.json" \
  POSTERIOR_OBJECTS=4706 POSTERIOR_SHARDS=32 POSTERIOR_MAX_PARALLEL=16 \
  POSTERIOR_PANELS=12 POSTERIOR_DRAWS=256 POSTERIOR_RESAMPLE_DRAWS=64 \
  POSTERIOR_OBJECT_BATCH_SIZE=16 POSTERIOR_PRIOR_DRAWS=512 \
  POPULATION_POSTERIOR_CODE_ROOT="$REPO_DIR" \
  REPO_DIR="$REPO_DIR" CACHE_ROOT="$CACHE_ROOT" \
    bash scripts/submit_feniks_sc_drws_population_posterior.sh "$BENCHMARK_ENV"
fi
source "$FULL_ENV"
NPE_FULL_INFERENCE_JOB="$INFERENCE_JOB"
NPE_FULL_FINAL_JOB="$FINAL_JOB"
NPE_FULL_JOBS="$ALL_JOBS"

if [[ ! -s "$SUPPORT_ENV" ]]; then
  POSTERIOR_ROOT="$SUPPORT_ROOT" \
  POSTERIOR_LOG_ROOT="$CACHE_ROOT/slurm_logs/$(basename "$NPE_ROOT")-support" \
  POSTERIOR_ENV="$SUPPORT_ENV" \
  POSTERIOR_MODEL_RECEIPT="$NPE_ROOT/NPE_WINNER_FROZEN.json" \
  POSTERIOR_OBJECTS=512 POSTERIOR_SHARDS=32 POSTERIOR_MAX_PARALLEL=16 \
  POSTERIOR_PANELS=12 POSTERIOR_DRAWS=1024 POSTERIOR_RESAMPLE_DRAWS=256 \
  POSTERIOR_OBJECT_BATCH_SIZE=8 POSTERIOR_PRIOR_DRAWS=512 \
  POPULATION_POSTERIOR_CODE_ROOT="$REPO_DIR" \
  REPO_DIR="$REPO_DIR" CACHE_ROOT="$CACHE_ROOT" \
    bash scripts/submit_feniks_sc_drws_population_posterior.sh "$BENCHMARK_ENV"
fi
source "$SUPPORT_ENV"
NPE_SUPPORT_INFERENCE_JOB="$INFERENCE_JOB"
NPE_SUPPORT_FINAL_JOB="$FINAL_JOB"
NPE_SUPPORT_JOBS="$ALL_JOBS"

source "$BASELINE_ENV"
CLOSURE_DEPENDENCY="$NPE_FULL_FINAL_JOB:$NPE_SUPPORT_FINAL_JOB"
if ! python - "$BASELINE_FULL_ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'INDIVIDUAL_POSTERIOR_DIAGNOSTIC_COMPLETE.json'
raise SystemExit(0 if p.is_file() and json.load(open(p)).get('status') == 'DIAGNOSTIC_COMPLETE' else 1)
PY
then
  CLOSURE_DEPENDENCY="$CLOSURE_DEPENDENCY:$BASELINE_FULL_FINAL_JOB"
fi
if ! python - "$BASELINE_SUPPORT_ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'INDIVIDUAL_POSTERIOR_DIAGNOSTIC_COMPLETE.json'
raise SystemExit(0 if p.is_file() and json.load(open(p)).get('status') == 'DIAGNOSTIC_COMPLETE' else 1)
PY
then
  CLOSURE_DEPENDENCY="$CLOSURE_DEPENDENCY:$BASELINE_SUPPORT_FINAL_JOB"
fi
EXPORTS="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,NPE_ROOT=$NPE_ROOT,BASELINE_ROOT=$BASELINE_ROOT,FULL_ROOT=$FULL_ROOT,SUPPORT_ROOT=$SUPPORT_ROOT"
CLOSURE_RAW=$(sbatch --parsable --dependency="afterok:$CLOSURE_DEPENDENCY" \
  --output="$CACHE_ROOT/slurm_logs/$(basename "$NPE_ROOT")-closure-%j.out" \
  --error="$CACHE_ROOT/slurm_logs/$(basename "$NPE_ROOT")-closure-%j.err" \
  --export="$EXPORTS" \
  "$REPO_DIR/scripts/feniks_sc_drws_frozen_parent_npe_closure.slurm")
CLOSURE_JOB="${CLOSURE_RAW%%;*}"
ALL_DOWNSTREAM_JOBS="$NPE_FULL_JOBS,$NPE_SUPPORT_JOBS,$CLOSURE_JOB"

printf 'export NPE_FULL_ROOT=%q\nexport NPE_SUPPORT_ROOT=%q\nexport NPE_FULL_ENV=%q\nexport NPE_SUPPORT_ENV=%q\nexport NPE_FULL_INFERENCE_JOB=%q\nexport NPE_FULL_FINAL_JOB=%q\nexport NPE_SUPPORT_INFERENCE_JOB=%q\nexport NPE_SUPPORT_FINAL_JOB=%q\nexport CLOSURE_JOB=%q\nexport ALL_DOWNSTREAM_JOBS=%q\n' \
  "$FULL_ROOT" "$SUPPORT_ROOT" "$FULL_ENV" "$SUPPORT_ENV" \
  "$NPE_FULL_INFERENCE_JOB" "$NPE_FULL_FINAL_JOB" \
  "$NPE_SUPPORT_INFERENCE_JOB" "$NPE_SUPPORT_FINAL_JOB" \
  "$CLOSURE_JOB" "$ALL_DOWNSTREAM_JOBS" > "$NPE_ROOT/downstream.env"

python - "$NPE_ROOT/STAGE4_SUBMISSION.json" "$NPE_FULL_INFERENCE_JOB" \
  "$NPE_FULL_FINAL_JOB" "$NPE_SUPPORT_INFERENCE_JOB" \
  "$NPE_SUPPORT_FINAL_JOB" "$CLOSURE_JOB" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "status": "SUBMITTED",
    "stage": 4,
    "full_test_k256_inference_job": sys.argv[2],
    "full_test_k256_final_job": sys.argv[3],
    "support_k1024_inference_job": sys.argv[4],
    "support_k1024_final_job": sys.argv[5],
    "stage5_closure_job": sys.argv[6],
    "truth_used_for_submission": False,
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "npe_full=$NPE_FULL_INFERENCE_JOB -> $NPE_FULL_FINAL_JOB"
echo "npe_support=$NPE_SUPPORT_INFERENCE_JOB -> $NPE_SUPPORT_FINAL_JOB"
echo "closure=$CLOSURE_JOB"
