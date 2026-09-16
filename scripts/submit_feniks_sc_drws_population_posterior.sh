#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
BENCHMARK_ENV="${1:-outputs/logs/feniks_sc_drws_population_projection_benchmark_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
POSTERIOR_OBJECTS="${POSTERIOR_OBJECTS:-64}"
POSTERIOR_SHARDS="${POSTERIOR_SHARDS:-8}"
POSTERIOR_PANELS="${POSTERIOR_PANELS:-8}"
POSTERIOR_DRAWS="${POSTERIOR_DRAWS:-1024}"
POSTERIOR_RESAMPLE_DRAWS="${POSTERIOR_RESAMPLE_DRAWS:-256}"
POSTERIOR_MAX_PARALLEL="${POSTERIOR_MAX_PARALLEL:-8}"
POSTERIOR_OBJECT_BATCH_SIZE="${POSTERIOR_OBJECT_BATCH_SIZE:-8}"
POSTERIOR_PRIOR_DRAWS="${POSTERIOR_PRIOR_DRAWS:-512}"
POSTERIOR_MODEL_RECEIPT="${POSTERIOR_MODEL_RECEIPT:-}"
POPULATION_POSTERIOR_RUNTIME_COMMIT="${POPULATION_POSTERIOR_RUNTIME_COMMIT:-}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -s "$BENCHMARK_ENV" || {
  echo "[population-posterior][error] missing benchmark environment: $BENCHMARK_ENV" >&2
  exit 2
}
source "$BENCHMARK_ENV"
SOURCE_BENCHMARK_ROOT="$BENCHMARK_ROOT"
POSTERIOR_ROOT="${POSTERIOR_ROOT:-$RECOVERY_ROOT/population_projection_epoch160_architecture_v2_individual_k1024_v1}"
POSTERIOR_LOG_ROOT="${POSTERIOR_LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$POSTERIOR_ROOT")}"

python - "$REPO_DIR" <<'PY'
import sys
from pathlib import Path

import euclid_dsps
import euclid_dsps.amortized.posthoc_calibration as posthoc

repo = Path(sys.argv[1]).resolve()
for module in (euclid_dsps, posthoc):
    path = Path(module.__file__).resolve()
    if repo not in path.parents:
        raise SystemExit(f"active-checkout import required: {path} is outside {repo}")
print(f"[population-posterior] module={Path(posthoc.__file__).resolve()}")
PY

for path in \
  "$SOURCE_BENCHMARK_ROOT/RUN_MANIFEST.json" \
  "$SOURCE_BENCHMARK_ROOT/TRUTH_FREE_ARCHITECTURE_WINNER.json" \
  "$SOURCE_BENCHMARK_ROOT/PROJECTION_FIT_COMPLETE.json" \
  "$SOURCE_BENCHMARK_ROOT/POPULATION_PROJECTION_COMPLETE.json"; do
  test -s "$path" || {
    echo "[population-posterior][error] missing benchmark artifact: $path" >&2
    exit 2
  }
done
if command -v git >/dev/null 2>&1; then
  if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
    echo "[population-posterior][error] tracked source changes are not committed" >&2
    exit 2
  fi
  CODE_COMMIT="$(git rev-parse HEAD)"
  if [[ -n "$POPULATION_POSTERIOR_RUNTIME_COMMIT" \
    && "$CODE_COMMIT" != "$POPULATION_POSTERIOR_RUNTIME_COMMIT" ]]; then
    echo "[population-posterior][error] authorized runtime commit mismatch" >&2
    exit 2
  fi
else
  test -n "$POPULATION_POSTERIOR_RUNTIME_COMMIT" || {
    echo "[population-posterior][error] git absent and no runtime commit supplied" >&2
    exit 2
  }
  CODE_COMMIT="$(python - "$REPO_DIR" "$POPULATION_POSTERIOR_RUNTIME_COMMIT" <<'PY'
import sys
from euclid_dsps.amortized.population_vem import require_git_commit
print(require_git_commit(sys.argv[1], sys.argv[2]))
PY
)"
fi
export POPULATION_POSTERIOR_RUNTIME_COMMIT="$CODE_COMMIT"
mkdir -p "$POSTERIOR_LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs

PREPARE_MODEL_ARGS=()
if [[ -n "$POSTERIOR_MODEL_RECEIPT" ]]; then
  PREPARE_MODEL_ARGS+=(--model-receipt "$POSTERIOR_MODEL_RECEIPT")
fi
python scripts/prepare_feniks_sc_drws_population_posterior.py \
  --benchmark-root "$SOURCE_BENCHMARK_ROOT" \
  --out "$POSTERIOR_ROOT" \
  --objects "$POSTERIOR_OBJECTS" \
  --shards "$POSTERIOR_SHARDS" \
  --panels "$POSTERIOR_PANELS" \
  --posterior-draws "$POSTERIOR_DRAWS" \
  --resample-draws "$POSTERIOR_RESAMPLE_DRAWS" \
  --object-batch-size "$POSTERIOR_OBJECT_BATCH_SIZE" \
  --prior-draws "$POSTERIOR_PRIOR_DRAWS" \
  "${PREPARE_MODEL_ARGS[@]}"
test ! -e "$POSTERIOR_ROOT/SUBMISSION.json" || {
  echo "[population-posterior][error] diagnostic already submitted" >&2
  exit 2
}

JOB_REPO_DIR="${POPULATION_POSTERIOR_CODE_ROOT:-$CACHE_ROOT/code/population-posterior-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")"
if [[ -e "$JOB_REPO_DIR" ]]; then
  python - "$JOB_REPO_DIR" "$CODE_COMMIT" <<'PY'
import sys
from euclid_dsps.amortized.population_vem import require_git_commit
require_git_commit(sys.argv[1], sys.argv[2])
PY
else
  command -v git >/dev/null 2>&1 || {
    echo "[population-posterior][error] cannot create code snapshot without git" >&2
    exit 2
  }
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  mkdir -p "$JOB_REPO_DIR/Data"
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi
test -e "$JOB_REPO_DIR/Data/diffsky"

LAST_TASK="$((POSTERIOR_SHARDS - 1))"
EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,POSTERIOR_ROOT=$POSTERIOR_ROOT,CACHE_ROOT=$CACHE_ROOT"
INFERENCE_RAW=$(sbatch --parsable \
  --array="0-${LAST_TASK}%${POSTERIOR_MAX_PARALLEL}" \
  --output="$POSTERIOR_LOG_ROOT/inference-%A_%a.out" \
  --error="$POSTERIOR_LOG_ROOT/inference-%A_%a.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_posterior_h100.slurm")
INFERENCE_JOB="${INFERENCE_RAW%%;*}"
FINAL_RAW=$(sbatch --parsable --dependency="afterok:$INFERENCE_JOB" \
  --output="$POSTERIOR_LOG_ROOT/finalize-%j.out" \
  --error="$POSTERIOR_LOG_ROOT/finalize-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_posterior_finalize_h100.slurm")
FINAL_JOB="${FINAL_RAW%%;*}"
ALL_JOBS="$INFERENCE_JOB,$FINAL_JOB"
LATEST="${POSTERIOR_ENV:-outputs/logs/feniks_sc_drws_population_posterior_latest.env}"
mkdir -p "$(dirname "$LATEST")"

printf 'export INFERENCE_JOB=%q\nexport FINAL_JOB=%q\nexport ALL_JOBS=%q\nexport POSTERIOR_ROOT=%q\nexport POSTERIOR_LOG_ROOT=%q\nexport SOURCE_BENCHMARK_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\n' \
  "$INFERENCE_JOB" "$FINAL_JOB" "$ALL_JOBS" "$POSTERIOR_ROOT" \
  "$POSTERIOR_LOG_ROOT" "$SOURCE_BENCHMARK_ROOT" "$RECOVERY_ROOT" \
  "$JOB_REPO_DIR" "$CODE_COMMIT" > "$LATEST"

python - "$POSTERIOR_ROOT/SUBMISSION.json" "$INFERENCE_JOB" "$FINAL_JOB" \
  "$ALL_JOBS" "$CODE_COMMIT" "$JOB_REPO_DIR" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "status": "SUBMITTED",
    "inference_array_job": sys.argv[2],
    "finalizer_job": sys.argv[3],
    "all_jobs": sys.argv[4],
    "runtime_code_commit": sys.argv[5],
    "code_snapshot": sys.argv[6],
    "truth_used_for_inference_or_support": False,
    "truth_used_for_final_closure": True,
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "inference_job=$INFERENCE_JOB (${POSTERIOR_SHARDS} parallel one-H100 shards)"
echo "final_job=$FINAL_JOB"
echo "root=$POSTERIOR_ROOT"
echo "latest_env=$LATEST"
echo "monitor: bash scripts/monitor_feniks_sc_drws_population_posterior.sh $LATEST 30"
