#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
SOURCE_ENV="${1:-outputs/logs/feniks_sc_drws_population_projection_continuation_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
BENCHMARK_PASSES="${BENCHMARK_PASSES:-32}"
BENCHMARK_PATIENCE="${BENCHMARK_PATIENCE:-6}"
BENCHMARK_PEAK_LEARNING_RATE="${BENCHMARK_PEAK_LEARNING_RATE:-5.0e-5}"
BENCHMARK_FINAL_LEARNING_RATE="${BENCHMARK_FINAL_LEARNING_RATE:-1.0e-6}"
BENCHMARK_PRIOR_SAMPLES="${BENCHMARK_PRIOR_SAMPLES:-32768}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -s "$SOURCE_ENV" || {
  echo "[projection-benchmark][error] missing source environment: $SOURCE_ENV" >&2
  exit 2
}
source "$SOURCE_ENV"
SOURCE_PROJECTION_ROOT="${SOURCE_PROJECTION_ROOT_OVERRIDE:-$PROJECTION_ROOT}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$RECOVERY_ROOT/population_projection_epoch160_architecture_v1}"
BENCHMARK_LOG_ROOT="${BENCHMARK_LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$BENCHMARK_ROOT")}"

python - "$REPO_DIR" <<'PY'
import sys
from pathlib import Path

import euclid_dsps
import euclid_dsps.amortized.population_projection_benchmark as benchmark

repo = Path(sys.argv[1]).resolve()
for module in (euclid_dsps, benchmark):
    path = Path(module.__file__).resolve()
    if repo not in path.parents:
        raise SystemExit(f"active-checkout import required: {path} is outside {repo}")
print(f"[projection-benchmark] module={Path(benchmark.__file__).resolve()}")
PY

for path in \
  "$SOURCE_PROJECTION_ROOT/RUN_MANIFEST.json" \
  "$SOURCE_PROJECTION_ROOT/BETA_TARGET_COMPLETE.json" \
  "$SOURCE_PROJECTION_ROOT/PROJECTION_FIT_COMPLETE.json" \
  "$SOURCE_PROJECTION_ROOT/POPULATION_PROJECTION_COMPLETE.json"; do
  test -s "$path" || {
    echo "[projection-benchmark][error] missing source artifact: $path" >&2
    exit 2
  }
done
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[projection-benchmark][error] tracked source changes are not committed" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
mkdir -p "$BENCHMARK_LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs

python scripts/prepare_feniks_sc_drws_population_projection_benchmark.py \
  --source-root "$SOURCE_PROJECTION_ROOT" \
  --out "$BENCHMARK_ROOT" \
  --passes "$BENCHMARK_PASSES" \
  --patience "$BENCHMARK_PATIENCE" \
  --peak-learning-rate "$BENCHMARK_PEAK_LEARNING_RATE" \
  --final-learning-rate "$BENCHMARK_FINAL_LEARNING_RATE" \
  --prior-samples "$BENCHMARK_PRIOR_SAMPLES"
test ! -e "$BENCHMARK_ROOT/SUBMISSION.json" || {
  echo "[projection-benchmark][error] benchmark already submitted" >&2
  exit 2
}

JOB_REPO_DIR="${PROJECTION_BENCHMARK_CODE_ROOT:-$CACHE_ROOT/code/population-projection-benchmark-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")"
if [[ -e "$JOB_REPO_DIR" ]]; then
  test "$(git -C "$JOB_REPO_DIR" rev-parse HEAD)" = "$CODE_COMMIT" || {
    echo "[projection-benchmark][error] code snapshot has wrong commit" >&2
    exit 2
  }
else
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi
test -e "$JOB_REPO_DIR/Data/diffsky"

EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,BENCHMARK_ROOT=$BENCHMARK_ROOT,CACHE_ROOT=$CACHE_ROOT"
FIT_RAW=$(sbatch --parsable --array=0-2%3 \
  --output="$BENCHMARK_LOG_ROOT/fit-%A_%a.out" \
  --error="$BENCHMARK_LOG_ROOT/fit-%A_%a.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_projection_benchmark_fit_h100.slurm")
FIT_JOB="${FIT_RAW%%;*}"
VALIDATION_RAW=$(sbatch --parsable --dependency="afterok:$FIT_JOB" --array=0-3%4 \
  --output="$BENCHMARK_LOG_ROOT/validation-%A_%a.out" \
  --error="$BENCHMARK_LOG_ROOT/validation-%A_%a.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_projection_benchmark_validate_h100.slurm")
VALIDATION_JOB="${VALIDATION_RAW%%;*}"
GATE_RAW=$(sbatch --parsable --dependency="afterok:$VALIDATION_JOB" \
  --output="$BENCHMARK_LOG_ROOT/gate-%j.out" \
  --error="$BENCHMARK_LOG_ROOT/gate-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_projection_benchmark_finalize.slurm")
GATE_JOB="${GATE_RAW%%;*}"
CLOSURE_RAW=$(sbatch --parsable --dependency="afterok:$GATE_JOB" \
  --output="$BENCHMARK_LOG_ROOT/closure-%j.out" \
  --error="$BENCHMARK_LOG_ROOT/closure-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_projection_benchmark_closure_h100.slurm")
CLOSURE_JOB="${CLOSURE_RAW%%;*}"
ALL_JOBS="$FIT_JOB,$VALIDATION_JOB,$GATE_JOB,$CLOSURE_JOB"
LATEST="outputs/logs/feniks_sc_drws_population_projection_benchmark_latest.env"

printf 'export FIT_JOB=%q\nexport VALIDATION_JOB=%q\nexport GATE_JOB=%q\nexport CLOSURE_JOB=%q\nexport ALL_JOBS=%q\nexport BENCHMARK_ROOT=%q\nexport BENCHMARK_LOG_ROOT=%q\nexport SOURCE_PROJECTION_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\n' \
  "$FIT_JOB" "$VALIDATION_JOB" "$GATE_JOB" "$CLOSURE_JOB" "$ALL_JOBS" \
  "$BENCHMARK_ROOT" "$BENCHMARK_LOG_ROOT" "$SOURCE_PROJECTION_ROOT" \
  "$RECOVERY_ROOT" "$JOB_REPO_DIR" "$CODE_COMMIT" > "$LATEST"

python - "$BENCHMARK_ROOT/SUBMISSION.json" "$SOURCE_PROJECTION_ROOT" \
  "$FIT_JOB" "$VALIDATION_JOB" "$GATE_JOB" "$CLOSURE_JOB" "$ALL_JOBS" \
  "$CODE_COMMIT" "$JOB_REPO_DIR" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "status": "SUBMITTED",
    "source_projection_root": sys.argv[2],
    "fit_job": sys.argv[3],
    "validation_job": sys.argv[4],
    "truth_free_gate_job": sys.argv[5],
    "winner_closure_job": sys.argv[6],
    "all_jobs": sys.argv[7],
    "runtime_code_commit": sys.argv[8],
    "code_snapshot": sys.argv[9],
    "new_posterior_inference": False,
    "truth_used_before_winner_freeze": False,
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "fit_job=$FIT_JOB (3 parallel tasks x 4 H100; no DSPS inference)"
echo "validation_job=$VALIDATION_JOB (4 parallel tasks x 1 H100; truth-free)"
echo "truth_free_gate_job=$GATE_JOB"
echo "winner_closure_job=$CLOSURE_JOB (1 H100; PIT/coverage only after freeze)"
echo "root=$BENCHMARK_ROOT"
echo "latest_env=$LATEST"
echo "monitor: bash scripts/monitor_feniks_sc_drws_population_projection_benchmark.sh 30"
