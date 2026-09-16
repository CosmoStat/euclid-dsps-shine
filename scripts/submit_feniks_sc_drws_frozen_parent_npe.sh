#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
BENCHMARK_ENV="${1:-outputs/logs/feniks_sc_drws_population_projection_benchmark_latest.env}"
BASELINE_ENV="${2:-outputs/logs/feniks_sc_drws_full_test_posterior_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
NPE_EPOCHS="${NPE_EPOCHS:-16}"
NPE_SEED="${NPE_SEED:-260904}"
NPE_CONFIG="${NPE_CONFIG:-configs/experiments/feniks_sc_drws_r29_frozen_parent_sleep_npe.yaml}"
NPE_ENV="${NPE_ENV:-outputs/logs/feniks_sc_drws_frozen_parent_npe_latest.env}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
BENCHMARK_ENV="$(realpath "$BENCHMARK_ENV")"
BASELINE_ENV="$(realpath "$BASELINE_ENV")"
test -s "$BENCHMARK_ENV" || {
  echo "[frozen-npe][error] missing benchmark env: $BENCHMARK_ENV" >&2
  exit 2
}
test -s "$BASELINE_ENV" || {
  echo "[frozen-npe][error] submit the current-q baseline first: $BASELINE_ENV" >&2
  exit 2
}
source "$BENCHMARK_ENV"
SOURCE_BENCHMARK_ROOT="$BENCHMARK_ROOT"
NPE_ROOT="${NPE_ROOT:-$RECOVERY_ROOT/frozen_parent_sleep_npe_v1}"
NPE_LOG_ROOT="${NPE_LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$NPE_ROOT")}"

python - "$REPO_DIR" <<'PY'
import sys
from pathlib import Path

import euclid_dsps
import euclid_dsps.amortized.population_vem as population_vem

repo = Path(sys.argv[1]).resolve()
for module in (euclid_dsps, population_vem):
    path = Path(module.__file__).resolve()
    if repo not in path.parents:
        raise SystemExit(f"active-checkout import required: {path} is outside {repo}")
print(f"[frozen-npe] module={Path(population_vem.__file__).resolve()}")
PY

for path in "$NPE_CONFIG" \
  "$SOURCE_BENCHMARK_ROOT/RUN_MANIFEST.json" \
  "$SOURCE_BENCHMARK_ROOT/TRUTH_FREE_ARCHITECTURE_WINNER.json" \
  "$SOURCE_BENCHMARK_ROOT/PROJECTION_FIT_COMPLETE.json" \
  "$SOURCE_BENCHMARK_ROOT/POPULATION_PROJECTION_COMPLETE.json"; do
  test -s "$path" || { echo "[frozen-npe][error] missing: $path" >&2; exit 2; }
done
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[frozen-npe][error] tracked source changes are not committed" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
mkdir -p "$NPE_LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs

python scripts/prepare_feniks_sc_drws_frozen_parent_npe.py \
  --benchmark-root "$SOURCE_BENCHMARK_ROOT" --config "$NPE_CONFIG" \
  --out "$NPE_ROOT" --epochs "$NPE_EPOCHS" --seed "$NPE_SEED"
test ! -e "$NPE_ROOT/SUBMISSION.json" || {
  echo "[frozen-npe][error] experiment already submitted: $NPE_ROOT" >&2
  exit 2
}

JOB_REPO_DIR="${FROZEN_NPE_CODE_ROOT:-$CACHE_ROOT/code/frozen-npe-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")"
if [[ -e "$JOB_REPO_DIR" ]]; then
  test "$(git -C "$JOB_REPO_DIR" rev-parse HEAD)" = "$CODE_COMMIT" || {
    echo "[frozen-npe][error] code snapshot has wrong commit" >&2
    exit 2
  }
else
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  mkdir -p "$JOB_REPO_DIR/Data"
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi
test -e "$JOB_REPO_DIR/Data/diffsky"

EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,NPE_ROOT=$NPE_ROOT,CACHE_ROOT=$CACHE_ROOT,BENCHMARK_ENV=$BENCHMARK_ENV,BASELINE_ENV=$BASELINE_ENV,NPE_STAGE4_CODE_COMMIT=$CODE_COMMIT"
ARM_RAW=$(sbatch --parsable --array=0-1%2 \
  --output="$NPE_LOG_ROOT/arm-%A_%a.out" \
  --error="$NPE_LOG_ROOT/arm-%A_%a.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_frozen_parent_npe_train_h100.slurm")
ARM_JOB="${ARM_RAW%%;*}"
GATE_RAW=$(sbatch --parsable --dependency="afterok:$ARM_JOB" \
  --output="$NPE_LOG_ROOT/gate-%j.out" \
  --error="$NPE_LOG_ROOT/gate-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_frozen_parent_npe_gate.slurm")
GATE_JOB="${GATE_RAW%%;*}"
SUBMIT_RAW=$(sbatch --parsable --dependency="afterok:$GATE_JOB" \
  --output="$NPE_LOG_ROOT/submit-evaluation-%j.out" \
  --error="$NPE_LOG_ROOT/submit-evaluation-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_frozen_parent_npe_submit_evaluation.slurm")
SUBMIT_EVALUATION_JOB="${SUBMIT_RAW%%;*}"
ALL_JOBS="$ARM_JOB,$GATE_JOB,$SUBMIT_EVALUATION_JOB"

mkdir -p "$(dirname "$NPE_ENV")"
printf 'export ARM_JOB=%q\nexport GATE_JOB=%q\nexport SUBMIT_EVALUATION_JOB=%q\nexport ALL_JOBS=%q\nexport NPE_ROOT=%q\nexport NPE_LOG_ROOT=%q\nexport BASELINE_ENV=%q\nexport BENCHMARK_ENV=%q\nexport SOURCE_BENCHMARK_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport CACHE_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\n' \
  "$ARM_JOB" "$GATE_JOB" "$SUBMIT_EVALUATION_JOB" "$ALL_JOBS" \
  "$NPE_ROOT" "$NPE_LOG_ROOT" "$BASELINE_ENV" "$BENCHMARK_ENV" \
  "$SOURCE_BENCHMARK_ROOT" "$RECOVERY_ROOT" "$CACHE_ROOT" \
  "$JOB_REPO_DIR" "$CODE_COMMIT" > "$NPE_ENV"

python - "$NPE_ROOT/SUBMISSION.json" "$ARM_JOB" "$GATE_JOB" \
  "$SUBMIT_EVALUATION_JOB" "$ALL_JOBS" "$CODE_COMMIT" \
  "$JOB_REPO_DIR" "$BASELINE_ENV" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "status": "SUBMITTED",
    "stage3_arm_array_job": sys.argv[2],
    "stage3_gate_job": sys.argv[3],
    "stage4_submitter_job": sys.argv[4],
    "initial_jobs": sys.argv[5],
    "runtime_code_commit": sys.argv[6],
    "code_snapshot": sys.argv[7],
    "stage2_baseline_environment": sys.argv[8],
    "truth_used_for_training_or_checkpoint_selection": False,
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "npe_arm_job=$ARM_JOB (2 arms x 4 H100)"
echo "npe_gate_job=$GATE_JOB"
echo "stage4_submitter_job=$SUBMIT_EVALUATION_JOB"
echo "root=$NPE_ROOT"
echo "latest_env=$NPE_ENV"
echo "monitor: bash scripts/monitor_feniks_sc_drws_frozen_parent_npe.sh $NPE_ENV 30"
