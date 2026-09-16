#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ENV_FILE="${1:-outputs/logs/feniks_sc_drws_population_projection_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
FIT_PASSES="${FIT_PASSES:-48}"
FIT_PATIENCE="${FIT_PATIENCE:-8}"
FIT_PEAK_LEARNING_RATE="${FIT_PEAK_LEARNING_RATE:-1.0e-5}"
FIT_FINAL_LEARNING_RATE="${FIT_FINAL_LEARNING_RATE:-5.0e-7}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -s "$ENV_FILE" || {
  echo "[projection-continuation][error] missing environment: $ENV_FILE" >&2
  exit 2
}
source "$ENV_FILE"
SOURCE_PROJECTION_ROOT="${SOURCE_PROJECTION_ROOT_OVERRIDE:-$PROJECTION_ROOT}"
CONTINUATION_ROOT="${CONTINUATION_ROOT:-$RECOVERY_ROOT/population_projection_epoch160_v2}"
CONTINUATION_LOG_ROOT="${CONTINUATION_LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$CONTINUATION_ROOT")}"

python - "$REPO_DIR" <<'PY'
import sys
from pathlib import Path

import euclid_dsps
import euclid_dsps.amortized.population_vem as population_vem

repo = Path(sys.argv[1]).resolve()
package = Path(euclid_dsps.__file__).resolve()
module = Path(population_vem.__file__).resolve()
if repo not in package.parents or repo not in module.parents:
    raise SystemExit(
        f"active-checkout import required: package={package} module={module} repo={repo}"
    )
print(f"[projection-continuation] euclid_dsps={package}")
print(f"[projection-continuation] population_vem={module}")
PY

for path in \
  "$SOURCE_PROJECTION_ROOT/RUN_MANIFEST.json" \
  "$SOURCE_PROJECTION_ROOT/BETA_TARGET_COMPLETE.json" \
  "$SOURCE_PROJECTION_ROOT/PROJECTION_FIT_COMPLETE.json" \
  "$SOURCE_PROJECTION_ROOT/POPULATION_PROJECTION_COMPLETE.json"; do
  test -s "$path" || {
    echo "[projection-continuation][error] missing source artifact: $path" >&2
    exit 2
  }
done
python - "$SOURCE_PROJECTION_ROOT/POPULATION_PROJECTION_COMPLETE.json" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1]))
if receipt.get("status") != "DIAGNOSTIC_COMPLETE":
    raise SystemExit("source population projection evaluation is not complete")
PY
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[projection-continuation][error] tracked source changes are not committed" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
mkdir -p "$CONTINUATION_LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs

python scripts/prepare_feniks_sc_drws_population_projection_continuation.py \
  --source-root "$SOURCE_PROJECTION_ROOT" \
  --out "$CONTINUATION_ROOT" \
  --passes "$FIT_PASSES" \
  --patience "$FIT_PATIENCE" \
  --peak-learning-rate "$FIT_PEAK_LEARNING_RATE" \
  --final-learning-rate "$FIT_FINAL_LEARNING_RATE"
test ! -e "$CONTINUATION_ROOT/SUBMISSION.json" || {
  echo "[projection-continuation][error] continuation already submitted" >&2
  exit 2
}

JOB_REPO_DIR="${PROJECTION_CONTINUATION_CODE_ROOT:-$CACHE_ROOT/code/population-projection-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")"
if [[ -e "$JOB_REPO_DIR" ]]; then
  test "$(git -C "$JOB_REPO_DIR" rev-parse HEAD)" = "$CODE_COMMIT" || {
    echo "[projection-continuation][error] code snapshot has wrong commit" >&2
    exit 2
  }
else
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi
test -e "$JOB_REPO_DIR/Data/diffsky"

EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,PROJECTION_ROOT=$CONTINUATION_ROOT,CACHE_ROOT=$CACHE_ROOT,FIT_PASSES=$FIT_PASSES,FIT_PATIENCE=$FIT_PATIENCE,FIT_PEAK_LEARNING_RATE=$FIT_PEAK_LEARNING_RATE,FIT_FINAL_LEARNING_RATE=$FIT_FINAL_LEARNING_RATE"
FIT_RAW=$(sbatch --parsable \
  --output="$CONTINUATION_LOG_ROOT/fit-%j.out" \
  --error="$CONTINUATION_LOG_ROOT/fit-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_projection_fit_h100.slurm")
FIT_JOB="${FIT_RAW%%;*}"
EVALUATION_RAW=$(sbatch --parsable --dependency="afterok:$FIT_JOB" \
  --output="$CONTINUATION_LOG_ROOT/evaluation-%j.out" \
  --error="$CONTINUATION_LOG_ROOT/evaluation-%j.err" \
  --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_projection_evaluate_h100.slurm")
EVALUATION_JOB="${EVALUATION_RAW%%;*}"
ALL_JOBS="$FIT_JOB,$EVALUATION_JOB"
LATEST="outputs/logs/feniks_sc_drws_population_projection_latest.env"
CONTINUATION_ENV="outputs/logs/feniks_sc_drws_population_projection_continuation_latest.env"

printf 'export FIT_JOB=%q\nexport EVALUATION_JOB=%q\nexport ALL_JOBS=%q\nexport PROJECTION_ROOT=%q\nexport PROJECTION_LOG_ROOT=%q\nexport SOURCE_PROJECTION_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\nexport FIT_PASSES=%q\nexport FIT_PATIENCE=%q\nexport FIT_PEAK_LEARNING_RATE=%q\nexport FIT_FINAL_LEARNING_RATE=%q\n' \
  "$FIT_JOB" "$EVALUATION_JOB" "$ALL_JOBS" "$CONTINUATION_ROOT" \
  "$CONTINUATION_LOG_ROOT" "$SOURCE_PROJECTION_ROOT" "$RECOVERY_ROOT" \
  "$JOB_REPO_DIR" "$CODE_COMMIT" "$FIT_PASSES" "$FIT_PATIENCE" \
  "$FIT_PEAK_LEARNING_RATE" "$FIT_FINAL_LEARNING_RATE" > "$CONTINUATION_ENV"
cp "$CONTINUATION_ENV" "$LATEST"

python - "$CONTINUATION_ROOT/SUBMISSION.json" "$SOURCE_PROJECTION_ROOT" \
  "$FIT_JOB" "$EVALUATION_JOB" "$CODE_COMMIT" "$JOB_REPO_DIR" \
  "$FIT_PASSES" "$FIT_PATIENCE" "$FIT_PEAK_LEARNING_RATE" \
  "$FIT_FINAL_LEARNING_RATE" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "status": "SUBMITTED",
    "source_projection_root": sys.argv[2],
    "fit_job": sys.argv[3],
    "evaluation_job": sys.argv[4],
    "runtime_code_commit": sys.argv[5],
    "code_snapshot": sys.argv[6],
    "passes": int(sys.argv[7]),
    "patience": int(sys.argv[8]),
    "peak_learning_rate": float(sys.argv[9]),
    "final_learning_rate": float(sys.argv[10]),
    "q_banks_reused": True,
    "beta_banks_reused": True,
    "new_posterior_inference": False,
    "truth_used_for_fit_or_checkpoint_selection": False,
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "fit_job=$FIT_JOB (4 H100; continued selected and parent flows)"
echo "evaluation_job=$EVALUATION_JOB (1 H100; frozen diagnostics)"
echo "source_root=$SOURCE_PROJECTION_ROOT"
echo "continuation_root=$CONTINUATION_ROOT"
echo "latest_env=$LATEST"
echo "monitor: bash scripts/monitor_feniks_sc_drws_population_projection.sh 30"
