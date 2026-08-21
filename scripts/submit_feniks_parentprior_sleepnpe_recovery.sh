#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_TRAIN_ROOT="${SOURCE_TRAIN_ROOT:-${TRAIN_ROOT:?Source the original training env first}}"
SOURCE_MANIFEST_ROOT="${SOURCE_MANIFEST_ROOT:-${MANIFEST_ROOT:?Missing MANIFEST_ROOT}}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${RUN_ROOT:?Missing RUN_ROOT}}"
SOURCE_SMOKE_MANIFEST_ROOT="${SOURCE_SMOKE_MANIFEST_ROOT:-${SOURCE_RUN_ROOT}_smoke/manifests}"
START_EPOCH="${START_EPOCH:-25}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$SOURCE_TRAIN_ROOT/checkpoints/epoch_0024.eqx}"
FIXED_FEATURE_STATS="${FIXED_FEATURE_STATS:-$SOURCE_TRAIN_ROOT/feature_stats.json}"
RUN_TAG="${RUN_TAG:-$(basename "$SOURCE_RUN_ROOT")_recovery_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="outputs/runs/$RUN_TAG"
TRAIN_ROOT="$RUN_ROOT/train"
SMOKE_ROOT="${RUN_ROOT}_smoke"

cd "$REPO_DIR"
mkdir -p outputs/logs "$TRAIN_ROOT"
for path in "$INITIAL_CHECKPOINT" "${INITIAL_CHECKPOINT}.json" \
  "$FIXED_FEATURE_STATS" "$SOURCE_MANIFEST_ROOT/manifest.json" \
  "$SOURCE_SMOKE_MANIFEST_ROOT/manifest.json"; do
  test -s "$path" || { echo "[recovery][error] missing: $path" >&2; exit 2; }
done

python - "$INITIAL_CHECKPOINT.json" "$START_EPOCH" <<'PY'
import json
import sys
sidecar = json.load(open(sys.argv[1]))
expected = int(sys.argv[2]) - 1
if int(sidecar["epoch"]) != expected:
    raise SystemExit(f"restart sidecar epoch={sidecar['epoch']} expected={expected}")
PY

export REPO_DIR MINICONDA_PATH CONDA_ENV START_EPOCH INITIAL_CHECKPOINT
export FIXED_FEATURE_STATS
smoke_raw=$(sbatch --parsable --time=00:45:00 \
  --export="ALL,MANIFEST_ROOT=$SOURCE_SMOKE_MANIFEST_ROOT,TRAIN_ROOT=$SMOKE_ROOT/train,SMOKE=1" \
  scripts/feniks_parentprior_sleepnpe_h100.slurm)
SMOKE_JOB="${smoke_raw%%;*}"
train_raw=$(sbatch --parsable --dependency="afterok:$SMOKE_JOB" \
  --export="ALL,MANIFEST_ROOT=$SOURCE_MANIFEST_ROOT,TRAIN_ROOT=$TRAIN_ROOT,SMOKE=0" \
  scripts/feniks_parentprior_sleepnpe_h100.slurm)
TRAIN_JOB="${train_raw%%;*}"

env_file=outputs/logs/feniks_parentprior_sleepnpe_recovery_latest.env
printf 'export SMOKE_JOB=%q\nexport TRAIN_JOB=%q\nexport RUN_ROOT=%q\nexport TRAIN_ROOT=%q\nexport MANIFEST_ROOT=%q\nexport SOURCE_TRAIN_ROOT=%q\nexport INITIAL_CHECKPOINT=%q\nexport START_EPOCH=%q\n' \
  "$SMOKE_JOB" "$TRAIN_JOB" "$RUN_ROOT" "$TRAIN_ROOT" \
  "$SOURCE_MANIFEST_ROOT" "$SOURCE_TRAIN_ROOT" "$INITIAL_CHECKPOINT" \
  "$START_EPOCH" > "$env_file"
cp "$env_file" outputs/logs/feniks_parentprior_sleepnpe_latest.env

printf 'smoke_job=%s\ntrain_job=%s\nrun_root=%s\nrestart=%s epoch=%s\nlatest_env=%s\n' \
  "$SMOKE_JOB" "$TRAIN_JOB" "$RUN_ROOT" "$INITIAL_CHECKPOINT" \
  "$START_EPOCH" "$env_file"
