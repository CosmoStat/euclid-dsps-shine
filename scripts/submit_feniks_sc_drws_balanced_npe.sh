#!/bin/bash
set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
SOURCE_ENV="${1:-outputs/logs/feniks_sc_drws_topology_npe_pilot_latest.env}"
source "$SOURCE_ENV"
REPO_DIR="$(pwd -P)"
SOURCE_PILOT_ROOT="${PILOT_ROOT:?source pilot environment lacks PILOT_ROOT}"
export BALANCED_ROOT="${BALANCED_ROOT:-$(dirname "$SOURCE_PILOT_ROOT")/frozen_parent_balanced_npe_v1}"
export CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?}/feniks_sc_drws_runtime}"
export MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?}/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-shine}"
export BALANCED_LOG_ROOT="$CACHE_ROOT/slurm_logs/$(basename "$BALANCED_ROOT")"
export BALANCED_ENV="$REPO_DIR/outputs/logs/feniks_sc_drws_balanced_npe_latest.env"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
if [[ "${RESUME_SUBMISSION:-0}" != 1 ]]; then
  JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu \
    python scripts/run_feniks_sc_drws_balanced_npe.py prepare \
    --root "$BALANCED_ROOT" --source-root "$SOURCE_PILOT_ROOT"
fi
SNAPSHOT="$CACHE_ROOT/code/balanced-npe-${COMMIT:0:12}"
mkdir -p "$(dirname "$SNAPSHOT")" "$BALANCED_LOG_ROOT" outputs/logs
if [[ ! -e "$SNAPSHOT" ]]; then
  git worktree add --detach "$SNAPSHOT" "$COMMIT"
fi
test "$(git -C "$SNAPSHOT" rev-parse HEAD)" = "$COMMIT"
git -C "$SNAPSHOT" diff --quiet
git -C "$SNAPSHOT" diff --cached --quiet
if [[ ! -e "$SNAPSHOT/Data/diffsky" ]]; then
  mkdir -p "$SNAPSHOT/Data"
  ln -s "$REPO_DIR/Data/diffsky" "$SNAPSHOT/Data/diffsky"
fi
export REPO_DIR="$SNAPSHOT"
python scripts/run_feniks_sc_drws_balanced_npe.py submit --root "$BALANCED_ROOT"
echo "monitor: bash scripts/monitor_feniks_sc_drws_balanced_npe.sh $BALANCED_ENV"
