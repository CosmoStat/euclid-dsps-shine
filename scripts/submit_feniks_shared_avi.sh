#!/bin/bash
set -Eeuo pipefail
export REPO_DIR="$(pwd -P)"
export CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?}/feniks_sc_drws_runtime}"
SOURCE="${1:?precision night root}"
export AVI_ROOT="${2:?new output root}"
git diff --quiet
git diff --cached --quiet
python -m scripts.feniks_shared_avi prepare --source "$SOURCE" --root "$AVI_ROOT"
mkdir -p "$CACHE_ROOT/slurm_logs" "$CACHE_ROOT/code"
COMMIT="$(git rev-parse HEAD)"
SNAPSHOT="$CACHE_ROOT/code/shared-avi-$COMMIT"
if [[ ! -e "$SNAPSHOT" ]]; then
  git worktree add --detach "$SNAPSHOT" "$COMMIT"
fi
test "$(git -C "$SNAPSHOT" rev-parse HEAD)" = "$COMMIT"
if [[ ! -e "$SNAPSHOT/Data/diffsky" ]]; then
  mkdir -p "$SNAPSHOT/Data"
  ln -s "$REPO_DIR/Data/diffsky" "$SNAPSHOT/Data/diffsky"
fi
export REPO_DIR="$SNAPSHOT"
sbatch --parsable --export=ALL \
  --output="$CACHE_ROOT/slurm_logs/shared-avi-%A_%a.out" \
  --error="$CACHE_ROOT/slurm_logs/shared-avi-%A_%a.err" \
  "$REPO_DIR/scripts/feniks_shared_avi.slurm"
