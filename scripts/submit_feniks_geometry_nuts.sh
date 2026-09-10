#!/bin/bash
set -Eeuo pipefail
MODE="${1:?geometry or nuts}"
export EXPERIMENT_ROOT="$(realpath -m "${2:?experiment root}")"
export CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?}/feniks_sc_drws_runtime}"
if [[ "$MODE" == geometry || "$MODE" == nuts-new ]]; then
  REFERENCE="$(realpath "${3:?completed transport64 objective pilot root}")"
  git diff --quiet
  git diff --cached --quiet
  COMMIT="$(git rev-parse HEAD)"
  REPO="$(pwd -P)"
  export REPO_DIR="$CACHE_ROOT/code/geometry-nuts-$COMMIT"
  mkdir -p "$CACHE_ROOT/code"
  if [[ ! -e "$REPO_DIR" ]]; then
    git worktree add --detach "$REPO_DIR" "$COMMIT"
    mkdir -p "$REPO_DIR/Data"
    ln -s "$REPO/Data/diffsky" "$REPO_DIR/Data/diffsky"
  fi
  cd "$REPO_DIR"
  PREPARE=prepare
  if [[ "$MODE" == nuts-new ]]; then PREPARE=prepare-nuts; fi
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m scripts.feniks_geometry_nuts "$PREPARE" \
    --root "$EXPERIMENT_ROOT" --reference "$REFERENCE"
  printf '%s\n' "$REPO_DIR" > "$EXPERIMENT_ROOT/CODE_DIR"
  mkdir -p "$EXPERIMENT_ROOT/logs"
fi
if [[ "$MODE" == geometry ]]; then
  export EXPERIMENT_MODE=geometry
  JOB=$(sbatch --parsable --export=ALL --output="$EXPERIMENT_ROOT/logs/geometry-%j.out" \
    --error="$EXPERIMENT_ROOT/logs/geometry-%j.err" "$REPO_DIR/scripts/feniks_geometry_nuts.slurm")
elif [[ "$MODE" == nuts || "$MODE" == nuts-new ]]; then
  if [[ ! -s "$EXPERIMENT_ROOT/GEOMETRY_COMPLETE.json" ]]; then
    echo 'Geometry is not complete. No NUTS job submitted.' >&2
    exit 1
  fi
  export REPO_DIR="$(<"$EXPERIMENT_ROOT/CODE_DIR")"
  python - "$EXPERIMENT_ROOT/MANIFEST.json" <<'PY'
import json, sys
if json.load(open(sys.argv[1])).get('nuts_target_dtype') != 'float64':
    raise SystemExit('Old NUTS code contract: use nuts-new NEW_ROOT COMPLETED_GEOMETRY_ROOT.')
PY
  export EXPERIMENT_MODE=nuts
  # Explicit second command is the user's review/approval; no afterok auto-launch.
  JOB=$(sbatch --parsable --job-name=feniks_nuts --array=0-11%4 --time=20:00:00 --export=ALL \
    --output="$EXPERIMENT_ROOT/logs/nuts-%A_%a.out" --error="$EXPERIMENT_ROOT/logs/nuts-%A_%a.err" \
    "$REPO_DIR/scripts/feniks_geometry_nuts.slurm")
  MODE=nuts
else
  echo 'mode must be geometry or nuts' >&2
  exit 2
fi
printf '%s\n' "$JOB" > "$EXPERIMENT_ROOT/${MODE}_job.txt"
printf 'job=%s\nroot=%s\n' "$JOB" "$EXPERIMENT_ROOT"
