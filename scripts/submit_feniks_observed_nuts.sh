#!/bin/bash
set -Eeuo pipefail

export EXPERIMENT_ROOT="$(realpath -m "${1:?new observed experiment root}")"
REFERENCE="$(realpath "${2:?completed transport64 objective pilot root}")"
export CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?}/feniks_sc_drws_runtime}"

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
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m scripts.feniks_geometry_nuts \
  prepare-observed --root "$EXPERIMENT_ROOT" --reference "$REFERENCE"
printf '%s\n' "$REPO_DIR" > "$EXPERIMENT_ROOT/CODE_DIR"
mkdir -p "$EXPERIMENT_ROOT/logs"

export EXPERIMENT_MODE=geometry
GEOMETRY_JOB=$(sbatch --parsable --job-name=feniks_obs_geometry --time=03:00:00 \
  --export=ALL --output="$EXPERIMENT_ROOT/logs/geometry-%j.out" \
  --error="$EXPERIMENT_ROOT/logs/geometry-%j.err" \
  "$REPO_DIR/scripts/feniks_geometry_nuts.slurm")

export EXPERIMENT_MODE=nuts
NUTS_ARRAY_CONCURRENCY="${NUTS_ARRAY_CONCURRENCY:-8}"
if [[ ! "$NUTS_ARRAY_CONCURRENCY" =~ ^[0-9]+$ ]] || \
  (( NUTS_ARRAY_CONCURRENCY < 1 || NUTS_ARRAY_CONCURRENCY > 8 )); then
  echo 'NUTS_ARRAY_CONCURRENCY must be an integer from 1 through 8.' >&2
  exit 2
fi
NUTS_JOB=$(sbatch --parsable --job-name=feniks_obs_nuts --array="0-7%${NUTS_ARRAY_CONCURRENCY}" \
  --time=12:00:00 --dependency="afterok:${GEOMETRY_JOB}" --kill-on-invalid-dep=yes \
  --export=ALL --output="$EXPERIMENT_ROOT/logs/nuts-%A_%a.out" \
  --error="$EXPERIMENT_ROOT/logs/nuts-%A_%a.err" \
  "$REPO_DIR/scripts/feniks_geometry_nuts.slurm")

printf '%s\n' "$GEOMETRY_JOB" > "$EXPERIMENT_ROOT/geometry_job.txt"
printf '%s\n' "$NUTS_JOB" > "$EXPERIMENT_ROOT/nuts_job.txt"
printf 'geometry_job=%s\nnuts_job=%s\nroot=%s\n' \
  "$GEOMETRY_JOB" "$NUTS_JOB" "$EXPERIMENT_ROOT"
printf 'peak_h100=%s\nchains_per_h100=8\n' "$NUTS_ARRAY_CONCURRENCY"
