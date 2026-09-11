#!/bin/bash
set -Eeuo pipefail
MODE="${1:?geometry, nuts, nuts-new, nuts-probe-new, or nuts-long-new}"
export EXPERIMENT_ROOT="$(realpath -m "${2:?experiment root}")"
export CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?}/feniks_sc_drws_runtime}"
if [[ "$MODE" == geometry || "$MODE" == nuts-new || \
      "$MODE" == nuts-probe-new || "$MODE" == nuts-long-new ]]; then
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
  PROFILE=float64_depth4_parallel_v1
  if [[ "$MODE" == nuts-new || "$MODE" == nuts-probe-new || "$MODE" == nuts-long-new ]]; then
    PREPARE=prepare-nuts
  fi
  if [[ "$MODE" == nuts-probe-new ]]; then
    PROFILE=float64_dense_depth56_probe_v1
  elif [[ "$MODE" == nuts-long-new ]]; then
    PROFILE=float64_dense_depth6_long_v1
  fi
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m scripts.feniks_geometry_nuts "$PREPARE" \
    --root "$EXPERIMENT_ROOT" --reference "$REFERENCE" --profile "$PROFILE"
  printf '%s\n' "$REPO_DIR" > "$EXPERIMENT_ROOT/CODE_DIR"
  mkdir -p "$EXPERIMENT_ROOT/logs"
fi
if [[ "$MODE" == geometry ]]; then
  export EXPERIMENT_MODE=geometry
  JOB=$(sbatch --parsable --export=ALL --output="$EXPERIMENT_ROOT/logs/geometry-%j.out" \
    --error="$EXPERIMENT_ROOT/logs/geometry-%j.err" "$REPO_DIR/scripts/feniks_geometry_nuts.slurm")
elif [[ "$MODE" == nuts || "$MODE" == nuts-new || \
        "$MODE" == nuts-probe-new || "$MODE" == nuts-long-new ]]; then
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
  readarray -t PROFILE_VALUES < <(python - "$EXPERIMENT_ROOT/MANIFEST.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))['nuts_execution_profile']
print(p['array_tasks'])
print(p['recommended_array_concurrency'])
print(p.get('slurm_time', '08:00:00'))
PY
)
  ARRAY_TASKS="${PROFILE_VALUES[0]}"
  RECOMMENDED_CONCURRENCY="${PROFILE_VALUES[1]}"
  NUTS_TIME="${PROFILE_VALUES[2]}"
  NUTS_ARRAY_CONCURRENCY="${NUTS_ARRAY_CONCURRENCY:-$RECOMMENDED_CONCURRENCY}"
  if [[ ! "$NUTS_ARRAY_CONCURRENCY" =~ ^[0-9]+$ ]] || \
    (( NUTS_ARRAY_CONCURRENCY < 1 || NUTS_ARRAY_CONCURRENCY > ARRAY_TASKS )); then
    echo "NUTS_ARRAY_CONCURRENCY must be an integer from 1 through $ARRAY_TASKS." >&2
    exit 2
  fi
  ARRAY_LAST=$((ARRAY_TASKS - 1))
  # Explicit second command is the user's review/approval; no afterok auto-launch.
  JOB=$(sbatch --parsable --job-name=feniks_nuts --array="0-${ARRAY_LAST}%${NUTS_ARRAY_CONCURRENCY}" --time="$NUTS_TIME" --export=ALL \
    --output="$EXPERIMENT_ROOT/logs/nuts-%A_%a.out" --error="$EXPERIMENT_ROOT/logs/nuts-%A_%a.err" \
    "$REPO_DIR/scripts/feniks_geometry_nuts.slurm")
  MODE=nuts
else
  echo 'unsupported mode' >&2
  exit 2
fi
printf '%s\n' "$JOB" > "$EXPERIMENT_ROOT/${MODE}_job.txt"
printf 'job=%s\nroot=%s\n' "$JOB" "$EXPERIMENT_ROOT"
if [[ "$MODE" == nuts ]]; then
  printf 'array_concurrency=%s\n' "$NUTS_ARRAY_CONCURRENCY"
  printf 'array_tasks=%s\ntime_limit=%s\n' "$ARRAY_TASKS" "$NUTS_TIME"
fi
