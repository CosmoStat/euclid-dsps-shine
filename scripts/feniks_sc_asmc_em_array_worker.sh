#!/bin/bash
# Shared phase orchestrator for the 4/8/16-H100 launchers.

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="${CONFIG:-configs/experiments/feniks_sc_asmc_em_r25.yaml}"
CATALOG="${CATALOG:?Set CATALOG to the existing FENIKS parquet}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to an immutable SCRATCH or WORK directory}"
SMOKE_ROOT="${SMOKE_ROOT:?Set SMOKE_ROOT to the completed immutable 4-H100 smoke}"
SHARD_COUNT="${SHARD_COUNT:?SHARD_COUNT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
WAIT_SECONDS="${WAIT_SECONDS:-172800}"
JOB_TOKEN="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
BARRIER_ROOT="$RUN_ROOT/.barriers/$JOB_TOKEN"
FAILURE_MARKER="$BARRIER_ROOT/FAILED"

if (( TASK_ID < 0 || TASK_ID >= SHARD_COUNT )); then
  echo "[sc-asmc][error] task $TASK_ID outside shard count $SHARD_COUNT" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.88}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${USER}-${JOB_TOKEN}-${TASK_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${SCRATCH:?Set SCRATCH}/$USER/jax-cache/euclid-dsps/sc-asmc-em}"

cd "$REPO_DIR"
mkdir -p outputs/logs "$RUN_ROOT" "$BARRIER_ROOT" "$MPLCONFIGDIR" \
  "$JAX_COMPILATION_CACHE_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

atomic_marker() {
  local destination="$1"
  local temporary="${destination}.tmp.$$"
  printf '%s\n' "$(date -Is)" >"$temporary"
  mv "$temporary" "$destination"
}

on_exit() {
  local status=$?
  if (( status != 0 )); then
    printf 'task=%s status=%s time=%s\n' "$TASK_ID" "$status" "$(date -Is)" \
      >"${FAILURE_MARKER}.tmp.${TASK_ID}.$$"
    mv "${FAILURE_MARKER}.tmp.${TASK_ID}.$$" "$FAILURE_MARKER"
  fi
  exit "$status"
}
trap on_exit EXIT

wait_for() {
  local marker="$1"
  local started now
  started=$(date +%s)
  while [[ ! -s "$marker" ]]; do
    if [[ -s "$FAILURE_MARKER" ]]; then
      echo "[sc-asmc][error] peer failure: $(cat "$FAILURE_MARKER")" >&2
      return 1
    fi
    now=$(date +%s)
    if (( now - started > WAIT_SECONDS )); then
      echo "[sc-asmc][error] timed out waiting for $marker" >&2
      return 1
    fi
    sleep 30
  done
}

wait_for_tasks() {
  local phase="$1"
  local index
  for ((index = 0; index < SHARD_COUNT; index++)); do
    wait_for "$BARRIER_ROOT/${phase}_${index}.done"
  done
}

run_stage() {
  python scripts/run_feniks_sc_asmc_em.py \
    --config "$CONFIG" \
    --catalog "$CATALOG" \
    --out "$RUN_ROOT" \
    --require-gpu \
    --expected-devices 4 \
    "$@"
}

test -s "$CONFIG"
test -s "$CATALOG"
python - <<'PY'
import jax

print("[sc-asmc] backend:", jax.default_backend())
print("[sc-asmc] local devices:", jax.local_devices())
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 4:
    raise SystemExit("expected exactly four local H100 devices")
PY
nvidia-smi

python scripts/validate_feniks_sc_asmc_smoke.py \
  --smoke-root "$SMOKE_ROOT" --config "$CONFIG" --catalog "$CATALOG"
if [[ "${REQUIRE_SMOKE16:-0}" == "1" ]]; then
  SMOKE16_ROOT="${SMOKE16_ROOT:?Set SMOKE16_ROOT to the completed 16-H100 smoke}"
  for smoke_shard in 0 1 2 3; do
    python scripts/validate_feniks_sc_asmc_smoke.py \
      --smoke-root "$SMOKE16_ROOT/shard_$smoke_shard" \
      --config "$CONFIG" --catalog "$CATALOG"
  done
fi

if (( TASK_ID == 0 )); then
  run_stage prepare --estep-shards "$SHARD_COUNT"
  run_stage sleep
  run_stage preflight --parallel-shards "$SHARD_COUNT"
  atomic_marker "$BARRIER_ROOT/bootstrap.done"
else
  wait_for "$BARRIER_ROOT/bootstrap.done"
fi

run_stage estep --iteration 1 --shard-id "$TASK_ID" --shard-count "$SHARD_COUNT"
atomic_marker "$BARRIER_ROOT/estep1_${TASK_ID}.done"
if (( TASK_ID == 0 )); then
  wait_for_tasks estep1
  run_stage merge-estep --iteration 1 --shard-count "$SHARD_COUNT"
  run_stage prior-mstep --iteration 1
  atomic_marker "$BARRIER_ROOT/mstep1.done"
else
  wait_for "$BARRIER_ROOT/mstep1.done"
fi

run_stage reweight --iteration 1 --shard-id "$TASK_ID" --shard-count "$SHARD_COUNT"
atomic_marker "$BARRIER_ROOT/reweight1_${TASK_ID}.done"
if (( TASK_ID == 0 )); then
  wait_for_tasks reweight1
  run_stage merge-reweight --iteration 1 --shard-count "$SHARD_COUNT"
  run_stage q-distill --iteration 1
  atomic_marker "$BARRIER_ROOT/distill1.done"
else
  wait_for "$BARRIER_ROOT/distill1.done"
fi

run_stage estep --iteration 2 --shard-id "$TASK_ID" --shard-count "$SHARD_COUNT"
atomic_marker "$BARRIER_ROOT/estep2_${TASK_ID}.done"
if (( TASK_ID == 0 )); then
  wait_for_tasks estep2
  run_stage merge-estep --iteration 2 --shard-count "$SHARD_COUNT"
  run_stage prior-mstep --iteration 2
  atomic_marker "$BARRIER_ROOT/mstep2.done"
else
  wait_for "$BARRIER_ROOT/mstep2.done"
fi

run_stage reweight --iteration 2 --shard-id "$TASK_ID" --shard-count "$SHARD_COUNT"
atomic_marker "$BARRIER_ROOT/reweight2_${TASK_ID}.done"
if (( TASK_ID == 0 )); then
  wait_for_tasks reweight2
  run_stage merge-reweight --iteration 2 --shard-count "$SHARD_COUNT"
  run_stage mark-training-complete
  run_stage report
  run_stage validate
  atomic_marker "$BARRIER_ROOT/final.done"
else
  wait_for "$BARRIER_ROOT/final.done"
fi

run_stage status
