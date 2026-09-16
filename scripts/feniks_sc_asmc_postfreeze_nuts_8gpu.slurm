#!/bin/bash
#SBATCH --job-name=feniks_sc_nuts
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --array=0-7%8
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="${CONFIG:-configs/experiments/feniks_sc_asmc_em_r25.yaml}"
CATALOG="${CATALOG:?Set CATALOG to the exact frozen-run FENIKS parquet}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the completed SC-ASMC-EM run}"
NUTS_OUT="${NUTS_OUT:?Set NUTS_OUT to a SCRATCH or WORK validation directory}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
JOB_TOKEN="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
BARRIER_ROOT="$NUTS_OUT/.barriers/$JOB_TOKEN"
FAILURE_MARKER="$BARRIER_ROOT/FAILED"
WAIT_SECONDS="${WAIT_SECONDS:-43200}"

export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.88
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${USER}-${JOB_TOKEN}-${TASK_ID}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${SCRATCH:?Set SCRATCH}/$USER/jax-cache/euclid-dsps/sc-asmc-postfreeze-nuts}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

cd "$REPO_DIR"
mkdir -p outputs/logs "$NUTS_OUT" "$BARRIER_ROOT" "$MPLCONFIGDIR" \
  "$JAX_COMPILATION_CACHE_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

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
      echo "[sc-asmc][postfreeze-nuts] peer failure: $(cat "$FAILURE_MARKER")" >&2
      return 1
    fi
    now=$(date +%s)
    if (( now - started > WAIT_SECONDS )); then
      echo "[sc-asmc][postfreeze-nuts] timed out waiting for $marker" >&2
      return 1
    fi
    sleep 30
  done
}

atomic_marker() {
  local destination="$1"
  printf '%s\n' "$(date -Is)" >"${destination}.tmp.$$"
  mv "${destination}.tmp.$$" "$destination"
}

if (( TASK_ID == 0 )); then
  python - "$RUN_ROOT" "$CATALOG" <<'PY'
import sys
from pathlib import Path

from euclid_dsps.amortized.posterior_bank import sha256_file
from euclid_dsps.amortized.sc_asmc_postfreeze import validate_postfreeze_gate

receipt = validate_postfreeze_gate(Path(sys.argv[1]))
catalog = Path(sys.argv[2]).resolve()
if catalog != Path(receipt["dataset"]["path"]).resolve():
    raise SystemExit("CATALOG path differs from the frozen receipt")
if sha256_file(catalog) != receipt["dataset"]["sha256"]:
    raise SystemExit("CATALOG hash differs from the frozen receipt")
PY
  python scripts/prepare_feniks_sc_asmc_postfreeze_nuts.py \
    --run-root "$RUN_ROOT" --out "$NUTS_OUT" --count 8
  atomic_marker "$BARRIER_ROOT/cohort.done"
else
  wait_for "$BARRIER_ROOT/cohort.done"
fi

CHECKPOINT="$RUN_ROOT/report/frozen/sc_asmc_em_final.eqx"
FEATURE_STATS="$RUN_ROOT/runtime/feature_stats.json"
COMMON=(
  python scripts/run_feniks_exact_posterior_benchmark.py
  --config "$CONFIG" --dataset "$CATALOG" --checkpoint "$CHECKPOINT"
  --feature-stats "$FEATURE_STATS" --out "$NUTS_OUT"
  --galaxy-index "$TASK_ID" --mode smoke --ignore-truth
)
"${COMMON[@]}" prepare-galaxy --encoder-samples 4096 --map-starts 16 \
  --map-iterations 500 --adaptive-smc-mode skip --skip-defensive-is
"${COMMON[@]}" sample-nuts-batched --chain-indices 0,1,2,3 \
  --nuts-warmup "${NUTS_WARMUP:-500}" \
  --nuts-max-doublings "${NUTS_MAX_DOUBLINGS:-7}" \
  --sample-chunks "${NUTS_SAMPLE_CHUNKS:-500,1000}"
"${COMMON[@]}" finalize-galaxy --samplers nuts
atomic_marker "$BARRIER_ROOT/nuts_${TASK_ID}.done"

if (( TASK_ID == 0 )); then
  for index in $(seq 0 7); do
    wait_for "$BARRIER_ROOT/nuts_${index}.done"
  done
  python scripts/run_feniks_exact_posterior_benchmark.py finalize-run \
    --config "$CONFIG" --dataset "$CATALOG" --checkpoint "$CHECKPOINT" \
    --feature-stats "$FEATURE_STATS" --out "$NUTS_OUT" \
    --mode smoke --ignore-truth --samplers nuts
fi
