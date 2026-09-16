#!/bin/bash
#SBATCH --job-name=feniks_sc_repair
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=05:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="${CONFIG:-configs/experiments/feniks_sc_asmc_em_r25.yaml}"
CATALOG="${CATALOG:?Set CATALOG to the existing FENIKS parquet}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the completed training root}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/sc_asmc_repair_cache}"

export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.72}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${USER}-${SLURM_JOB_ID}"
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="${CACHE_ROOT}/jax"
export JAX_COMPILATION_CACHE_DIR="${CACHE_ROOT}/jax"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"

cd "$REPO_DIR"
mkdir -p "$CACHE_ROOT/jax" "$MPLCONFIGDIR"
test -s "$RUN_ROOT/TRAINING_COMPLETE.json"
test -s "$RUN_ROOT/banks/em2_p2/posterior_bank_manifest.json"

module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

COMMON=(
  python scripts/run_feniks_sc_asmc_em.py
  --config "$CONFIG" --catalog "$CATALOG" --out "$RUN_ROOT"
  --require-gpu --expected-devices 4
)
"${COMMON[@]}" repair-final --shard-id 0 --shard-count 1
"${COMMON[@]}" merge-repair-final --shard-count 1

if [[ -s "$RUN_ROOT/report/report_receipt.json" ]]; then
  ARCHIVE="$RUN_ROOT/report_before_final_repair"
  if [[ ! -e "$ARCHIVE" ]]; then
    mv "$RUN_ROOT/report" "$ARCHIVE"
  fi
fi

"${COMMON[@]}" report
"${COMMON[@]}" validate
"${COMMON[@]}" status
test -s "$RUN_ROOT/FINAL_RECEIPT.json"
test -s "$RUN_ROOT/FINAL_PASS"
