#!/bin/bash
#SBATCH --job-name=feniks_sc_ppc
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="${CONFIG:-configs/experiments/feniks_sc_asmc_em_r25.yaml}"
TRUTH_CONFIG="${TRUTH_CONFIG:-configs/experiments/feniks_sc_asmc_em_r25_truth_closure.yaml}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the frozen SC-ASMC-EM run}"
CLOSURE_ROOT="${CLOSURE_ROOT:?Set CLOSURE_ROOT to the completed truth closure}"
AUDIT_ROOT="${AUDIT_ROOT:?Set a new empty AUDIT_ROOT}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/sc_asmc_predictive_cache}"

export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.82}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${USER}-${SLURM_JOB_ID}"
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="${CACHE_ROOT}/jax"
export JAX_COMPILATION_CACHE_DIR="${CACHE_ROOT}/jax"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"

cd "$REPO_DIR"
mkdir -p "$CACHE_ROOT/jax" "$MPLCONFIGDIR"
test -s "$RUN_ROOT/FINAL_PASS"
test -s "$CLOSURE_ROOT/truth_closure_receipt.json"
test ! -e "$AUDIT_ROOT"

module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python scripts/run_feniks_sc_asmc_predictive_audit.py \
  --config "$CONFIG" \
  --truth-config "$TRUTH_CONFIG" \
  --run-root "$RUN_ROOT" \
  --closure-root "$CLOSURE_ROOT" \
  --out "$AUDIT_ROOT" \
  --posterior-draws 16 \
  --decoder-pairs-per-batch 128

test -s "$AUDIT_ROOT/predictive_audit_receipt.json"
