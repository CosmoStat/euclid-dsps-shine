#!/bin/bash
#SBATCH --job-name=feniks_sc_smoke
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=03:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="${CONFIG:-configs/experiments/feniks_sc_asmc_em_r25.yaml}"
CATALOG="${CATALOG:?Set CATALOG to the existing FENIKS parquet}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to a new smoke directory}"
SMOKE_ROW_OFFSET="${SMOKE_ROW_OFFSET:-0}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"

export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.88
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${USER}-${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH:?Set SCRATCH}/$USER/jax-cache/euclid-dsps/sc-asmc-em"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"

cd "$REPO_DIR"
mkdir -p outputs/logs "$RUN_ROOT" "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

COMMON=(
  python scripts/run_feniks_sc_asmc_em.py
  --config "$CONFIG" --catalog "$CATALOG" --out "$RUN_ROOT"
  --require-gpu --expected-devices 4
)
if [[ -n "${SMOKE4_ROOT:-}" ]]; then
  python scripts/validate_feniks_sc_asmc_smoke.py \
    --smoke-root "$SMOKE4_ROOT" --config "$CONFIG" --catalog "$CATALOG"
fi
"${COMMON[@]}" prepare --estep-shards 1
"${COMMON[@]}" sleep --smoke
"${COMMON[@]}" smoke --objects 8 --row-offset "$SMOKE_ROW_OFFSET"
test -s "$RUN_ROOT/smoke/SMOKE_PASS.json"
test -s "$RUN_ROOT/smoke/SMOKE_PASS"
