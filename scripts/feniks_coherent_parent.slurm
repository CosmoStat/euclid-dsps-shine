#!/bin/bash
set -Eeuo pipefail
cd "${FENIKS_PARENT_CODE:?}"
source "${MINICONDA_PATH:-${WORK:?}/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-shine}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export JAX_ENABLE_X64=true MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ "${FENIKS_PARENT_GPU:-0}" == 1 ]]; then
  module purge
  module load arch/h100
  export JAX_PLATFORMS=cuda EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_REQUIRE_GPU=1
  export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
else
  export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu EUCLID_DSPS_REQUIRE_GPU=0
  export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
fi
python -m scripts.feniks_coherent_parent "${FENIKS_PARENT_MODE:?}" \
  --root "${FENIKS_PARENT_ROOT:?}" --task "${SLURM_ARRAY_TASK_ID:-0}"
