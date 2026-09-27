#!/bin/bash
set -Eeuo pipefail
cd "${FENIKS_CRA_CODE:?}"
source "${MINICONDA_PATH:-${WORK:?}/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-shine}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
python -m scripts.audit_feniks_coherent_reference \
  --source "${FENIKS_CRA_SOURCE:?}" --root "${FENIKS_CRA_ROOT:?}"
