#!/bin/bash
#SBATCH --job-name=feniks_shared_avi
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --array=0-1%2
set -Eeuo pipefail
cd "${REPO_DIR:?}"
module purge
module load arch/h100
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export MPLCONFIGDIR="$CACHE_ROOT/mpl/$SLURM_JOB_ID"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
python -m scripts.feniks_shared_avi run --root "$AVI_ROOT" --task "$SLURM_ARRAY_TASK_ID"
