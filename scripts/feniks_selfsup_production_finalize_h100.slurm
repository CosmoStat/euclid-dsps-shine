#!/bin/bash
#SBATCH --job-name=feniks_t2final
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-2%3
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

CONFIGS=(
  configs/experiments/feniks_selfsup_learned_rws_k8_t2.yaml
  configs/experiments/feniks_selfsup_learned_rws_mix2_k8_t2.yaml
  configs/experiments/feniks_selfsup_learned_smcwake_mix2_k4_t2.yaml
)
LABELS=(selfsup_rws_k8_t2 selfsup_rws_mix2_k8_t2 selfsup_smcwake_mix2_k4_t2)
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CONFIG="${CONFIGS[$TASK_ID]}"
LABEL="${LABELS[$TASK_ID]}"
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-jacobian-lens --out "$ROOT_DIR/$LABEL/jacobian_lens"
touch "$ROOT_DIR/$LABEL/JLENS_DONE"

exec 9>"$ROOT_DIR/.aggregate.lock"
flock -x 9
if [[ $(find "$ROOT_DIR" -mindepth 2 -maxdepth 2 -name JLENS_DONE | wc -l) -eq 3 ]]; then
  python scripts/summarize_feniks_selfsup_production_task.py \
    --aggregate-root "$ROOT_DIR" --expected 3
fi
