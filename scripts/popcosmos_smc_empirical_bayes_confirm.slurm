#!/bin/bash
#SBATCH --job-name=cosmos_em_confirm
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:?Set CANDIDATE_CHECKPOINT}"
SMC_ROOT="${SMC_ROOT:?Set SMC_ROOT}"
OUT="${OUT:?Set OUT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

CONFIG=configs/experiments/popcosmos_native15d_rws_floor05.yaml
BANK_A="$SMC_ROOT/floor_0p05/seed_260817"
BANK_B="$SMC_ROOT/floor_0p05/seed_260818"
for path in "$CONFIG" "$SOURCE_CHECKPOINT" "$CANDIDATE_CHECKPOINT" "$BANK_A/DONE" "$BANK_B/DONE"; do
  test -e "$path"
done

python scripts/confirm_popcosmos_smc_empirical_bayes.py \
  --config "$CONFIG" \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --candidate-checkpoint "$CANDIDATE_CHECKPOINT" \
  --bank "$BANK_A" \
  --bank "$BANK_B" \
  --out "$OUT"
