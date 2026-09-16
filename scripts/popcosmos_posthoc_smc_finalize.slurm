#!/bin/bash
#SBATCH --job-name=cosmos_smc_gate
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
N_SHARDS="${N_SHARDS:-4}"
LIMIT="${LIMIT:-256}"
SMC_VARIANTS_CSV="${SMC_VARIANTS_CSV:-floor_0p00,floor_0p02,floor_0p05}"
SMC_SEEDS_CSV="${SMC_SEEDS_CSV:-260817,260818}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$REPO_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
IFS=',' read -r -a SMC_VARIANTS <<< "$SMC_VARIANTS_CSV"
IFS=',' read -r -a SMC_SEEDS <<< "$SMC_SEEDS_CSV"
for variant in "${SMC_VARIANTS[@]}"; do
  for seed in "${SMC_SEEDS[@]}"; do
    python scripts/combine_popcosmos_posthoc_smc_shards.py \
      --root "$OUTPUT_ROOT/$variant/seed_$seed" \
      --expected-shards "$N_SHARDS" \
      --expected-objects "$LIMIT"
  done
done
python scripts/summarize_popcosmos_posthoc_smc.py \
  --root "$OUTPUT_ROOT" \
  --variants "$SMC_VARIANTS_CSV"
