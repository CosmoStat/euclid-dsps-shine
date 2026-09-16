#!/bin/bash
#SBATCH --job-name=feniks_xagg
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
FINAL_SAMPLERS="${FINAL_SAMPLERS:-nuts,mclmc}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}"
cd "$REPO_DIR"
mkdir -p "$MPLCONFIGDIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
python scripts/run_feniks_exact_posterior_benchmark.py finalize-run \
  --config "$CONFIG" --dataset "$DATASET" --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" --out "$ROOT_DIR" \
  --samplers "$FINAL_SAMPLERS"
