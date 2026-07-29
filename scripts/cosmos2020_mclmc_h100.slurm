#!/bin/bash
#SBATCH --job-name=cosmos_mclmc
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-5%2
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

CONFIG="${CONFIG:-configs/experiments/popcosmos_a24_rws_joint.yaml}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
DATASET="${DATASET:-Data/cosmos2020/prepared/farmer_a24_full.parquet}"
MODEL_STAGE="${MODEL_STAGE:-n40k}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/popcosmos_a24_rws_v2/$MODEL_STAGE}"
OUT="${OUT:-outputs/runs/popcosmos_a24_mclmc_v2}"
MODE="${MODE:-pilot}"
INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${INDEX}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

test -s "$OUT/cohort.parquet"
test -s "$MODEL_ROOT/train/checkpoints/best.eqx"
COMMON=(
  --config "$CONFIG" --dataset "$DATASET"
  --checkpoint "$MODEL_ROOT/train/checkpoints/best.eqx"
  --feature-stats "$MODEL_ROOT/train/feature_stats.json"
  --out "$OUT" --galaxy-index "$INDEX" --mode "$MODE"
)
python scripts/run_feniks_exact_posterior_benchmark.py prepare-galaxy \
  "${COMMON[@]}" --encoder-samples 2048 --map-starts 8 --map-iterations 500

NCHAINS=4
if [[ "$MODE" == "smoke" ]]; then NCHAINS=2; fi
for ((chain=0; chain<NCHAINS; chain++)); do
  python scripts/run_feniks_exact_posterior_benchmark.py sample-chain \
    "${COMMON[@]}" --sampler mclmc --sampler-label mclmc \
    --chain-index "$chain" --mclmc-tune 250 --sample-chunks 100,500 \
    --thinning 4
done
python scripts/run_feniks_exact_posterior_benchmark.py finalize-mclmc "${COMMON[@]}"
