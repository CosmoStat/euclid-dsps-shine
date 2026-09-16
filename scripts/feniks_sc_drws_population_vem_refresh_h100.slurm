#!/bin/bash
#SBATCH --job-name=feniks_sc_vem_refresh
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
VEM_ROOT="${VEM_ROOT:?VEM_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
CONFIG="configs/experiments/feniks_sc_drws_r29_population_vem_refresh.yaml"
FINAL="$VEM_ROOT/q_refresh"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1 EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"

cd "$REPO_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$CACHE_ROOT/jax" "$MPLCONFIGDIR"

python - "$VEM_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from euclid_dsps.amortized.population_vem import require_git_commit

manifest = json.loads((Path(sys.argv[1]) / "RUN_MANIFEST.json").read_text())
require_git_commit(Path.cwd(), manifest["code_commit"])
PY

if [[ -s "$FINAL/Q_REFRESH_COMPLETE.json" ]]; then
  python scripts/finalize_feniks_sc_drws_population_refresh.py \
    --root "$VEM_ROOT" --refresh "$FINAL"
  exit 0
fi
if [[ -d "$FINAL" ]]; then
  python scripts/finalize_feniks_sc_drws_population_refresh.py \
    --root "$VEM_ROOT" --refresh "$FINAL"
  exit 0
fi

readarray -t VALUES < <(python - "$VEM_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
manifest=json.load(open(root/'RUN_MANIFEST.json'))
prior=json.load(open(root/'prior'/'PRIOR_MSTEP_COMPLETE.json'))
if prior.get('status') != 'COMPLETE':
    raise SystemExit('prior M-step is not complete')
print(manifest['datasets']['train']['path'])
print(manifest['banks']['q_fit']['cohort_path'])
print(manifest['banks']['q_validation']['cohort_path'])
print(manifest['frozen_source']['feature_stats'])
print(prior['checkpoint'])
PY
)
TRAIN_CATALOG="${VALUES[0]}"
TRAIN_INDICES="${VALUES[1]}"
VALIDATION_INDICES="${VALUES[2]}"
FEATURE_STATS="${VALUES[3]}"
PRIOR_CHECKPOINT="${VALUES[4]}"
ATTEMPT="$VEM_ROOT/.q_refresh-${SLURM_JOB_ID}"
rm -rf "$ATTEMPT"

python -m euclid_dsps.cli --config "$CONFIG" amortized-train-diffsky \
  --dataset "$TRAIN_CATALOG" --out "$ATTEMPT" \
  --train-indices-file "$TRAIN_INDICES" \
  --validation-indices-file "$VALIDATION_INDICES" \
  --initial-checkpoint "$PRIOR_CHECKPOINT" \
  --fixed-feature-stats "$FEATURE_STATS" \
  --epochs 2 --batch-size 1024 --jax-batch-size 256 --n-samples 2 \
  --kl-annealing-epochs 0 --kl-weight-max 1.0 --validation-every 1 \
  --best-checkpoint-min-epoch 1 --data-parallel pmap --freeze-prior \
  --seed 266500 --no-progress

mv "$ATTEMPT" "$FINAL"
python scripts/finalize_feniks_sc_drws_population_refresh.py \
  --root "$VEM_ROOT" --refresh "$FINAL"
