#!/bin/bash
#SBATCH --job-name=feniks_norm_flow
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail

if [[ -z "${REPO_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/euclid_dsps" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
  elif [[ -n "${WORK:-}" ]]; then
    REPO_DIR="$WORK/euclid-dsps-shine"
  else
    REPO_DIR="$PWD"
  fi
fi
if [[ -z "${MINICONDA_PATH:-}" ]]; then
  MINICONDA_PATH="${WORK:-$(dirname "$REPO_DIR")}/miniconda3"
fi

CONDA_ENV="${CONDA_ENV:-shine}"
STAGE="${STAGE:-train}"
SMOKE="${SMOKE:-0}"
RUNS_ROOT="${RUNS_ROOT:-outputs/runs}"
COMPARISON_OUT="${COMPARISON_OUT:-${RUNS_ROOT}/feniks_normalization_flow_comparison}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -z "${RUN_SUFFIX+x}" ]]; then
  if [[ "$SMOKE" == "1" ]]; then RUN_SUFFIX="_smoke"; else RUN_SUFFIX=""; fi
fi

CONFIGS=(
  configs/experiments/feniks_normalization_hybrid_realnvp_18d_h100.yaml
  configs/experiments/feniks_normalization_hybrid_rqspline_18d_h100.yaml
  configs/experiments/feniks_normalization_dirac_realnvp_18d_h100.yaml
  configs/experiments/feniks_normalization_dirac_rqspline_18d_h100.yaml
)
RUN_NAMES=(
  feniks_norm_hybrid_realnvp
  feniks_norm_hybrid_rqspline
  feniks_norm_dirac_realnvp
  feniks_norm_dirac_rqspline
)

if (( TASK_ID < 0 || TASK_ID >= ${#CONFIGS[@]} )); then
  echo "[job][error] array index must be in 0..3, got ${TASK_ID}"
  exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"

cd "$REPO_DIR"
mkdir -p outputs/logs "$RUNS_ROOT"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[job] start=$(date --iso-8601=seconds) stage=$STAGE smoke=$SMOKE task=$TASK_ID"
echo "[job] repo=$REPO_DIR commit=$(git rev-parse HEAD)"
python - <<'PY'
import jax
print("[job] backend:", jax.default_backend())
print("[job] devices:", jax.devices())
PY

if [[ "$STAGE" == "compare" ]]; then
  compare_runs=()
  for run_name in "${RUN_NAMES[@]}"; do compare_runs+=("${run_name}${RUN_SUFFIX}"); done
  python scripts/compare_prior_normalization_benchmarks.py \
    --runs-root "$RUNS_ROOT" \
    --out "${COMPARISON_OUT}${RUN_SUFFIX}" \
    --runs "${compare_runs[@]}"
  exit 0
fi
if [[ "$STAGE" != "train" ]]; then
  echo "[job][error] STAGE must be train or compare"
  exit 2
fi

CONFIG="${CONFIG:-${CONFIGS[$TASK_ID]}}"
RUN_NAME="${RUN_NAME:-${RUN_NAMES[$TASK_ID]}${RUN_SUFFIX}}"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/${RUN_NAME}}"
if [[ -e "$OUT_DIR" ]]; then
  echo "[job][error] output exists: $OUT_DIR (choose a new RUN_SUFFIX or RUNS_ROOT)"
  exit 2
fi

cmd=(
  python scripts/run_prior_normalization_benchmark.py
  --config "$CONFIG"
  --out "$OUT_DIR"
  --data-parallel "${DATA_PARALLEL:-auto}"
)
if [[ "$SMOKE" == "1" ]]; then
  cmd+=(
    --limit "${LIMIT:-2000}"
    --epochs "${EPOCHS:-3}"
    --batch-size "${BATCH_SIZE:-512}"
    --n-layers "${N_LAYERS:-2}"
    --hidden-size "${HIDDEN_SIZE:-32}"
    --prior-samples "${PRIOR_SAMPLES:-1000}"
  )
else
  [[ -n "${LIMIT:-}" ]] && cmd+=(--limit "$LIMIT")
  [[ -n "${EPOCHS:-}" ]] && cmd+=(--epochs "$EPOCHS")
  [[ -n "${BATCH_SIZE:-}" ]] && cmd+=(--batch-size "$BATCH_SIZE")
  [[ -n "${PRIOR_SAMPLES:-}" ]] && cmd+=(--prior-samples "$PRIOR_SAMPLES")
fi

printf '[job] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
echo "[job] done=$(date --iso-8601=seconds)"
