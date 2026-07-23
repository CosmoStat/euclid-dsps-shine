#!/bin/bash
#SBATCH --job-name=feniks_pfinal
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3%4
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

CONFIGS=(
  configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml
  configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed3.yaml
  configs/experiments/feniks_selfsup_paper_fixed_prior_rws_k8_t2.yaml
  configs/experiments/feniks_selfsup_paper_avi_joint_t2.yaml
)
LABELS=(rws_k8_t2_seed2 rws_k8_t2_seed3 fixed_prior_rws_k8_t2 avi_joint_t2)
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CONFIG="${CONFIGS[$TASK_ID]}"
LABEL="${LABELS[$TASK_ID]}"
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
LENS_DIR="$ROOT_DIR/$LABEL/jacobian_lens"

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
  amortized-finalize-jacobian-lens --out "$LENS_DIR"
for artifact in jacobian_lens_summary.json singular_values.parquet \
  singular_spectrum_median.png effective_rank_hist.png ae_noise_amplification_hist.png; do
  test -s "$LENS_DIR/$artifact" || {
    echo "[paper-finalize][error] missing Lens artifact: $LENS_DIR/$artifact"; exit 2;
  }
done
touch "$ROOT_DIR/$LABEL/JLENS_DONE"

exec 9>"$ROOT_DIR/.aggregate.lock"
flock -x 9
if [[ $(find "$ROOT_DIR" -mindepth 2 -maxdepth 2 -name JLENS_DONE | wc -l) -eq 4 ]]; then
  python scripts/summarize_feniks_selfsup_paper.py \
    --aggregate-root "$ROOT_DIR" --expected 4
  for artifact in README.md experiment_metrics.csv coverage_by_parameter.png \
    scientific_scoreboard.png jacobian_spectrum_comparison.png \
    photoz_by_redshift.png prior_marginal_error_by_parameter.png; do
    test -s "$ROOT_DIR/comparison/$artifact" || {
      echo "[paper-finalize][error] missing comparison artifact: $artifact"; exit 2;
    }
  done
fi
