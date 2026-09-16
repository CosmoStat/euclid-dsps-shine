#!/bin/bash
#SBATCH --job-name=feniks_sc_drws_infer_gate
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --hint=nomultithread
set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
RECOVERY_ROOT="${RECOVERY_ROOT:?RECOVERY_ROOT is required}"
INFERENCE_ROOT="${INFERENCE_ROOT:?INFERENCE_ROOT is required}"
ALLOW_DIAGNOSTIC_FULL="${ALLOW_DIAGNOSTIC_FULL:-0}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" JAX_PLATFORMS=cpu
cd "$REPO_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
RECEIPT="$RECOVERY_ROOT/FULL_TRAIN_PASS.json"
ARGS=()
if [[ ! -s "$RECEIPT" && "$ALLOW_DIAGNOSTIC_FULL" == "1" ]]; then
  RECEIPT="$RECOVERY_ROOT/FULL_TRAIN_FAIL.json"
  ARGS+=(--diagnostic-only)
fi
python scripts/finalize_feniks_sc_drws_inference.py --root "$INFERENCE_ROOT" \
  --full-receipt "$RECEIPT" "${ARGS[@]}"
