#!/bin/bash
#SBATCH --job-name=diffsky_full_h100
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

# Budgeted H100 science-validation run.
# The default pipeline now runs forward closure, standard-normal baseline,
# joint-RealNVP target, sharded inference, and reports. Supervised-prior stages
# are kept as explicit dev diagnostics and are not run by this script.
# This job is intentionally offline: it does not run pip/uv/git network
# commands and assumes the repository, conda environment, and Diffsky assets
# are already present on Jean-Zay.
#
# Submit from any directory with:
#
#   sbatch scripts/diffsky_full_validation_h100.slurm
#
# Useful overrides:
#
#   sbatch --export=ALL,RUN_NAME=diffsky_full_h100_v1,EPOCHS=8,BATCH_SIZE=64 \
#     scripts/diffsky_full_validation_h100.slurm
#
#   sbatch --export=ALL,RUN_NAME=diffsky_full_h100_v2,BATCH_SIZE=128,JAX_BATCH_SIZE=128 \
#     scripts/diffsky_full_validation_h100.slurm
#
#   sbatch --export=ALL,LIMIT=2048,EPOCHS=2,POSTERIOR_SAMPLES=8 \
#     scripts/diffsky_full_validation_h100.slurm
#
# Leave LIMIT empty for the real full-dataset run.

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
  if [[ -n "${WORK:-}" && -d "$WORK/miniconda3" ]]; then
    MINICONDA_PATH="$WORK/miniconda3"
  else
    MINICONDA_PATH="$(dirname "$REPO_DIR")/miniconda3"
  fi
fi
CONDA_ENV="${CONDA_ENV:-shine}"
OFFLINE="${OFFLINE:-1}"

RUN_NAME="${RUN_NAME:-diffsky_full_h100_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/runs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_${SLURM_JOB_ID:-manual}.log}"

CONFIG="${CONFIG:-configs/experiments/diffsky_hltds_full_h100.yaml}"
DATASET="${DATASET:-Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet}"
SOURCE_DATASET="${SOURCE_DATASET:-Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet}"
REDSHIFT_MIN="${REDSHIFT_MIN:-0.0}"
REDSHIFT_MAX="${REDSHIFT_MAX:-0.35}"
ERROR_MODEL="${ERROR_MODEL:-m5_depth}"
ERROR_SNR="${ERROR_SNR:-50}"
FORCE_REBUILD_DATASET="${FORCE_REBUILD_DATASET:-0}"
RAW_ROOT="${RAW_ROOT:-Data/diffsky/raw/hltds_cosmos_260215_04_14_2026}"

# These defaults are meant to fit a roughly 10h walltime budget on one H100.
# Increase EPOCHS for a deeper science run after checking the first report.
BATCH_SIZE="${BATCH_SIZE:-64}"
JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-64}"
DECODER_SAMPLE_CHUNK_SIZE="${DECODER_SAMPLE_CHUNK_SIZE:-4}"
PRIOR_PREDICTIVE_BATCH_SIZE="${PRIOR_PREDICTIVE_BATCH_SIZE:-1024}"
EPOCHS="${EPOCHS:-5}"
N_SAMPLES="${N_SAMPLES:-1}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-128}"
PRIOR_SAMPLES="${PRIOR_SAMPLES:-8192}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export REPO_DIR
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "$REPO_DIR"
mkdir -p "$LOG_DIR" "$(dirname "$OUT_DIR")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[job] start: $(date --iso-8601=seconds)"
echo "[job] host: $(hostname)"
echo "[job] repo: $REPO_DIR"
echo "[job] run_name: $RUN_NAME"
echo "[job] out_dir: $OUT_DIR"
echo "[job] log_file: $LOG_FILE"
echo "[job] slurm_job_id: ${SLURM_JOB_ID:-manual}"
echo "[job] slurm_cpus_per_task: ${SLURM_CPUS_PER_TASK:-unset}"
echo "[job] config: $CONFIG"
echo "[job] dataset: $DATASET"
echo "[job] source_dataset: $SOURCE_DATASET"
echo "[job] redshift_range: [$REDSHIFT_MIN, $REDSHIFT_MAX]"
echo "[job] error_model: $ERROR_MODEL snr_if_fractional=$ERROR_SNR"
echo "[job] force_rebuild_dataset: $FORCE_REBUILD_DATASET"
echo "[job] offline: $OFFLINE"
echo "[job] limit: ${LIMIT:-<full dataset>}"
echo "[job] batch_size: $BATCH_SIZE"
echo "[job] jax_batch_size: $JAX_BATCH_SIZE"
echo "[job] decoder_sample_chunk_size: $DECODER_SAMPLE_CHUNK_SIZE"
echo "[job] prior_predictive_batch_size: $PRIOR_PREDICTIVE_BATCH_SIZE"
echo "[job] inference outputs: sharded by config; posterior predictive flux/residual samples disabled by config"
echo "[job] epochs: $EPOCHS"
echo "[job] n_samples: $N_SAMPLES"
echo "[job] posterior_samples: $POSTERIOR_SAMPLES"
echo "[job] prior_samples: $PRIOR_SAMPLES"
echo "[job] seed: $SEED"

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[job] python: $(which python)"
python - <<'PY'
import os
from pathlib import Path

import euclid_dsps
import jax

repo = Path(os.environ["REPO_DIR"]).resolve()
pkg = Path(euclid_dsps.__file__).resolve()
if repo not in pkg.parents:
    raise SystemExit(
        f"[job][error] euclid_dsps imported from {pkg}, not from repo {repo}. "
        "Check PYTHONPATH or the conda environment."
    )

print("[job] euclid_dsps:", pkg)
print("[job] jax_backend:", jax.default_backend())
print("[job] jax_devices:", jax.devices())
PY
nvidia-smi || true
git rev-parse --show-toplevel
git status --short --branch
git rev-parse HEAD

ensure_diffsky_dataset() {
  case "$FORCE_REBUILD_DATASET" in
    1|true|TRUE|yes|YES|y|Y)
      echo "[job] force rebuild requested"
      ;;
    *)
      if [[ -s "$DATASET" ]]; then
        echo "[job] dataset exists"
        ls -lh "$DATASET"
        return
      fi
      ;;
  esac
  if [[ -s "$DATASET" ]]; then
    echo "[job] rebuilding existing dataset from source subset"
  else
    echo "[job] dataset missing, rebuilding from source subset"
  fi
  if [[ ! -s "$SOURCE_DATASET" ]]; then
    cat <<MSG
[job][error] Missing SOURCE_DATASET.
[job][error] DATASET=$DATASET
[job][error] SOURCE_DATASET=$SOURCE_DATASET
[job][hint] Either copy the local subset parquet to Jean-Zay or build the
[job][hint] full source parquet first with diffsky-prepare-dataset.
MSG
    exit 2
  fi
  mkdir -p "$(dirname "$DATASET")"
  python -m euclid_dsps.cli diffsky-redshift-subset \
    --dataset "$SOURCE_DATASET" \
    --out "$DATASET" \
    --redshift-min "$REDSHIFT_MIN" \
    --redshift-max "$REDSHIFT_MAX" \
    --error-model "$ERROR_MODEL" \
    --snr "$ERROR_SNR"
  ls -lh "$DATASET"
}

ensure_diffsky_dataset

required_files=(
  "$DATASET"
  "$RAW_ROOT/diffsky_hltds_cosmos_260215_04_14_2026_transmission_curves.hdf5"
  "$RAW_ROOT/diffsky_hltds_cosmos_260215_04_14_2026_ssp_data.hdf5"
  "$RAW_ROOT/diffsky_hltds_cosmos_260215_04_14_2026_ssp_basis_k64_coeff16.hdf5"
  "$RAW_ROOT/diffsky_hltds_cosmos_260215_04_14_2026_t_table.hdf5"
)

missing=0
echo "[job] preflight assets"
for path in "${required_files[@]}"; do
  if [[ -s "$path" ]]; then
    ls -lh "$path"
  else
    echo "[job][missing] $path"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  cat <<'MSG'
[job][error] Missing required Diffsky assets.
If the compressed SSP basis is missing, build it once with:

python scripts/build_compressed_ssp_grid.py \
  --input Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/diffsky_hltds_cosmos_260215_04_14_2026_ssp_data.hdf5 \
  --output Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/diffsky_hltds_cosmos_260215_04_14_2026_ssp_basis_k64_coeff16.hdf5 \
  --k 64 \
  --basis-dtype float32 \
  --coeff-dtype float16 \
  --overwrite
MSG
  exit 2
fi

cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  diffsky-run-full-validation
  --dataset "$DATASET"
  --out "$OUT_DIR"
  --batch-size "$BATCH_SIZE"
  --jax-batch-size "$JAX_BATCH_SIZE"
  --decoder-sample-chunk-size "$DECODER_SAMPLE_CHUNK_SIZE"
  --prior-predictive-batch-size "$PRIOR_PREDICTIVE_BATCH_SIZE"
  --epochs "$EPOCHS"
  --n-samples "$N_SAMPLES"
  --posterior-samples "$POSTERIOR_SAMPLES"
  --prior-samples "$PRIOR_SAMPLES"
  --seed "$SEED"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

echo "[job] command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

echo "[job] final report: $OUT_DIR/full_validation_report.md"
echo "[job] final summary: $OUT_DIR/full_validation_summary.json"
echo "[job] end: $(date --iso-8601=seconds)"
