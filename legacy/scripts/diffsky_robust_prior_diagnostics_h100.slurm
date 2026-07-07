#!/bin/bash
#SBATCH --job-name=diffsky_prior_diag
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-5
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?WORK is not set}/euclid-dsps-shine}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?WORK is not set}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"

ROWSET_FILE="${ROWSET_FILE:-outputs/rowsets/diffsky_hltds_nokl_reference/balanced20k.txt}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/diffsky_robust_prior_diagnostics}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
SEED="${SEED:-42}"

CHECKPOINT="${CHECKPOINT:-}"
FEATURE_STATS="${FEATURE_STATS:-}"
NN_RUN="${NN_RUN:-}"
CLOSURE_CHECKPOINT="${CLOSURE_CHECKPOINT:-$CHECKPOINT}"
CLOSURE_FEATURE_STATS="${CLOSURE_FEATURE_STATS:-$FEATURE_STATS}"
MAP_SWEEP_CHECKPOINT="${MAP_SWEEP_CHECKPOINT:-$CHECKPOINT}"
MAP_SWEEP_FEATURE_STATS="${MAP_SWEEP_FEATURE_STATS:-$FEATURE_STATS}"
GEOMETRY_SAMPLES="${GEOMETRY_SAMPLES:-200000}"
CLOSURE_BATCH_SIZE="${CLOSURE_BATCH_SIZE:-128}"
CLOSURE_MAP_N_STARTS="${CLOSURE_MAP_N_STARTS:-8}"
CLOSURE_MAP_MAXITER="${CLOSURE_MAP_MAXITER:-160}"
CLOSURE_MAP_LR="${CLOSURE_MAP_LR:-0.015}"
CLOSURE_MAP_START_CHUNK_SIZE="${CLOSURE_MAP_START_CHUNK_SIZE:-4}"
CLOSURE_MAP_PRIOR_DENSITY_SPACE="${CLOSURE_MAP_PRIOR_DENSITY_SPACE:-x}"
CLOSURE_RUN_MAP="${CLOSURE_RUN_MAP:-0}"
REDSHIFT_PROFILE_COUNT="${REDSHIFT_PROFILE_COUNT:-24}"
REDSHIFT_GRID_SIZE="${REDSHIFT_GRID_SIZE:-96}"
MAP_SWEEP_WEIGHTS="${MAP_SWEEP_WEIGHTS:-0,0.1,1.0}"
MAP_SWEEP_LIMIT="${MAP_SWEEP_LIMIT:-512}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export REPO_DIR
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT" "$LOG_DIR"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LOG_FILE="${LOG_DIR}/diffsky_prior_diag_${SLURM_JOB_ID:-manual}_${TASK_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[prior-diag] start: $(date --iso-8601=seconds)"
echo "[prior-diag] task: $TASK_ID"
echo "[prior-diag] repo: $REPO_DIR"
echo "[prior-diag] rowset: $ROWSET_FILE"

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python - <<'PY'
import os
from pathlib import Path

import euclid_dsps
import jax

repo = Path(os.environ["REPO_DIR"]).resolve()
pkg = Path(euclid_dsps.__file__).resolve()
if repo not in pkg.parents:
    raise SystemExit(f"[prior-diag][error] euclid_dsps imported from {pkg}, not {repo}")
print("[prior-diag] euclid_dsps:", pkg)
print("[prior-diag] jax_backend:", jax.default_backend())
print("[prior-diag] jax_devices:", jax.devices())
PY
nvidia-smi || true

case "$TASK_ID" in
  0)
    LABEL="geometry_annealed_current"
    CONFIG="configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_h100.yaml"
    cmd=(python -m euclid_dsps.cli --config "$CONFIG" diffsky-latent-prior-geometry --out "$RUN_ROOT/$LABEL" --n-samples "$GEOMETRY_SAMPLES" --seed "$SEED")
    ;;
  1)
    LABEL="geometry_annealed_zscale005"
    CONFIG="configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_zscale005_h100.yaml"
    cmd=(python -m euclid_dsps.cli --config "$CONFIG" diffsky-latent-prior-geometry --out "$RUN_ROOT/$LABEL" --n-samples "$GEOMETRY_SAMPLES" --seed "$SEED")
    ;;
  2)
    LABEL="geometry_annealed_zscale003"
    CONFIG="configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_zscale003_h100.yaml"
    cmd=(python -m euclid_dsps.cli --config "$CONFIG" diffsky-latent-prior-geometry --out "$RUN_ROOT/$LABEL" --n-samples "$GEOMETRY_SAMPLES" --seed "$SEED")
    ;;
  3)
    LABEL="geometry_annealed_zscale005_tau2safe"
    CONFIG="configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_zscale005_tau2safe_h100.yaml"
    cmd=(python -m euclid_dsps.cli --config "$CONFIG" diffsky-latent-prior-geometry --out "$RUN_ROOT/$LABEL" --n-samples "$GEOMETRY_SAMPLES" --seed "$SEED")
    ;;
  4)
    LABEL="closure_optimum"
    CONFIG="${CLOSURE_CONFIG:-configs/experiments/diffsky_hltds_autoencoder_nokl_deterministic_h100.yaml}"
    for path in "$ROWSET_FILE" "$CLOSURE_CHECKPOINT" "$CLOSURE_FEATURE_STATS"; do
      if [[ ! -s "$path" ]]; then
        echo "[prior-diag][missing] $path"
        exit 2
      fi
    done
    cmd=(python -m euclid_dsps.cli --config "$CONFIG" diffsky-closure-optimum-diagnostics --checkpoint "$CLOSURE_CHECKPOINT" --feature-stats "$CLOSURE_FEATURE_STATS" --row-indices-file "$ROWSET_FILE" --out "$RUN_ROOT/$LABEL" --batch-size "$CLOSURE_BATCH_SIZE" --map-n-starts "$CLOSURE_MAP_N_STARTS" --map-maxiter "$CLOSURE_MAP_MAXITER" --map-learning-rate "$CLOSURE_MAP_LR" --map-start-mode mixed --map-start-chunk-size "$CLOSURE_MAP_START_CHUNK_SIZE" --map-prior-density-space "$CLOSURE_MAP_PRIOR_DENSITY_SPACE" --redshift-profile-count "$REDSHIFT_PROFILE_COUNT" --redshift-grid-size "$REDSHIFT_GRID_SIZE" --selection-seed "$SEED")
    if [[ "$CLOSURE_RUN_MAP" != "1" ]]; then
      cmd+=(--skip-map)
    fi
    if [[ -n "$NN_RUN" ]]; then
      cmd+=(--nn-run "$NN_RUN")
    fi
    ;;
  5)
    LABEL="map_prior_sweep"
    CONFIG="${MAP_SWEEP_CONFIG:-configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_h100.yaml}"
    for path in "$ROWSET_FILE" "$MAP_SWEEP_CHECKPOINT" "$MAP_SWEEP_FEATURE_STATS"; do
      if [[ ! -s "$path" ]]; then
        echo "[prior-diag][missing] $path"
        exit 2
      fi
    done
    cmd=(python -m euclid_dsps.cli --config "$CONFIG" diffsky-map-prior-sweep --checkpoint "$MAP_SWEEP_CHECKPOINT" --feature-stats "$MAP_SWEEP_FEATURE_STATS" --row-indices-file "$ROWSET_FILE" --out "$RUN_ROOT/$LABEL" --weights "$MAP_SWEEP_WEIGHTS" --batch-size "$CLOSURE_BATCH_SIZE" --n-starts "$CLOSURE_MAP_N_STARTS" --maxiter "$CLOSURE_MAP_MAXITER" --learning-rate "$CLOSURE_MAP_LR" --start-mode mixed --start-chunk-size "$CLOSURE_MAP_START_CHUNK_SIZE" --prior-density-space "$CLOSURE_MAP_PRIOR_DENSITY_SPACE" --seed "$SEED" --selection-seed "$SEED")
    if [[ -n "$MAP_SWEEP_LIMIT" ]]; then
      cmd+=(--limit "$MAP_SWEEP_LIMIT")
    fi
    ;;
  *)
    echo "[prior-diag][error] unsupported task id: $TASK_ID"
    exit 2
    ;;
esac

echo "[prior-diag] label: $LABEL"
echo "[prior-diag] command:"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"

echo "[prior-diag] end: $(date --iso-8601=seconds)"
