#!/bin/bash
#SBATCH --job-name=cosmos15_array
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=15:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG_26="${CONFIG_26:-configs/experiments/popcosmos_native15d_rws.yaml}"
CONFIG_24="${CONFIG_24:-configs/experiments/popcosmos_native15d_rws_24band.yaml}"
ROOT_26="${ROOT_26:?Set ROOT_26 to the 26-band run root}"
ROOT_24="${ROOT_24:?Set ROOT_24 to the 24-band run root}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

case "$TASK" in
  0)
    export CONFIG="$CONFIG_26" ROOT_DIR="$ROOT_26"
    export BAND_VARIANT=26
    ;;
  1)
    export CONFIG="$CONFIG_24" ROOT_DIR="$ROOT_24"
    export BAND_VARIANT=24
    ;;
  *)
    echo "[cosmos-rws15-array][error] array task must be 0 (26-band) or 1 (24-band)"
    exit 2
    ;;
esac

export REPO_DIR
echo "[cosmos-rws15-array] stage=${STAGE:-n5k} task=$TASK band_variant=$BAND_VARIANT"
echo "[cosmos-rws15-array] config=$CONFIG root=$ROOT_DIR objective=reweighted_wake_sleep k=8"

exec bash "$REPO_DIR/scripts/popcosmos_native15d_rws_h100.slurm"
