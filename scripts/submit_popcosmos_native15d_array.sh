#!/bin/bash
set -Eeuo pipefail

# Submit two synchronized native-15D scaling chains:
#   array task 0 = all 26 COSMOS bands
#   array task 1 = the 24-band no-IRAC ablation
# Each task owns one H100; stages are chained 5k -> 20k -> full. The
# validated likelihood-only MAP is not an input to RWS training.

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG_26="${CONFIG_26:-configs/experiments/popcosmos_native15d_rws.yaml}"
CONFIG_24="${CONFIG_24:-configs/experiments/popcosmos_native15d_rws_24band.yaml}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
TRAIN_JAX_BATCH_SIZE="${TRAIN_JAX_BATCH_SIZE:-64}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-256}"
INFER_JAX_BATCH_SIZE="${INFER_JAX_BATCH_SIZE:-128}"
RESUME_N5K="${RESUME_N5K:-0}"

ROOT_BASE="${ROOT_BASE:-outputs/runs/popcosmos_native15d_array_$(date +%Y%m%d_%H%M%S)}"
ROOT_26="$ROOT_BASE/bands26"
ROOT_24="$ROOT_BASE/bands24_no_irac"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LATEST_ENV="$LOG_DIR/popcosmos_native15d_array_latest.env"

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"
test -s Data/cosmos2020/prepared/PREPOST_COMPLETE.json
test -s Data/cosmos2020/prepared/farmer_a24_n40000.parquet
test -s "$CONFIG_26"
test -s "$CONFIG_24"
if [[ "$RESUME_N5K" == "1" ]]; then
  test -s "$ROOT_26/n5k/train/training_summary.json"
  test -s "$ROOT_26/n5k/train/checkpoints/best.eqx"
  test -s "$ROOT_24/n5k/train/training_summary.json"
  test -s "$ROOT_24/n5k/train/checkpoints/best.eqx"
  echo "[cosmos-rws15-array] resuming n5k inference from existing checkpoints"
else
  test ! -e "$ROOT_BASE" || {
    echo "[cosmos-rws15-array][error] run root already exists: $ROOT_BASE"
    exit 2
  }
fi

export REPO_DIR MINICONDA_PATH CONDA_ENV CONFIG_26 CONFIG_24
export ROOT_26 ROOT_24
export TRAIN_BATCH_SIZE TRAIN_JAX_BATCH_SIZE INFER_BATCH_SIZE INFER_JAX_BATCH_SIZE

COMMON_EXPORT="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,CONFIG_26=$CONFIG_26,CONFIG_24=$CONFIG_24,ROOT_26=$ROOT_26,ROOT_24=$ROOT_24,TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE,TRAIN_JAX_BATCH_SIZE=$TRAIN_JAX_BATCH_SIZE,INFER_BATCH_SIZE=$INFER_BATCH_SIZE,INFER_JAX_BATCH_SIZE=$INFER_JAX_BATCH_SIZE"

submit_stage() {
  local stage="$1" walltime="$2" dependency_arg="${3:-}"
  local raw job skip_training=0
  if [[ "$stage" == "n5k" && "$RESUME_N5K" == "1" ]]; then
    skip_training=1
  fi
  raw=$(sbatch --parsable \
    --array="0-1%${ARRAY_CONCURRENCY}" \
    --time="$walltime" \
    ${dependency_arg:+--dependency="$dependency_arg"} \
    --export="$COMMON_EXPORT,STAGE=$stage,SKIP_TRAINING=$skip_training" \
    scripts/popcosmos_native15d_array_h100.slurm)
  job="${raw%%;*}"
  printf '%s\n' "$job"
}

n5k=$(submit_stage n5k 04:00:00)
n20k=$(submit_stage n20k 08:00:00 "afterok:$n5k")
full=$(submit_stage full 15:00:00 "afterok:$n20k")

stamp=$(date +%Y%m%d_%H%M%S)
submission_log="$LOG_DIR/submit_popcosmos_native15d_array_${stamp}.log"
{
  echo "n5k=$n5k"
  echo "n20k=$n20k"
  echo "full=$full"
  echo "root_base=$ROOT_BASE"
  echo "root_26=$ROOT_26"
  echo "root_24=$ROOT_24"
  echo "objective=reweighted_wake_sleep"
  echo "wake_particles=8"
  echo "resume_n5k=$RESUME_N5K"
  echo "config_26=$CONFIG_26"
  echo "config_24=$CONFIG_24"
  echo "train_jax_batch_size=$TRAIN_JAX_BATCH_SIZE"
  echo "submission_log=$submission_log"
} | tee "$submission_log"

job_ids="$n5k,$n20k,$full"
printf 'export ROOT_BASE=%q\nexport ROOT_26=%q\nexport ROOT_24=%q\nexport CONFIG_26=%q\nexport CONFIG_24=%q\nexport RESUME_N5K=%q\nexport JOB_IDS=%q\nexport SUBMISSION_LOG=%q\n' \
  "$ROOT_BASE" "$ROOT_26" "$ROOT_24" "$CONFIG_26" "$CONFIG_24" \
  "$RESUME_N5K" "$job_ids" "$submission_log" \
  > "$LATEST_ENV"

echo "monitor: squeue -j $job_ids"
echo "logs: outputs/logs/cosmos15_array-<arrayjob>_<taskid>.out"
echo "latest_env=$LATEST_ENV"
