#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-$SOURCE_ROOT_BASE}"
OUTPUT_STAGE="${OUTPUT_STAGE:-full_cont120}"
CONFIG_26="${CONFIG_26:-configs/experiments/popcosmos_native15d_rws.yaml}"
CONFIG_24="${CONFIG_24:-configs/experiments/popcosmos_native15d_rws_24band.yaml}"
START_EPOCH="${START_EPOCH:-33}"
END_EPOCH="${END_EPOCH:-120}"
EXPECTED_SOURCE_EPOCH="${EXPECTED_SOURCE_EPOCH:-32}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
TRAIN_JAX_BATCH_SIZE="${TRAIN_JAX_BATCH_SIZE:-64}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-256}"
INFER_JAX_BATCH_SIZE="${INFER_JAX_BATCH_SIZE:-128}"
WALLTIME="${WALLTIME:-15:00:00}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
LOG_DIR="${LOG_DIR:-outputs/logs}"

case "$OUTPUT_STAGE" in
  *[!A-Za-z0-9._-]* | "")
    echo "[cosmos-rws15-cont-submit][error] invalid OUTPUT_STAGE=$OUTPUT_STAGE"
    exit 2
    ;;
esac
if (( START_EPOCH != EXPECTED_SOURCE_EPOCH + 1 )); then
  echo "[cosmos-rws15-cont-submit][error] START_EPOCH must follow the source epoch"
  exit 2
fi
if (( END_EPOCH < 100 )); then
  echo "[cosmos-rws15-cont-submit][error] END_EPOCH must be at least 100"
  exit 2
fi
if (( EXPECTED_GPUS != 4 )); then
  echo "[cosmos-rws15-cont-submit][error] this publication continuation requires 4 H100s per model"
  exit 2
fi
if (( TRAIN_JAX_BATCH_SIZE % EXPECTED_GPUS != 0 )); then
  echo "[cosmos-rws15-cont-submit][error] global JAX batch must divide across four GPUs"
  exit 2
fi

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"
test -s Data/cosmos2020/prepared/PREPOST_COMPLETE.json
test -s Data/cosmos2020/prepared/farmer_a24_n40000.parquet
test -s Data/cosmos2020/prepared/farmer_a24_full.parquet
test -s "$CONFIG_26"
test -s "$CONFIG_24"

evaluation_files=()
for variant in bands26 bands24_no_irac; do
  source_full="$SOURCE_ROOT_BASE/$variant/full"
  output="$OUTPUT_ROOT_BASE/$variant/$OUTPUT_STAGE"
  test -e "$source_full/DONE"
  test -s "$source_full/train/checkpoints/best.eqx"
  test -s "$source_full/train/checkpoints/best.eqx.json"
  test -s "$source_full/train/train_indices.npy"
  test -s "$source_full/train/validation_indices.npy"
  test -s "$source_full/train/feature_stats.json"
  test -s "$source_full/inference/redshift_predictions.parquet"
  test -s "$source_full/stage_contract.json"
  test ! -e "$output" || {
    echo "[cosmos-rws15-cont-submit][error] output already exists: $output"
    exit 2
  }
  evaluation_file=$(
    python - "$source_full/stage_contract.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["evaluation_cohort"]["row_indices"])
PY
  )
  test -s "$evaluation_file"
  evaluation_files+=("$evaluation_file")
done

python - \
  "${evaluation_files[0]}" \
  "${evaluation_files[1]}" \
  "$SOURCE_ROOT_BASE/bands26/full/train/train_indices.npy" \
  "$SOURCE_ROOT_BASE/bands24_no_irac/full/train/train_indices.npy" \
  "$SOURCE_ROOT_BASE/bands26/full/train/validation_indices.npy" \
  "$SOURCE_ROOT_BASE/bands24_no_irac/full/train/validation_indices.npy" <<'PY'
import sys

import numpy as np

labels = ("evaluation", "train", "validation")
for label, left, right in zip(labels, sys.argv[1::2], sys.argv[2::2], strict=True):
    if not np.array_equal(np.load(left), np.load(right)):
        raise SystemExit(f"band variants do not share {label} indices")
print("[cosmos-rws15-cont-submit] shared train/validation/evaluation cohorts: PASS")
PY

python scripts/validate_popcosmos_native15d.py \
  --config "$CONFIG_26" \
  --data-dir Data/cosmos2020/prepared \
  --asset-dir Data/cosmos2020/assets
python scripts/validate_popcosmos_native15d.py \
  --config "$CONFIG_24" \
  --data-dir Data/cosmos2020/prepared \
  --asset-dir Data/cosmos2020/assets

raw=$(sbatch --parsable \
  --array="0-1%${ARRAY_CONCURRENCY}" \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:4 \
  --cpus-per-task=96 \
  --time="$WALLTIME" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",SOURCE_ROOT_BASE="$SOURCE_ROOT_BASE",OUTPUT_ROOT_BASE="$OUTPUT_ROOT_BASE",OUTPUT_STAGE="$OUTPUT_STAGE",CONFIG_26="$CONFIG_26",CONFIG_24="$CONFIG_24",START_EPOCH="$START_EPOCH",END_EPOCH="$END_EPOCH",EXPECTED_SOURCE_EPOCH="$EXPECTED_SOURCE_EPOCH",EXPECTED_GPUS="$EXPECTED_GPUS",TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE",TRAIN_JAX_BATCH_SIZE="$TRAIN_JAX_BATCH_SIZE",INFER_BATCH_SIZE="$INFER_BATCH_SIZE",INFER_JAX_BATCH_SIZE="$INFER_JAX_BATCH_SIZE" \
  scripts/popcosmos_native15d_continue_full_h100.slurm)
job="${raw%%;*}"

stamp=$(date +%Y%m%d_%H%M%S)
submission_log="$LOG_DIR/submit_popcosmos_native15d_continuation_${stamp}.log"
latest_env="$LOG_DIR/popcosmos_native15d_continuation_latest.env"
{
  echo "continuation=$job"
  echo "source_root_base=$SOURCE_ROOT_BASE"
  echo "output_root_base=$OUTPUT_ROOT_BASE"
  echo "output_stage=$OUTPUT_STAGE"
  echo "epochs=${START_EPOCH}-${END_EPOCH}"
  echo "h100_per_model=$EXPECTED_GPUS"
  echo "global_jax_batch=$TRAIN_JAX_BATCH_SIZE"
  echo "per_device_batch=$((TRAIN_JAX_BATCH_SIZE / EXPECTED_GPUS))"
  echo "submission_log=$submission_log"
} | tee "$submission_log"

printf 'export CONTINUATION_JOB=%q\nexport SOURCE_ROOT_BASE=%q\nexport OUTPUT_ROOT_BASE=%q\nexport OUTPUT_STAGE=%q\nexport START_EPOCH=%q\nexport END_EPOCH=%q\nexport JOB_IDS=%q\nexport SUBMISSION_LOG=%q\n' \
  "$job" "$SOURCE_ROOT_BASE" "$OUTPUT_ROOT_BASE" "$OUTPUT_STAGE" \
  "$START_EPOCH" "$END_EPOCH" "$job" "$submission_log" \
  > "$latest_env"

echo "monitor: squeue -r -j $job -o '%.18i %.10T %.20j %.12R'"
echo "logs: outputs/logs/cosmos15_cont-${job}_{0,1}.{out,err}"
echo "latest_env=$latest_env"
