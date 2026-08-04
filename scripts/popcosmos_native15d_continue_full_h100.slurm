#!/bin/bash
#SBATCH --job-name=cosmos15_cont
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=15:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:?Set SOURCE_ROOT_BASE to the completed array root}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-$SOURCE_ROOT_BASE}"
OUTPUT_STAGE="${OUTPUT_STAGE:-full_cont120}"
CONFIG_26="${CONFIG_26:-configs/experiments/popcosmos_native15d_rws.yaml}"
CONFIG_24="${CONFIG_24:-configs/experiments/popcosmos_native15d_rws_24band.yaml}"
DATA_DIR="${DATA_DIR:-Data/cosmos2020/prepared}"
ASSET_DIR="${ASSET_DIR:-Data/cosmos2020/assets}"
START_EPOCH="${START_EPOCH:-33}"
END_EPOCH="${END_EPOCH:-120}"
EXPECTED_SOURCE_EPOCH="${EXPECTED_SOURCE_EPOCH:-32}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
# Keep the original global optimizer batch. With four devices this is 16/device.
TRAIN_JAX_BATCH_SIZE="${TRAIN_JAX_BATCH_SIZE:-64}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-256}"
INFER_JAX_BATCH_SIZE="${INFER_JAX_BATCH_SIZE:-128}"
INFER_LIMIT="${INFER_LIMIT:-5000}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

case "$TASK" in
  0)
    VARIANT="bands26"
    CONFIG="$CONFIG_26"
    ;;
  1)
    VARIANT="bands24_no_irac"
    CONFIG="$CONFIG_24"
    ;;
  *)
    echo "[cosmos-rws15-cont][error] array task must be 0 or 1"
    exit 2
    ;;
esac

SOURCE_FULL="$SOURCE_ROOT_BASE/$VARIANT/full"
RUN_DIR="$OUTPUT_ROOT_BASE/$VARIANT/$OUTPUT_STAGE"
TRAIN_DIR="$RUN_DIR/train"
INFER_DIR="$RUN_DIR/inference"
TRAIN_DATASET="$DATA_DIR/farmer_a24_n40000.parquet"
FULL_DATASET="$DATA_DIR/farmer_a24_full.parquet"
SOURCE_CHECKPOINT="$SOURCE_FULL/train/checkpoints/best.eqx"
SOURCE_TRAIN_INDICES="$SOURCE_FULL/train/train_indices.npy"
SOURCE_VALIDATION_INDICES="$SOURCE_FULL/train/validation_indices.npy"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID:-manual}-${TASK}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"

module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

test -s "$DATA_DIR/PREPOST_COMPLETE.json"
test -s "$TRAIN_DATASET"
test -s "$FULL_DATASET"
test -e "$SOURCE_FULL/DONE"
test -s "$SOURCE_FULL/stage_contract.json"
test -s "$SOURCE_CHECKPOINT"
test -s "${SOURCE_CHECKPOINT}.json"
test -s "$SOURCE_TRAIN_INDICES"
test -s "$SOURCE_VALIDATION_INDICES"
test -s "$SOURCE_FULL/train/feature_stats.json"
test -s "$SOURCE_FULL/inference/redshift_predictions.parquet"
test ! -e "$RUN_DIR" || {
  echo "[cosmos-rws15-cont][error] refusing to overwrite $RUN_DIR"
  exit 2
}

EVAL_INDICES=$(
  python - "$SOURCE_FULL/stage_contract.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["evaluation_cohort"]["row_indices"])
PY
)
test -s "$EVAL_INDICES"

python scripts/validate_popcosmos_native15d.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --asset-dir "$ASSET_DIR"

python - "$SOURCE_CHECKPOINT" "$EXPECTED_SOURCE_EPOCH" <<'PY'
import json
import sys

checkpoint = sys.argv[1]
expected_epoch = int(sys.argv[2])
metadata = json.load(open(f"{checkpoint}.json", encoding="utf-8"))
if int(metadata.get("epoch", -1)) != expected_epoch:
    raise SystemExit(
        "source checkpoint epoch mismatch: "
        f"expected={expected_epoch} got={metadata.get('epoch')}"
    )
PY

python - "$EXPECTED_GPUS" "$TRAIN_JAX_BATCH_SIZE" <<'PY'
import sys

import jax

expected = int(sys.argv[1])
global_batch = int(sys.argv[2])
devices = jax.local_devices()
print("[cosmos-rws15-cont] backend:", jax.default_backend())
print("[cosmos-rws15-cont] devices:", devices)
if jax.default_backend() != "gpu":
    raise SystemExit("JAX did not select the H100 backend")
if len(devices) != expected:
    raise SystemExit(f"expected {expected} local GPUs, got {len(devices)}")
if global_batch % expected:
    raise SystemExit(
        f"global JAX batch {global_batch} is not divisible by {expected} GPUs"
    )
print(
    "[cosmos-rws15-cont] data_parallel=pmap "
    f"global_batch={global_batch} per_device_batch={global_batch // expected}"
)
PY

mkdir -p "$RUN_DIR"
echo "[cosmos-rws15-cont] variant=$VARIANT source=$SOURCE_FULL"
echo "[cosmos-rws15-cont] epochs=$START_EPOCH..$END_EPOCH GPUs=$EXPECTED_GPUS"
echo "[cosmos-rws15-cont] evaluation_indices=$EVAL_INDICES limit=$INFER_LIMIT"
nvidia-smi

python -m euclid_dsps.cli --config "$CONFIG" amortized-train-cosmos \
  --runtime gpu \
  --dataset "$TRAIN_DATASET" \
  --out "$TRAIN_DIR" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --jax-batch-size "$TRAIN_JAX_BATCH_SIZE" \
  --epochs "$END_EPOCH" \
  --initial-checkpoint "$SOURCE_CHECKPOINT" \
  --start-epoch "$START_EPOCH" \
  --train-indices-file "$SOURCE_TRAIN_INDICES" \
  --validation-indices-file "$SOURCE_VALIDATION_INDICES" \
  --n-samples 1 \
  --kl-annealing-epochs 40 \
  --validation-every 8 \
  --best-checkpoint-min-epoch "$START_EPOCH" \
  --data-parallel pmap \
  --no-progress

python - \
  "$TRAIN_DIR/training_summary.json" \
  "$TRAIN_DIR/train_indices.npy" \
  "$SOURCE_TRAIN_INDICES" \
  "$TRAIN_DIR/validation_indices.npy" \
  "$SOURCE_VALIDATION_INDICES" \
  "$START_EPOCH" \
  "$END_EPOCH" \
  "$EXPECTED_GPUS" \
  "$TRAIN_JAX_BATCH_SIZE" \
  "$TRAIN_DIR/feature_stats.json" \
  "$SOURCE_FULL/train/feature_stats.json" <<'PY'
import json
import sys

import numpy as np

(
    summary_path,
    train_indices_path,
    source_train_indices_path,
    validation_indices_path,
    source_validation_indices_path,
    start_epoch,
    end_epoch,
    expected_gpus,
    global_batch,
    feature_stats_path,
    source_feature_stats_path,
) = sys.argv[1:]
summary = json.load(open(summary_path, encoding="utf-8"))
parallel = summary["data_parallel"]
checks = {
    "start_epoch": (int(summary["start_epoch"]), int(start_epoch)),
    "epochs": (int(summary["epochs"]), int(end_epoch)),
    "local_device_count": (
        int(parallel["local_device_count"]),
        int(expected_gpus),
    ),
    "global_batch_size": (int(parallel["global_batch_size"]), int(global_batch)),
    "per_device_batch_size": (
        int(parallel["per_device_batch_size"]),
        int(global_batch) // int(expected_gpus),
    ),
}
for name, (actual, expected) in checks.items():
    if actual != expected:
        raise SystemExit(f"{name} mismatch: expected={expected} got={actual}")
if parallel["effective"] != "pmap":
    raise SystemExit(f"training did not use pmap: {parallel}")
if summary.get("optimizer_state_resumed") is not False:
    raise SystemExit("unexpected optimizer-state continuation semantics")
if int(summary.get("best_checkpoint_epoch", -1)) < 40:
    raise SystemExit("no valid post-annealing checkpoint was selected")
for current, source, label in (
    (train_indices_path, source_train_indices_path, "train"),
    (validation_indices_path, source_validation_indices_path, "validation"),
):
    if not np.array_equal(np.load(current), np.load(source)):
        raise SystemExit(f"{label} indices changed during continuation")
if json.load(open(feature_stats_path, encoding="utf-8")) != json.load(
    open(source_feature_stats_path, encoding="utf-8")
):
    raise SystemExit("feature statistics changed during continuation")
print("[cosmos-rws15-cont] training continuation contract: PASS")
PY

python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-cosmos \
  --dataset "$FULL_DATASET" \
  --checkpoint "$TRAIN_DIR/checkpoints/best.eqx" \
  --feature-stats "$TRAIN_DIR/feature_stats.json" \
  --out "$INFER_DIR" \
  --limit "$INFER_LIMIT" \
  --batch-size "$INFER_BATCH_SIZE" \
  --jax-batch-size "$INFER_JAX_BATCH_SIZE" \
  --posterior-samples 128 \
  --prior-samples 4096 \
  --decoder-sample-chunk-size 1 \
  --prior-predictive-batch-size 128 \
  --selection-mode sequential \
  --row-indices-file "$EVAL_INDICES" \
  --shard-outputs \
  --no-residual-samples

python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-inference --out "$INFER_DIR" --limit "$INFER_LIMIT"

python scripts/evaluate_popcosmos_native15d_redshift.py \
  --dataset "$FULL_DATASET" \
  --inference "$INFER_DIR"

python - \
  "$INFER_DIR/redshift_predictions.parquet" \
  "$SOURCE_FULL/inference/redshift_predictions.parquet" <<'PY'
import sys

import numpy as np
import pyarrow.parquet as pq

columns = ["object_id", "row_index"]
current = pq.read_table(sys.argv[1], columns=columns).to_pandas()
source = pq.read_table(sys.argv[2], columns=columns).to_pandas()
if len(current) != len(source):
    raise SystemExit(
        f"held-out inference size changed: source={len(source)} current={len(current)}"
    )
for column in columns:
    if not np.array_equal(current[column].to_numpy(), source[column].to_numpy()):
        raise SystemExit(f"held-out inference {column} ordering changed")
print("[cosmos-rws15-cont] held-out inference cohort: unchanged")
PY

python - "$INFER_DIR" <<'PY'
import json
import sys

from euclid_dsps.amortized.collapse_gates import write_inference_collapse_gate

out = sys.argv[1]
gate = write_inference_collapse_gate(out)
metrics = json.load(open(f"{out}/redshift_metrics.json", encoding="utf-8"))
print("[cosmos-rws15-cont] redshift metrics:", json.dumps(metrics["metrics"], sort_keys=True))
print("[cosmos-rws15-cont] inference science gate:", gate["status"])
PY

python - \
  "$RUN_DIR/stage_contract.json" \
  "$SOURCE_FULL" \
  "$SOURCE_CHECKPOINT" \
  "$EVAL_INDICES" \
  "$INFER_LIMIT" \
  "$START_EPOCH" \
  "$END_EPOCH" \
  "$EXPECTED_GPUS" \
  "$OUTPUT_STAGE" <<'PY'
import hashlib
import json
import os
import subprocess
import sys

(
    out,
    source_full,
    source_checkpoint,
    evaluation_indices,
    infer_limit,
    start_epoch,
    end_epoch,
    expected_gpus,
    output_stage,
) = sys.argv[1:]

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "status": "complete",
    "stage": output_stage,
    "science_target": "z_obs_only",
    "latent_model": "native_feniks_spline15d",
    "nuisance_latent_dimensions": 14,
    "training_catalog": "farmer_a24_n40000.parquet",
    "source_run": source_full,
    "source_checkpoint": source_checkpoint,
    "source_checkpoint_sha256": sha256(source_checkpoint),
    "start_epoch": int(start_epoch),
    "end_epoch": int(end_epoch),
    "optimizer_state_resumed": False,
    "data_parallel": {
        "mode": "pmap",
        "local_h100_count": int(expected_gpus),
    },
    "evaluation_cohort": {
        "row_indices": evaluation_indices,
        "row_indices_sha256": sha256(evaluation_indices),
        "selected_rows": int(infer_limit),
        "excluded_from_training_pool": True,
        "redshift_used_for_selection": False,
    },
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
}
with open(out, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY

test -s "$TRAIN_DIR/training_summary.json"
test -s "$TRAIN_DIR/checkpoints/best.eqx"
test -s "$INFER_DIR/inference_summary.json"
test -s "$INFER_DIR/posterior_summary.parquet"
test -s "$INFER_DIR/redshift_metrics.json"
test -s "$INFER_DIR/redshift_predictions.parquet"
touch "$RUN_DIR/DONE"

if ! python scripts/validate_popcosmos_native15d.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --asset-dir "$ASSET_DIR" \
  --run-dir "$RUN_DIR"
then
  rm -f "$RUN_DIR/DONE"
  exit 1
fi

echo "[cosmos-rws15-cont] complete -> $RUN_DIR"
