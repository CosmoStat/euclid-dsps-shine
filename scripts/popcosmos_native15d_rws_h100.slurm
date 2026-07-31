#!/bin/bash
#SBATCH --job-name=cosmos_rws15
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

STAGE="${STAGE:-n5k}"
CONFIG="${CONFIG:-configs/experiments/popcosmos_native15d_rws.yaml}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
DATA_DIR="${DATA_DIR:-Data/cosmos2020/prepared}"
ROOT_DIR="${ROOT_DIR:?Set ROOT_DIR to an isolated native15d run root}"
MAP_DIR="${MAP_DIR:-}"
NORMALIZATION_CHECKPOINT="${NORMALIZATION_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"

case "$STAGE" in
  n5k)
    SIZE=5000; EPOCHS=12; BEST_MIN_EPOCH=4; INFER_LIMIT=512; PREVIOUS=""
    ;;
  n20k)
    SIZE=20000; EPOCHS=20; BEST_MIN_EPOCH=4; INFER_LIMIT=1000
    PREVIOUS="$ROOT_DIR/n5k"
    ;;
  n40k)
    SIZE=40000; EPOCHS=32; BEST_MIN_EPOCH=8; INFER_LIMIT=5000
    PREVIOUS="$ROOT_DIR/n20k"
    ;;
  *)
    echo "[cosmos-rws15][error] STAGE must be n5k,n20k,n40k"
    exit 2
    ;;
esac

DATASET="$DATA_DIR/farmer_a24_n${SIZE}.parquet"
FULL_DATASET="$DATA_DIR/farmer_a24_full.parquet"
TRAIN_POOL="$DATA_DIR/farmer_a24_n40000.parquet"
RUN_DIR="$ROOT_DIR/$STAGE"
TRAIN_DIR="$RUN_DIR/train"
INFER_DIR="$RUN_DIR/inference"
EVAL_INDICES="$ROOT_DIR/evaluation_indices_n5000.npy"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
TRAIN_JAX_BATCH_SIZE="${TRAIN_JAX_BATCH_SIZE:-64}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-256}"
INFER_JAX_BATCH_SIZE="${INFER_JAX_BATCH_SIZE:-128}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID:-manual}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$RUN_DIR"

module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

test -s "$DATA_DIR/PREPOST_COMPLETE.json"
test -s "$DATASET"
test -s "$FULL_DATASET"
test -s "$TRAIN_POOL"
test -s "$NORMALIZATION_CHECKPOINT"
test -s "${NORMALIZATION_CHECKPOINT}.json"
test ! -e "$TRAIN_DIR" || {
  echo "[cosmos-rws15][error] refusing to overwrite $TRAIN_DIR"
  exit 2
}

INITIAL_ARGS=()
if [[ "$STAGE" == "n5k" ]]; then
  test -n "$MAP_DIR" || {
    echo "[cosmos-rws15][error] n5k requires MAP_DIR"
    exit 2
  }
  test -e "$MAP_DIR/DONE"
  python - "$MAP_DIR/map_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("checkpoint") is not None or summary.get("uses_learned_prior"):
    raise SystemExit("[cosmos-rws15][error] MAP gate was not likelihood-only")
if summary.get("photometric_fit", {}).get("n_parameters") != 15:
    raise SystemExit("[cosmos-rws15][error] MAP gate was not native 15D")
PY
else
  test -e "$PREVIOUS/DONE"
  test -s "$PREVIOUS/train/checkpoints/best.eqx"
  INITIAL_ARGS=(--initial-checkpoint "$PREVIOUS/train/checkpoints/best.eqx")
fi

python scripts/validate_popcosmos_native15d.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --asset-dir Data/cosmos2020/assets

python scripts/build_popcosmos_native15d_eval_indices.py \
  --full "$FULL_DATASET" \
  --exclude "$TRAIN_POOL" \
  --out "$EVAL_INDICES" \
  --size 5000 \
  --seed 260731

python - "$CONFIG" <<'PY'
import sys

import jax

from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.config import load_config

config = load_config(sys.argv[1])
amortized = amortized_config(config)
names = tuple(config["fit"]["free_parameters"])
target = config.get("science_target", {})
print("[cosmos-rws15] backend:", jax.default_backend())
print("[cosmos-rws15] devices:", jax.devices())
print("[cosmos-rws15] latent:", names)
print("[cosmos-rws15] science target:", target)
if jax.default_backend() != "gpu":
    raise SystemExit("[cosmos-rws15][error] JAX did not select the H100")
if len(names) != 15 or config["model"]["sfh_model"] != "spline15d":
    raise SystemExit("[cosmos-rws15][error] native spline15D is not active")
if target.get("reported_parameters") != ["z_obs"]:
    raise SystemExit("[cosmos-rws15][error] science target is not redshift-only")
if len(config["bands"]) != 26:
    raise SystemExit("[cosmos-rws15][error] expected 26 COSMOS passbands")
if amortized["likelihood"]["type"] != "student_t":
    raise SystemExit("[cosmos-rws15][error] Student-t likelihood is not active")
PY

echo "[cosmos-rws15] stage=$STAGE dataset=$DATASET epochs=$EPOCHS"
echo "[cosmos-rws15] train_batch=$TRAIN_BATCH_SIZE jax_batch=$TRAIN_JAX_BATCH_SIZE"
echo "[cosmos-rws15] science_target=z_obs nuisance_latents=14"
nvidia-smi

python -m euclid_dsps.cli --config "$CONFIG" amortized-train-cosmos \
  --runtime gpu \
  --dataset "$DATASET" \
  --out "$TRAIN_DIR" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --jax-batch-size "$TRAIN_JAX_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --n-samples 1 \
  --best-checkpoint-min-epoch "$BEST_MIN_EPOCH" \
  --data-parallel single \
  --no-progress \
  "${INITIAL_ARGS[@]}"

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

python - "$INFER_DIR" <<'PY'
import json
import sys

from euclid_dsps.amortized.collapse_gates import write_inference_collapse_gate

out = sys.argv[1]
gate = write_inference_collapse_gate(out)
metrics = json.load(open(f"{out}/redshift_metrics.json", encoding="utf-8"))
print("[cosmos-rws15] redshift metrics:", json.dumps(metrics["metrics"], sort_keys=True))
print("[cosmos-rws15] inference gate:", gate["status"])
PY

test -s "$TRAIN_DIR/training_summary.json"
test -s "$TRAIN_DIR/checkpoints/best.eqx"
test -s "$INFER_DIR/inference_summary.json"
test -s "$INFER_DIR/posterior_summary.parquet"
test -s "$INFER_DIR/redshift_metrics.json"
test -s "$INFER_DIR/redshift_predictions.parquet"
test -s "$INFER_DIR/photoz_metrics.csv"

python - "$RUN_DIR/stage_contract.json" "$EVAL_INDICES" "$INFER_LIMIT" <<'PY'
import json
import os
import subprocess
import sys

payload = {
    "status": "complete",
    "stage": os.environ["STAGE"],
    "science_target": "z_obs_only",
    "latent_model": "native_feniks_spline15d",
    "nuisance_latent_dimensions": 14,
    "observations": "COSMOS2020 Farmer 26-band fluxes and reported errors",
    "evaluation_cohort": {
        "row_indices": sys.argv[2],
        "selected_rows": int(sys.argv[3]),
        "excluded_from_training_pool": True,
        "redshift_used_for_selection": False,
    },
    "likelihood": "Student-t2",
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY

if [[ "$STAGE" == "n40k" ]]; then
  python scripts/summarize_popcosmos_native15d_scaling.py --root "$ROOT_DIR"
  test -s "$ROOT_DIR/redshift_scaling/redshift_scaling_metrics.csv"
  test -s "$ROOT_DIR/redshift_scaling/redshift_scaling_summary.json"
fi

touch "$RUN_DIR/DONE"
if ! python scripts/validate_popcosmos_native15d.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --asset-dir Data/cosmos2020/assets \
  --run-dir "$RUN_DIR"
then
  rm -f "$RUN_DIR/DONE"
  exit 1
fi
echo "[cosmos-rws15] complete -> $RUN_DIR"
