#!/bin/bash
#SBATCH --job-name=posthoc_em
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
TRAIN_LIMIT="${TRAIN_LIMIT:-5000}"
EVAL_LIMIT="${EVAL_LIMIT:-500}"
PROPOSAL_SAMPLES="${PROPOSAL_SAMPLES:-512}"
EVAL_SAMPLES="${EVAL_SAMPLES:-512}"
EM_ITERATIONS="${EM_ITERATIONS:-3}"
MSTEP_EPOCHS="${MSTEP_EPOCHS:-5}"
PROPOSAL_REFRESH_EPOCHS="${PROPOSAL_REFRESH_EPOCHS:-4}"
ALLOW_LOW_ESS="${ALLOW_LOW_ESS:-0}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

[[ "$EM_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[posthoc-em][error] EM_ITERATIONS must be a positive integer" >&2
  exit 2
}
for value_name in TRAIN_LIMIT EVAL_LIMIT PROPOSAL_SAMPLES EVAL_SAMPLES; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "[posthoc-em][error] $value_name must be a positive integer" >&2
    exit 2
  }
done
[[ "$MSTEP_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[posthoc-em][error] MSTEP_EPOCHS must be a positive integer" >&2
  exit 2
}
[[ "$PROPOSAL_REFRESH_EPOCHS" =~ ^[0-9]+$ ]] || {
  echo "[posthoc-em][error] PROPOSAL_REFRESH_EPOCHS must be a non-negative integer" >&2
  exit 2
}
[[ "$ALLOW_LOW_ESS" == "0" || "$ALLOW_LOW_ESS" == "1" ]] || {
  echo "[posthoc-em][error] ALLOW_LOW_ESS must be 0 or 1" >&2
  exit 2
}

FENIKS_ROOT="${FENIKS_ROOT:-outputs/runs/feniks_selfsup_learned_t2_jaxcosmo_v1/selfsup_rws_k8_t2}"
POPCOSMOS_ROOT="${POPCOSMOS_ROOT:-outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536/bands26/full_cont120}"

case "$TASK" in
  0)
    DOMAIN=feniks
    CONFIG="configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml"
    TRAIN_DATASET="Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/train.parquet"
    EVAL_DATASET="Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet"
    SOURCE_ROOT="$FENIKS_ROOT"
    INFER_COMMAND=amortized-infer-diffsky
    REFRESH_COMMAND=amortized-train-diffsky
    EVAL_COMMAND=amortized-infer-diffsky
    ;;
  1)
    DOMAIN=popcosmos
    CONFIG="configs/experiments/popcosmos_native15d_rws.yaml"
    TRAIN_DATASET="Data/cosmos2020/prepared/farmer_a24_n40000.parquet"
    EVAL_DATASET="Data/cosmos2020/prepared/farmer_a24_full.parquet"
    SOURCE_ROOT="$POPCOSMOS_ROOT"
    INFER_COMMAND=amortized-infer-cosmos
    REFRESH_COMMAND=amortized-train-cosmos
    EVAL_COMMAND=amortized-infer-cosmos
    ;;
  *)
    echo "[posthoc-em][error] task must be 0 or 1" >&2
    exit 2
    ;;
esac

SOURCE_CHECKPOINT="$SOURCE_ROOT/train/checkpoints/best.eqx"
FEATURE_STATS="$SOURCE_ROOT/train/feature_stats.json"
TRAIN_INDICES="$SOURCE_ROOT/train/train_indices.npy"
EVAL_INDICES="$SOURCE_ROOT/inference/inference_indices.npy"
OUT="$OUTPUT_ROOT/$DOMAIN"
BASELINE="$OUT/evaluation_source"
UPDATED="$OUT/evaluation_updated"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export MPLBACKEND=Agg

cd "$REPO_DIR"
mkdir -p outputs/logs "$OUT"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

for path in "$CONFIG" "$TRAIN_DATASET" "$EVAL_DATASET" "$SOURCE_CHECKPOINT" "${SOURCE_CHECKPOINT}.json" "$FEATURE_STATS" "$TRAIN_INDICES" "$EVAL_INDICES"; do
  test -s "$path" || { echo "[posthoc-em][error] missing: $path" >&2; exit 2; }
done
test ! -e "$OUT/DONE"

infer_bank() {
  local command="$1" dataset="$2" checkpoint="$3" indices="$4"
  local limit="$5" samples="$6" destination="$7"
  python -m euclid_dsps.cli --config "$CONFIG" "$command" \
    --runtime gpu --dataset "$dataset" --checkpoint "$checkpoint" \
    --feature-stats "$FEATURE_STATS" --row-indices-file "$indices" \
    --selection-mode sequential --limit "$limit" --batch-size 128 \
    --jax-batch-size 64 --posterior-samples "$samples" --prior-samples 256 \
    --decoder-sample-chunk-size 1 --prior-predictive-batch-size 64 \
    --out "$destination" --shard-outputs --no-posterior-predictive \
    --no-residual-samples
  python -m euclid_dsps.cli --config "$CONFIG" \
    amortized-finalize-inference --out "$destination" --limit "$limit"
}

echo "[posthoc-em] domain=$DOMAIN train=$TRAIN_LIMIT K=$PROPOSAL_SAMPLES eval=$EVAL_LIMIT"
echo "[posthoc-em] encoder_refresh_epochs=$PROPOSAL_REFRESH_EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

extra_em=()
if [[ "$ALLOW_LOW_ESS" == "1" ]]; then extra_em+=(--allow-low-ess); fi
CURRENT_CHECKPOINT="$SOURCE_CHECKPOINT"
FINAL_EM=""
COMPLETED_ITERATIONS=0
STOPPING_REASON="requested_iterations_completed"
for iteration in $(seq 1 "$EM_ITERATIONS"); do
  tag=$(printf '%03d' "$iteration")
  step="$OUT/iteration_${tag}"
  proposal="$step/proposal_k${PROPOSAL_SAMPLES}_n${TRAIN_LIMIT}"
  prior_update="$step/prior_update"
  echo "[posthoc-em] E-step $iteration/$EM_ITERATIONS checkpoint=$CURRENT_CHECKPOINT"
  infer_bank "$INFER_COMMAND" "$TRAIN_DATASET" "$CURRENT_CHECKPOINT" \
    "$TRAIN_INDICES" "$TRAIN_LIMIT" "$PROPOSAL_SAMPLES" "$proposal"
  python scripts/train_posthoc_empirical_bayes_prior.py \
    --posterior "$proposal" \
    --config "$CONFIG" \
    --checkpoint "$CURRENT_CHECKPOINT" \
    --out "$prior_update" \
    --iterations 1 \
    --mstep-epochs "$MSTEP_EPOCHS" \
    --object-batch-size 64 \
    --learning-rate 2e-5 \
    --trust-strength 0.05 \
    --trust-samples 512 \
    --weight-kind raw \
    --seed 260807 \
    "${extra_em[@]}"
  CURRENT_CHECKPOINT="$prior_update/checkpoints/best.eqx"
  FINAL_EM="$prior_update"
  COMPLETED_ITERATIONS="$iteration"
  selected_candidate=$(python - "$prior_update/em_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(summary["selected_candidate"])
PY
)
  if [[ "$selected_candidate" == "source_prior" ]]; then
    STOPPING_REASON="heldout_evidence_or_support_rejected_update"
    echo "[posthoc-em] early stop after iteration $iteration: $STOPPING_REASON"
    break
  fi
  if (( PROPOSAL_REFRESH_EPOCHS > 0 )); then
    refresh="$step/encoder_refresh"
    mkdir -p "$refresh"
    python - "$prior_update" "$refresh" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

prior_update = Path(sys.argv[1])
refresh = Path(sys.argv[2])
for split in ("train", "validation"):
    frame = pd.read_parquet(prior_update / f"{split}_object_identities.parquet")
    if frame.shape[1] != 1:
        raise SystemExit(f"unexpected {split} identity schema: {list(frame.columns)}")
    np.save(refresh / f"{split}_indices.npy", frame.iloc[:, 0].to_numpy(np.int64))
PY
    echo "[posthoc-em] encoder refresh $iteration/$EM_ITERATIONS with frozen prior"
    python -m euclid_dsps.cli --config "$CONFIG" "$REFRESH_COMMAND" \
      --runtime gpu \
      --dataset "$TRAIN_DATASET" \
      --out "$refresh/train" \
      --batch-size 128 \
      --jax-batch-size 64 \
      --epochs "$PROPOSAL_REFRESH_EPOCHS" \
      --initial-checkpoint "$CURRENT_CHECKPOINT" \
      --start-epoch 1 \
      --fixed-feature-stats "$FEATURE_STATS" \
      --train-indices-file "$refresh/train_indices.npy" \
      --validation-indices-file "$refresh/validation_indices.npy" \
      --n-samples 1 \
      --kl-annealing-epochs 1 \
      --validation-every 1 \
      --best-checkpoint-min-epoch 1 \
      --freeze-prior \
      --wake-start-encoder-epoch 1 \
      --wake-every-encoder-epochs 1 \
      --data-parallel single \
      --no-progress
    python - "$CONFIG" "$CURRENT_CHECKPOINT" \
      "$refresh/train/checkpoints/best.eqx" "$FEATURE_STATS" \
      "$refresh/train/feature_stats.json" "$refresh/refresh_contract.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import jax
import numpy as np

from euclid_dsps.amortized.train import load_checkpoint
from euclid_dsps.config import load_config

config_path, before_path, after_path, stats_before, stats_after, output = sys.argv[1:]
config = load_config(config_path)
before = load_checkpoint(before_path, config).prior
after = load_checkpoint(after_path, config).prior
before_leaves, before_tree = jax.tree_util.tree_flatten(before)
after_leaves, after_tree = jax.tree_util.tree_flatten(after)
if before_tree != after_tree or len(before_leaves) != len(after_leaves):
    raise SystemExit("prior tree changed during encoder-only refresh")
max_abs = 0.0
for left, right in zip(before_leaves, after_leaves, strict=True):
    if hasattr(left, "shape"):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.dtype == np.bool_ or right_array.dtype == np.bool_:
            if not np.array_equal(left_array, right_array):
                raise SystemExit("boolean prior state changed during encoder refresh")
        else:
            max_abs = max(
                max_abs, float(np.max(np.abs(left_array - right_array)))
            )
if max_abs != 0.0:
    raise SystemExit(f"prior changed during encoder-only refresh: max_abs={max_abs}")
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
if sha(stats_before) != sha(stats_after):
    raise SystemExit("feature stats changed during encoder-only refresh")
payload = {
    "status": "complete",
    "prior_frozen_exactly": True,
    "prior_max_abs_parameter_delta": max_abs,
    "feature_stats_sha256": sha(stats_after),
    "input_checkpoint": before_path,
    "refreshed_checkpoint": after_path,
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    CURRENT_CHECKPOINT="$refresh/train/checkpoints/best.eqx"
  else
    echo "[posthoc-em] encoder refresh disabled: fixed-q generalized EM"
  fi
done

python - "$OUT" "$EM_ITERATIONS" "$COMPLETED_ITERATIONS" \
  "$STOPPING_REASON" "$CURRENT_CHECKPOINT" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
requested_iterations = int(sys.argv[2])
completed_iterations = int(sys.argv[3])
stopping_reason = sys.argv[4]
selected_checkpoint = Path(sys.argv[5])
rows = []
for iteration in range(1, completed_iterations + 1):
    step = root / f"iteration_{iteration:03d}"
    summary_path = step / "prior_update/em_summary.json"
    checkpoint = step / "prior_update/checkpoints/best.eqx"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows.append(
        {
            "outer_iteration": iteration,
            "proposal": str(next(step.glob("proposal_k*_n*"))),
            "prior_update": str(step / "prior_update"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "em_summary": summary,
        }
    )
payload = {
    "status": "complete",
    "algorithm": "alternating one-step generalized EM with optional frozen-prior encoder refresh",
    "requested_iterations": requested_iterations,
    "completed_iterations": completed_iterations,
    "stopping_reason": stopping_reason,
    "selected_checkpoint": str(selected_checkpoint),
    "selected_checkpoint_sha256": hashlib.sha256(
        selected_checkpoint.read_bytes()
    ).hexdigest(),
    "encoder_refreshes": sorted(
        str(path.parent) for path in root.glob("iteration_*/encoder_refresh/refresh_contract.json")
    ),
    "iterations": rows,
}
(root / "checkpoints").mkdir(parents=True, exist_ok=True)
shutil.copy2(selected_checkpoint, root / "checkpoints/best.eqx")
shutil.copy2(
    Path(str(selected_checkpoint) + ".json"),
    root / "checkpoints/best.eqx.json",
)
(root / "alternating_em_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

UPDATED_CHECKPOINT="$OUT/checkpoints/best.eqx"
infer_bank "$EVAL_COMMAND" "$EVAL_DATASET" "$SOURCE_CHECKPOINT" \
  "$EVAL_INDICES" "$EVAL_LIMIT" "$EVAL_SAMPLES" "$BASELINE/proposal"
python scripts/importance_correct_posterior.py \
  --posterior "$BASELINE/proposal" \
  --truth "$BASELINE/proposal/inference_truth.parquet" \
  --out "$BASELINE/importance_correction" --resample-count 256 --seed 260807

infer_bank "$EVAL_COMMAND" "$EVAL_DATASET" "$UPDATED_CHECKPOINT" \
  "$EVAL_INDICES" "$EVAL_LIMIT" "$EVAL_SAMPLES" "$UPDATED/proposal"
python scripts/importance_correct_posterior.py \
  --posterior "$UPDATED/proposal" \
  --truth "$UPDATED/proposal/inference_truth.parquet" \
  --out "$UPDATED/importance_correction" --resample-count 256 --seed 260807

if [[ "$DOMAIN" == "feniks" ]]; then
  calibration_args=(--drop-nonfinite-truth)
else
  calibration_args=(
    --parameter z_obs
    --truth-column z_obs=redshift_true
    --drop-nonfinite-truth
  )
fi
python scripts/evaluate_feniks_mira.py \
  --truth "$UPDATED/proposal/inference_truth.parquet" \
  --posterior "source_iw=$BASELINE/importance_correction/resampled_samples" \
  --posterior "em_iw=$UPDATED/importance_correction/resampled_samples" \
  --out "$OUT/calibration_comparison/mira" \
  "${calibration_args[@]}" \
  --samples-per-object 256 \
  --num-regions 512 \
  --num-bootstrap 500 \
  --seed 260807
python scripts/evaluate_feniks_tarp.py \
  --truth "$UPDATED/proposal/inference_truth.parquet" \
  --posterior "source_iw=$BASELINE/importance_correction/resampled_samples" \
  --posterior "em_iw=$UPDATED/importance_correction/resampled_samples" \
  --out "$OUT/calibration_comparison/tarp" \
  "${calibration_args[@]}" \
  --samples-per-object 256 \
  --num-bootstrap 500 \
  --seed 260807

for path in \
  "$FINAL_EM/DONE" \
  "$OUT/alternating_em_summary.json" \
  "$OUT/checkpoints/best.eqx" \
  "$OUT/checkpoints/best.eqx.json" \
  "$BASELINE/importance_correction/DONE" \
  "$UPDATED/importance_correction/DONE" \
  "$OUT/calibration_comparison/mira/DONE" \
  "$OUT/calibration_comparison/tarp/DONE"
do
  test -e "$path"
done
printf '{"status":"complete","domain":"%s","train_objects":%s,"evaluation_objects":%s}\n' \
  "$DOMAIN" "$TRAIN_LIMIT" "$EVAL_LIMIT" > "$OUT/DONE"
echo "[posthoc-em] complete -> $OUT"
