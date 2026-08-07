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
ALLOW_LOW_ESS="${ALLOW_LOW_ESS:-0}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

FENIKS_ROOT="${FENIKS_ROOT:-outputs/runs/feniks_selfsup_learned_t2_jaxcosmo_v1/selfsup_rws_k8_t2}"
POPCOSMOS_ROOT="${POPCOSMOS_ROOT:-outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536/bands26/full_cont120}"

case "$TASK" in
  0)
    DOMAIN=feniks
    CONFIG="configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml"
    TRAIN_DATASET="Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/train.parquet"
    EVAL_DATASET="Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet"
    SOURCE_ROOT="$FENIKS_ROOT"
    TRAIN_COMMAND=amortized-infer-diffsky
    EVAL_COMMAND=amortized-infer-diffsky
    ;;
  1)
    DOMAIN=popcosmos
    CONFIG="configs/experiments/popcosmos_native15d_rws.yaml"
    TRAIN_DATASET="Data/cosmos2020/prepared/farmer_a24_n40000.parquet"
    EVAL_DATASET="Data/cosmos2020/prepared/farmer_a24_full.parquet"
    SOURCE_ROOT="$POPCOSMOS_ROOT"
    TRAIN_COMMAND=amortized-infer-cosmos
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

extra_em=()
if [[ "$ALLOW_LOW_ESS" == "1" ]]; then extra_em+=(--allow-low-ess); fi
CURRENT_CHECKPOINT="$SOURCE_CHECKPOINT"
FINAL_EM=""
for iteration in $(seq 1 "$EM_ITERATIONS"); do
  tag=$(printf '%03d' "$iteration")
  step="$OUT/iteration_${tag}"
  proposal="$step/proposal_k${PROPOSAL_SAMPLES}_n${TRAIN_LIMIT}"
  prior_update="$step/prior_update"
  echo "[posthoc-em] E-step $iteration/$EM_ITERATIONS checkpoint=$CURRENT_CHECKPOINT"
  infer_bank "$TRAIN_COMMAND" "$TRAIN_DATASET" "$CURRENT_CHECKPOINT" \
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
done

python - "$OUT" "$EM_ITERATIONS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
iterations = int(sys.argv[2])
rows = []
for iteration in range(1, iterations + 1):
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
    "algorithm": "alternating proposal refresh and one-step generalized EM",
    "iterations": rows,
}
(root / "alternating_em_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

UPDATED_CHECKPOINT="$CURRENT_CHECKPOINT"
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
