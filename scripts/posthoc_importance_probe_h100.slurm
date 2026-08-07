#!/bin/bash
#SBATCH --job-name=posthoc_iw
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
LIMIT="${LIMIT:-256}"
BUDGETS_CSV="${BUDGETS_CSV:-128,512,2048}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

IFS=',' read -r -a BUDGETS <<< "$BUDGETS_CSV"
N_BUDGETS="${#BUDGETS[@]}"
DOMAIN_INDEX=$((TASK / N_BUDGETS))
BUDGET_INDEX=$((TASK % N_BUDGETS))
BUDGET="${BUDGETS[$BUDGET_INDEX]}"

FENIKS_ROOT="${FENIKS_ROOT:-outputs/runs/feniks_selfsup_learned_t2_jaxcosmo_v1/selfsup_rws_k8_t2}"
POPCOSMOS_ROOT="${POPCOSMOS_ROOT:-outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536/bands26/full_cont120}"

case "$DOMAIN_INDEX" in
  0)
    DOMAIN=feniks
    CONFIG="configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml"
    DATASET="Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet"
    SOURCE_ROOT="$FENIKS_ROOT"
    COMMAND=amortized-infer-diffsky
    ;;
  1)
    DOMAIN=popcosmos
    CONFIG="configs/experiments/popcosmos_native15d_rws.yaml"
    DATASET="Data/cosmos2020/prepared/farmer_a24_full.parquet"
    SOURCE_ROOT="$POPCOSMOS_ROOT"
    COMMAND=amortized-infer-cosmos
    ;;
  *)
    echo "[posthoc-iw][error] invalid array task: $TASK" >&2
    exit 2
    ;;
esac

CHECKPOINT="$SOURCE_ROOT/train/checkpoints/best.eqx"
FEATURE_STATS="$SOURCE_ROOT/train/feature_stats.json"
ROW_INDICES="$SOURCE_ROOT/inference/inference_indices.npy"
OUT="$OUTPUT_ROOT/$DOMAIN/k${BUDGET}_n${LIMIT}"
PROPOSAL="$OUT/proposal"
CORRECTION="$OUT/importance_correction"

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

for path in "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" "$FEATURE_STATS" "$ROW_INDICES"; do
  test -s "$path" || { echo "[posthoc-iw][error] missing: $path" >&2; exit 2; }
done
test ! -e "$OUT/DONE"

echo "[posthoc-iw] domain=$DOMAIN K=$BUDGET objects=$LIMIT out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m euclid_dsps.cli --config "$CONFIG" "$COMMAND" \
  --runtime gpu \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" \
  --row-indices-file "$ROW_INDICES" \
  --selection-mode sequential \
  --limit "$LIMIT" \
  --batch-size 128 \
  --jax-batch-size 64 \
  --posterior-samples "$BUDGET" \
  --prior-samples 256 \
  --decoder-sample-chunk-size 1 \
  --prior-predictive-batch-size 64 \
  --out "$PROPOSAL" \
  --shard-outputs \
  --no-posterior-predictive \
  --no-residual-samples

python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-inference --out "$PROPOSAL" --limit "$LIMIT"

python scripts/importance_correct_posterior.py \
  --posterior "$PROPOSAL" \
  --truth "$PROPOSAL/inference_truth.parquet" \
  --out "$CORRECTION" \
  --resample-count 256 \
  --seed 260807

if [[ "$DOMAIN" == "feniks" ]]; then
  calibration_args=(--drop-nonfinite-truth)
  mira_out="$CORRECTION/mira15d"
  tarp_out="$CORRECTION/tarp15d"
else
  calibration_args=(
    --parameter z_obs
    --truth-column z_obs=redshift_true
    --drop-nonfinite-truth
  )
  mira_out="$CORRECTION/mira_redshift"
  tarp_out="$CORRECTION/tarp_redshift"
fi
python scripts/evaluate_feniks_mira.py \
  --truth "$PROPOSAL/inference_truth.parquet" \
  --posterior "iw_psis=$CORRECTION/resampled_samples" \
  --out "$mira_out" \
  "${calibration_args[@]}" \
  --samples-per-object 256 \
  --num-regions 256 \
  --num-bootstrap 200 \
  --seed 260807
python scripts/evaluate_feniks_tarp.py \
  --truth "$PROPOSAL/inference_truth.parquet" \
  --posterior "iw_psis=$CORRECTION/resampled_samples" \
  --out "$tarp_out" \
  "${calibration_args[@]}" \
  --samples-per-object 256 \
  --num-bootstrap 200 \
  --seed 260807

test -e "$CORRECTION/DONE"
test -s "$CORRECTION/importance_summary.json"
test -s "$CORRECTION/redshift_weighted_summary.json"
printf '{"status":"complete","domain":"%s","posterior_samples":%s,"objects":%s}\n' \
  "$DOMAIN" "$BUDGET" "$LIMIT" > "$OUT/DONE"
echo "[posthoc-iw] complete -> $OUT"
