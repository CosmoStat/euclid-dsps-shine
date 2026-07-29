#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ROOT_DIR="${ROOT_DIR:?Set ROOT_DIR to the prepared two-galaxy NUTS run}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "$REPO_DIR"
mkdir -p outputs/logs
first_galaxy="$ROOT_DIR/galaxies/01_typical_row1358"
for chain in 00 01 02 03; do
  test -f "$first_galaxy/nuts/chain_$chain/DONE" || {
    echo "[batched-nuts-finish][error] first galaxy chain $chain incomplete"
    exit 2
  }
done
second_galaxy="$ROOT_DIR/galaxies/02_nearby_row400"
for chain in 00 01 02 03; do
  chain_dir="$second_galaxy/nuts/chain_$chain"
  find "$chain_dir" -maxdepth 0 -type d -empty -delete 2>/dev/null || true
  test ! -e "$chain_dir" || {
    echo "[batched-nuts-finish][error] non-empty chain: $chain_dir"
    exit 2
  }
done

common_export="ALL,ROOT_DIR=$ROOT_DIR,CONFIG=$CONFIG,DATASET=$DATASET,MODEL_ROOT=$MODEL_ROOT,CHECKPOINT=$CHECKPOINT,FEATURE_STATS=$FEATURE_STATS"
nuts=$(sbatch --parsable --array=1 --time=04:00:00 \
  --export="$common_export,MODE=pilot,NUTS_WARMUP=50,NUTS_MAX_DOUBLINGS=4,SAMPLE_CHUNKS=100" \
  scripts/feniks_exact_nuts_batched_h100.slurm)
nuts="${nuts%%;*}"
finalize=$(sbatch --parsable --array=0-1%2 --time=02:00:00 \
  --dependency="afterok:$nuts" \
  --export="$common_export,MODE=pilot,FINAL_SAMPLERS=nuts" \
  scripts/feniks_exact_finalize_h100.slurm)
finalize="${finalize%%;*}"
aggregate=$(sbatch --parsable --time=00:20:00 \
  --dependency="afterok:$finalize" \
  --export="$common_export,FINAL_SAMPLERS=nuts" \
  scripts/feniks_exact_aggregate_h100.slurm)
aggregate="${aggregate%%;*}"

log="outputs/logs/submit_feniks_exact_nuts_batched_finish_${STAMP}.log"
{
  printf 'root=%q\nnuts=%q finalize=%q aggregate=%q\n' \
    "$ROOT_DIR" "$nuts" "$finalize" "$aggregate"
} | tee "$log"
echo "monitor: squeue -j $nuts,$finalize,$aggregate"
echo "requested_upper_bound_h100_hours=8.33 peak_h100=2"
echo "submission_log=$log"
