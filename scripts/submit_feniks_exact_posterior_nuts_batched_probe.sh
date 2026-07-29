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
for path in "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" \
  "$FEATURE_STATS" "$ROOT_DIR/cohort.parquet" "$ROOT_DIR/contract.json"; do
  test -s "$path" || { echo "[batched-nuts-probe][error] missing $path"; exit 2; }
done
galaxy_dir="$ROOT_DIR/galaxies/01_typical_row1358"
test -f "$galaxy_dir/nuts/chain_00/DONE" || {
  echo "[batched-nuts-probe][error] scalar baseline is incomplete"
  exit 2
}
resume_state="$galaxy_dir/nuts/.chain_01-chain_02-chain_03.batched_nuts_state.pkl"
legacy_resumable=1
for chain in 01 02 03; do
  chain_dir="$galaxy_dir/nuts/chain_$chain"
  [[ -f "$chain_dir/sampling_state.pkl" \
    && -f "$chain_dir/tuned_parameters.npz" ]] || legacy_resumable=0
done
complete=0
for chain in 01 02 03; do
  chain_dir="$galaxy_dir/nuts/chain_$chain"
  [[ -f "$chain_dir/DONE" ]] && complete=$(( complete + 1 ))
  find "$chain_dir" -maxdepth 0 -type d -empty -delete 2>/dev/null || true
  [[ ! -d "$chain_dir" || -f "$chain_dir/DONE" || -f "$resume_state" \
    || "$legacy_resumable" -eq 1 ]] || {
    echo "[batched-nuts-probe][error] non-resumable chain: $chain_dir"
    exit 2
  }
done

common_export="ALL,ROOT_DIR=$ROOT_DIR,CONFIG=$CONFIG,DATASET=$DATASET,MODEL_ROOT=$MODEL_ROOT,CHECKPOINT=$CHECKPOINT,FEATURE_STATS=$FEATURE_STATS"
probe=""
if (( complete < 3 )); then
  probe=$(sbatch --parsable --array=0 --time=04:00:00 \
    --export="$common_export,MODE=pilot,CHAIN_INDICES=1:2:3,NUTS_WARMUP=50,NUTS_MAX_DOUBLINGS=4,SAMPLE_CHUNKS=100" \
    scripts/feniks_exact_nuts_batched_h100.slurm)
  probe="${probe%%;*}"
fi

log="outputs/logs/submit_feniks_exact_nuts_batched_probe_${STAMP}.log"
{
  printf 'root=%q\nprobe=%q\n' "$ROOT_DIR" "$probe"
  printf 'galaxy_index=0 chain_indices=%q nuts_warmup=50 nuts_max_doublings=4 sample_chunks=100\n' \
    "1:2:3"
} | tee "$log"
echo "monitor: squeue -j $probe"
echo "requested_upper_bound_h100_hours=4.00 peak_h100=1"
(( complete == 3 )) && echo "probe already complete; run the summarizer now"
echo "submission_log=$log"
