#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ROOT_DIR="${ROOT_DIR:?Set ROOT_DIR to the prepared two-galaxy NUTS run}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
NUTS_WARMUP="${NUTS_WARMUP:-200}"
NUTS_MAX_DOUBLINGS="${NUTS_MAX_DOUBLINGS:-6}"
SAMPLE_CHUNKS="${SAMPLE_CHUNKS:-100,300}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "$REPO_DIR"
mkdir -p outputs/logs
for path in "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" \
  "$FEATURE_STATS" "$ROOT_DIR/cohort.parquet" "$ROOT_DIR/contract.json"; do
  test -s "$path" || { echo "[nuts-recovery][error] missing $path"; exit 2; }
done
for galaxy_index in 0 1; do
  galaxy_dir=$(python - "$ROOT_DIR" "$galaxy_index" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
row = pd.read_parquet(root / "cohort.parquet").iloc[int(sys.argv[2])]
print(
    root
    / "galaxies"
    / f"{int(row['order']):02d}_{row['example_key']}_row{int(row['row_index'])}"
)
PY
)
  test -f "$galaxy_dir/PREP_DONE" || {
    echo "[nuts-recovery][error] incomplete preparation: $galaxy_dir"
    exit 2
  }
  completed=$(find "$galaxy_dir/nuts" -mindepth 2 -maxdepth 2 \
    -name DONE 2>/dev/null | wc -l)
  (( completed == 0 )) || {
    echo "[nuts-recovery][error] refusing to mix settings with completed chains: $galaxy_dir"
    exit 2
  }
  find "$galaxy_dir/nuts" -mindepth 1 -maxdepth 1 \
    -type d -name 'chain_*' -empty -delete 2>/dev/null || true
done

common_export="ALL,ROOT_DIR=$ROOT_DIR,CONFIG=$CONFIG,DATASET=$DATASET,MODEL_ROOT=$MODEL_ROOT,CHECKPOINT=$CHECKPOINT,FEATURE_STATS=$FEATURE_STATS"
nuts=$(sbatch --parsable --array=0-7%8 --time=06:00:00 \
  --export="$common_export,MODE=pilot,SAMPLER=nuts,CHAINS=4,N_GALAXIES=2,NUTS_WARMUP=$NUTS_WARMUP,NUTS_MAX_DOUBLINGS=$NUTS_MAX_DOUBLINGS,SAMPLE_CHUNKS=$SAMPLE_CHUNKS" \
  scripts/feniks_exact_chain_h100.slurm)
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

log="outputs/logs/submit_feniks_exact_two_galaxy_nuts_recovery_${STAMP}.log"
{
  printf 'root=%q\n' "$ROOT_DIR"
  printf 'nuts_warmup=%q nuts_max_doublings=%q sample_chunks=%q\n' \
    "$NUTS_WARMUP" "$NUTS_MAX_DOUBLINGS" "$SAMPLE_CHUNKS"
  printf 'nuts=%q finalize=%q aggregate=%q\n' \
    "$nuts" "$finalize" "$aggregate"
} | tee "$log"
echo "monitor: squeue -j $nuts,$finalize,$aggregate"
echo "requested_upper_bound_h100_hours=52.33 peak_h100=8"
echo "submission_log=$log"
