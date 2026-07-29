#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ROOT_DIR="${ROOT_DIR:?Set ROOT_DIR to the prepared two-galaxy NUTS run}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
NUTS_WARMUP="${NUTS_WARMUP:-50}"
NUTS_MAX_DOUBLINGS="${NUTS_MAX_DOUBLINGS:-4}"
SAMPLE_CHUNKS="${SAMPLE_CHUNKS:-100}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "$REPO_DIR"
mkdir -p outputs/logs
for path in "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" \
  "$FEATURE_STATS" "$ROOT_DIR/cohort.parquet" "$ROOT_DIR/contract.json"; do
  test -s "$path" || { echo "[nuts-probe][error] missing $path"; exit 2; }
done
galaxy_dir=$(python - "$ROOT_DIR" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
row = pd.read_parquet(root / "cohort.parquet").iloc[0]
print(
    root
    / "galaxies"
    / f"{int(row['order']):02d}_{row['example_key']}_row{int(row['row_index'])}"
)
PY
)
test -f "$galaxy_dir/PREP_DONE" || {
  echo "[nuts-probe][error] incomplete preparation: $galaxy_dir"
  exit 2
}
chain_dir="$galaxy_dir/nuts/chain_00"
test ! -f "$chain_dir/DONE" || {
  echo "[nuts-probe][error] chain already complete: $chain_dir"
  exit 2
}
find "$chain_dir" -maxdepth 0 -type d -empty -delete 2>/dev/null || true
test ! -e "$chain_dir" || {
  echo "[nuts-probe][error] non-empty incomplete chain: $chain_dir"
  exit 2
}

common_export="ALL,ROOT_DIR=$ROOT_DIR,CONFIG=$CONFIG,DATASET=$DATASET,MODEL_ROOT=$MODEL_ROOT,CHECKPOINT=$CHECKPOINT,FEATURE_STATS=$FEATURE_STATS"
probe=$(sbatch --parsable --array=0 --time=04:00:00 \
  --export="$common_export,MODE=pilot,SAMPLER=nuts,CHAINS=4,N_GALAXIES=2,NUTS_WARMUP=$NUTS_WARMUP,NUTS_MAX_DOUBLINGS=$NUTS_MAX_DOUBLINGS,SAMPLE_CHUNKS=$SAMPLE_CHUNKS" \
  scripts/feniks_exact_chain_h100.slurm)
probe="${probe%%;*}"

log="outputs/logs/submit_feniks_exact_two_galaxy_nuts_probe_${STAMP}.log"
{
  printf 'root=%q\n' "$ROOT_DIR"
  printf 'nuts_warmup=%q nuts_max_doublings=%q sample_chunks=%q\n' \
    "$NUTS_WARMUP" "$NUTS_MAX_DOUBLINGS" "$SAMPLE_CHUNKS"
  printf 'probe=%q\n' "$probe"
} | tee "$log"
echo "monitor: squeue -j $probe"
echo "requested_upper_bound_h100_hours=4.00 peak_h100=1"
echo "submission_log=$log"
