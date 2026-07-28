#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
SMOKE_ROOT="${SMOKE_ROOT:?Set SMOKE_ROOT to the completed exact-posterior smoke}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
PROBE_LABEL="${PROBE_LABEL:-mclmc_adaptation_probe_t10}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "$REPO_DIR"
mkdir -p outputs/logs
test -f "$SMOKE_ROOT/DONE" || {
  echo "[mclmc-probe][error] incomplete smoke: $SMOKE_ROOT"
  exit 2
}
for path in "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" \
  "$FEATURE_STATS"; do
  test -s "$path" || { echo "[mclmc-probe][error] missing $path"; exit 2; }
done

first_galaxy=$(find "$SMOKE_ROOT/galaxies" -mindepth 1 -maxdepth 1 \
  -type d -printf '%f\n' | sort | head -1)
test -n "$first_galaxy" || {
  echo "[mclmc-probe][error] no prepared galaxy in $SMOKE_ROOT"
  exit 2
}
probe_dir="$SMOKE_ROOT/galaxies/$first_galaxy/$PROBE_LABEL/chain_00"
test ! -e "$probe_dir" || {
  echo "[mclmc-probe][error] probe output already exists: $probe_dir"
  exit 2
}

common_export="ALL,ROOT_DIR=$SMOKE_ROOT,CONFIG=$CONFIG,DATASET=$DATASET,MODEL_ROOT=$MODEL_ROOT,CHECKPOINT=$CHECKPOINT,FEATURE_STATS=$FEATURE_STATS"
probe=$(sbatch --parsable --array=0 --time=01:00:00 \
  --export="$common_export,MODE=pilot,SAMPLER=mclmc,SAMPLER_LABEL=$PROBE_LABEL,CHAINS=1,N_GALAXIES=1,MCLMC_TUNE=10,SAMPLE_CHUNKS=10,THINNING=1" \
  scripts/feniks_exact_chain_h100.slurm)
probe="${probe%%;*}"

log="outputs/logs/submit_feniks_mclmc_probe_${STAMP}.log"
{
  printf 'root=%q\nprobe_label=%q\nprobe_dir=%q\n' \
    "$SMOKE_ROOT" "$PROBE_LABEL" "$probe_dir"
  printf 'probe=%q\n' "$probe"
} | tee "$log"
echo "monitor: squeue -j $probe"
echo "requested_upper_bound_h100_hours=1.00 peak_h100=1"
echo "submission_log=$log"
