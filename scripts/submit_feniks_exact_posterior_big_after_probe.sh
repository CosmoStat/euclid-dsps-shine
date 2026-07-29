#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PROBE_JOB_ID="${PROBE_JOB_ID:?Set PROBE_JOB_ID to the batched probe job ID}"
PREPARED_ROOT="${PREPARED_ROOT:-outputs/runs/feniks_exact_posterior_two_galaxy_nuts_20260728_211755}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_exact_posterior_two_galaxy_nuts_big_${STAMP}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
NUTS_WARMUP="${NUTS_WARMUP:-200}"
NUTS_MAX_DOUBLINGS="${NUTS_MAX_DOUBLINGS:-4}"
SAMPLE_CHUNKS="${SAMPLE_CHUNKS:-100:100:100:100:100:100:100:100:100:100}"
NUTS_TIME="${NUTS_TIME:-20:00:00}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
mkdir -p outputs/logs
[[ "$PROBE_JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "[two-galaxy-big-after-probe][error] invalid PROBE_JOB_ID=$PROBE_JOB_ID"
  exit 2
}
for path in "$PREPARED_ROOT/cohort.parquet" "$PREPARED_ROOT/contract.json" \
  "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" "$FEATURE_STATS"; do
  test -s "$path" || {
    echo "[two-galaxy-big-after-probe][error] missing $path"
    exit 2
  }
done
test ! -e "$ROOT_DIR" || {
  echo "[two-galaxy-big-after-probe][error] output already exists: $ROOT_DIR"
  exit 2
}

common_export="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,PREPARED_ROOT=$PREPARED_ROOT,ROOT_DIR=$ROOT_DIR,STAMP=$STAMP,CONFIG=$CONFIG,DATASET=$DATASET,MODEL_ROOT=$MODEL_ROOT,CHECKPOINT=$CHECKPOINT,FEATURE_STATS=$FEATURE_STATS,NUTS_WARMUP=$NUTS_WARMUP,NUTS_MAX_DOUBLINGS=$NUTS_MAX_DOUBLINGS,SAMPLE_CHUNKS=$SAMPLE_CHUNKS,NUTS_TIME=$NUTS_TIME"
gate=$(sbatch --parsable \
  --dependency="afterok:$PROBE_JOB_ID" \
  --export="$common_export" \
  scripts/feniks_exact_two_galaxy_big_gate.slurm)
gate="${gate%%;*}"

log="outputs/logs/submit_feniks_exact_big_after_probe_${STAMP}.log"
{
  printf 'probe=%q gate=%q\n' "$PROBE_JOB_ID" "$gate"
  printf 'prepared_root=%q\nroot=%q\n' "$PREPARED_ROOT" "$ROOT_DIR"
  printf 'nuts_warmup=%q nuts_max_doublings=%q sample_chunks=%q nuts_time=%q\n' \
    "$NUTS_WARMUP" "$NUTS_MAX_DOUBLINGS" "$SAMPLE_CHUNKS" "$NUTS_TIME"
} | tee "$log"
echo "monitor_gate: squeue -j $PROBE_JOB_ID,$gate"
echo "requested_upper_bound_h100_hours_after_gate=44.33 peak_h100=2"
echo "submission_log=$log"
