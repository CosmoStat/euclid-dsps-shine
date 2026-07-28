#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
SMOKE_ROOT="${SMOKE_ROOT:?Set SMOKE_ROOT to the completed exact-posterior smoke}"
PROBE_LABEL="${PROBE_LABEL:-mclmc_adaptation_probe_t10}"
PROBE_SUMMARY="${PROBE_SUMMARY:-}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_exact_posterior_two_galaxy_pilot_${STAMP}}"
NUTS_WARMUP="${NUTS_WARMUP:-500}"
MCLMC_TUNE="${MCLMC_TUNE:-500}"
MCLMC_THINNING="${MCLMC_THINNING:-8}"

cd "$REPO_DIR"
mkdir -p outputs/logs
test -f "$SMOKE_ROOT/DONE" || {
  echo "[two-galaxy-pilot][error] incomplete smoke: $SMOKE_ROOT"
  exit 2
}
test -s "$SMOKE_ROOT/cohort.csv" || {
  echo "[two-galaxy-pilot][error] missing smoke cohort"
  exit 2
}
if [[ -z "$PROBE_SUMMARY" ]]; then
  first_galaxy=$(find "$SMOKE_ROOT/galaxies" -mindepth 1 -maxdepth 1 \
    -type d -printf '%f\n' | sort | head -1)
  PROBE_SUMMARY="$SMOKE_ROOT/galaxies/$first_galaxy/$PROBE_LABEL/probe_summary.json"
fi
python - "$PROBE_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "passed":
    raise SystemExit(f"MCLMC adaptation probe did not pass: {path}")
print(f"[two-galaxy-pilot] accepted adaptation probe: {path}")
PY
for path in "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" \
  "$FEATURE_STATS"; do
  test -s "$path" || { echo "[two-galaxy-pilot][error] missing $path"; exit 2; }
done
test ! -e "$ROOT_DIR" || {
  echo "[two-galaxy-pilot][error] output already exists: $ROOT_DIR"
  exit 2
}

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
JAX_PLATFORMS=cpu python scripts/run_feniks_exact_posterior_benchmark.py \
  prepare-cohort --config "$CONFIG" --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" --feature-stats "$FEATURE_STATS" \
  --cohort-file "$SMOKE_ROOT/cohort.csv" \
  --out "$ROOT_DIR" --mode pilot

common_export="ALL,ROOT_DIR=$ROOT_DIR,CONFIG=$CONFIG,DATASET=$DATASET,MODEL_ROOT=$MODEL_ROOT,CHECKPOINT=$CHECKPOINT,FEATURE_STATS=$FEATURE_STATS"
prep=$(sbatch --parsable --array=0-1%2 --time=04:00:00 \
  --export="$common_export,MODE=pilot,N_GALAXIES=2" \
  scripts/feniks_exact_prepare_h100.slurm)
prep="${prep%%;*}"
nuts=$(sbatch --parsable --array=0-7%8 --time=06:00:00 \
  --dependency="afterok:$prep" \
  --export="$common_export,MODE=pilot,SAMPLER=nuts,CHAINS=4,N_GALAXIES=2,NUTS_WARMUP=$NUTS_WARMUP" \
  scripts/feniks_exact_chain_h100.slurm)
nuts="${nuts%%;*}"
mclmc=$(sbatch --parsable --array=0-7%8 --time=06:00:00 \
  --dependency="afterok:$prep" \
  --export="$common_export,MODE=pilot,SAMPLER=mclmc,CHAINS=4,N_GALAXIES=2,MCLMC_TUNE=$MCLMC_TUNE,THINNING=$MCLMC_THINNING" \
  scripts/feniks_exact_chain_h100.slurm)
mclmc="${mclmc%%;*}"
finalize=$(sbatch --parsable --array=0-1%2 --time=02:00:00 \
  --dependency="afterok:$nuts:$mclmc" \
  --export="$common_export,MODE=pilot" \
  scripts/feniks_exact_finalize_h100.slurm)
finalize="${finalize%%;*}"
aggregate=$(sbatch --parsable --time=00:20:00 \
  --dependency="afterok:$finalize" \
  --export="$common_export" scripts/feniks_exact_aggregate_h100.slurm)
aggregate="${aggregate%%;*}"

log="outputs/logs/submit_feniks_exact_two_galaxy_pilot_${STAMP}.log"
{
  printf 'root=%q\nsmoke_root=%q\nprobe_summary=%q\n' \
    "$ROOT_DIR" "$SMOKE_ROOT" "$PROBE_SUMMARY"
  printf 'nuts_warmup=%q mclmc_tune=%q mclmc_thinning=%q sample_chunks=%q\n' \
    "$NUTS_WARMUP" "$MCLMC_TUNE" "$MCLMC_THINNING" "100,500"
  printf 'prep=%q nuts=%q mclmc=%q finalize=%q aggregate=%q\n' \
    "$prep" "$nuts" "$mclmc" "$finalize" "$aggregate"
} | tee "$log"
echo "monitor: squeue -j $prep,$nuts,$mclmc,$finalize,$aggregate"
echo "requested_upper_bound_h100_hours=108.33 peak_h100=16"
echo "submission_log=$log"
