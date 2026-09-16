#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
BATCH_SIZES="${BATCH_SIZES:-1:2:4:8}"
CHAINS="${CHAINS:-4}"
NUTS_WARMUP="${NUTS_WARMUP:-10}"
NUTS_DRAWS="${NUTS_DRAWS:-10}"
NUTS_MAX_DOUBLINGS="${NUTS_MAX_DOUBLINGS:-4}"
PROBE_TIME="${PROBE_TIME:-01:00:00}"
CONCURRENCY="${CONCURRENCY:-4}"
SEED="${SEED:-260730}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_nuts_multigalaxy_capacity_${STAMP}}"

cd "$REPO_DIR"
mkdir -p outputs/logs
for path in "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" \
  "$FEATURE_STATS"; do
  test -s "$path" || {
    echo "[nuts-multigalaxy-submit][error] missing $path"
    exit 2
  }
done
[[ "$BATCH_SIZES" =~ ^[1-9][0-9]*(:[1-9][0-9]*)*$ ]] || {
  echo "[nuts-multigalaxy-submit][error] invalid BATCH_SIZES=$BATCH_SIZES"
  exit 2
}
IFS=: read -r -a batch_sizes <<< "$BATCH_SIZES"
max_galaxies=0
for value in "${batch_sizes[@]}"; do
  (( value > max_galaxies )) && max_galaxies="$value"
done
test "$CONCURRENCY" -ge 1 || {
  echo "[nuts-multigalaxy-submit][error] CONCURRENCY must be positive"
  exit 2
}
test ! -e "$ROOT_DIR" || {
  echo "[nuts-multigalaxy-submit][error] output already exists: $ROOT_DIR"
  exit 2
}

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
JAX_PLATFORMS=cpu python scripts/benchmark_feniks_nuts_multigalaxy.py \
  prepare-cohort \
  --dataset "$DATASET" \
  --out "$ROOT_DIR" \
  --max-galaxies "$max_galaxies" \
  --seed "$SEED"

last_index=$((${#batch_sizes[@]} - 1))
common_export="ALL,ROOT_DIR=$ROOT_DIR,CONFIG=$CONFIG,DATASET=$DATASET,MODEL_ROOT=$MODEL_ROOT,CHECKPOINT=$CHECKPOINT,FEATURE_STATS=$FEATURE_STATS,BATCH_SIZES=$BATCH_SIZES,CHAINS=$CHAINS,NUTS_WARMUP=$NUTS_WARMUP,NUTS_DRAWS=$NUTS_DRAWS,NUTS_MAX_DOUBLINGS=$NUTS_MAX_DOUBLINGS,SEED=$SEED"
probe=$(sbatch --parsable \
  --array="0-${last_index}%${CONCURRENCY}" \
  --time="$PROBE_TIME" \
  --export="$common_export" \
  scripts/feniks_nuts_multigalaxy_capacity_h100.slurm)
probe="${probe%%;*}"
aggregate=$(sbatch --parsable \
  --dependency="afterany:$probe" \
  --export="$common_export" \
  scripts/feniks_nuts_multigalaxy_capacity_aggregate.slurm)
aggregate="${aggregate%%;*}"

log="outputs/logs/submit_feniks_nuts_multigalaxy_capacity_${STAMP}.log"
{
  printf 'root=%q\n' "$ROOT_DIR"
  printf 'batch_sizes=%q chains=%q warmup=%q draws=%q max_doublings=%q\n' \
    "$BATCH_SIZES" "$CHAINS" "$NUTS_WARMUP" "$NUTS_DRAWS" \
    "$NUTS_MAX_DOUBLINGS"
  printf 'probe=%q aggregate=%q\n' "$probe" "$aggregate"
} | tee "$log"

probe_hours=$(python - "$PROBE_TIME" "${#batch_sizes[@]}" <<'PY'
import sys

hours, minutes, seconds = map(int, sys.argv[1].split(":"))
print(f"{(hours + minutes / 60 + seconds / 3600) * int(sys.argv[2]):.2f}")
PY
)
echo "monitor: squeue -j $probe,$aggregate"
echo "requested_upper_bound_h100_hours=$probe_hours peak_h100=$CONCURRENCY"
echo "submission_log=$log"
