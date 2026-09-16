#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
ROOT_DIR="${ROOT_DIR:?Set ROOT_DIR to the existing smoke output}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "$REPO_DIR"
mkdir -p outputs/logs
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
test -f "$ROOT_DIR/COHORT_DONE" || {
  echo "[exact-smoke-recovery][error] missing cohort: $ROOT_DIR"
  exit 2
}
prep_count=$(find "$ROOT_DIR/galaxies" -mindepth 2 -maxdepth 2 -name PREP_DONE | wc -l)
test "$prep_count" -eq 2 || {
  echo "[exact-smoke-recovery][error] expected two PREP_DONE, found $prep_count"
  exit 2
}
test ! -f "$ROOT_DIR/DONE" || {
  echo "[exact-smoke-recovery][error] smoke is already complete"
  exit 2
}

common_export="ALL,ROOT_DIR=$ROOT_DIR,CONFIG=$CONFIG,DATASET=$DATASET,MODEL_ROOT=$MODEL_ROOT,CHECKPOINT=$CHECKPOINT,FEATURE_STATS=$FEATURE_STATS"
nuts=$(sbatch --parsable --array=0-7%8 --time=00:30:00 \
  --export="$common_export,MODE=smoke,SAMPLER=nuts,CHAINS=4,N_GALAXIES=2" \
  scripts/feniks_exact_chain_h100.slurm)
nuts="${nuts%%;*}"
mclmc=$(sbatch --parsable --array=0-3%4 --time=00:30:00 \
  --export="$common_export,MODE=smoke,SAMPLER=mclmc,CHAINS=2,N_GALAXIES=2" \
  scripts/feniks_exact_chain_h100.slurm)
mclmc="${mclmc%%;*}"
finalize=$(sbatch --parsable --array=0-1%2 --time=00:30:00 \
  --dependency="afterok:$nuts:$mclmc" \
  --export="$common_export,MODE=smoke" \
  scripts/feniks_exact_finalize_h100.slurm)
finalize="${finalize%%;*}"
aggregate=$(sbatch --parsable --time=00:20:00 \
  --dependency="afterok:$finalize" \
  --export="$common_export" scripts/feniks_exact_aggregate_h100.slurm)
aggregate="${aggregate%%;*}"

log="outputs/logs/submit_feniks_exact_smoke_recovery_${STAMP}.log"
{
  printf 'root=%q\n' "$ROOT_DIR"
  printf 'nuts=%q mclmc=%q finalize=%q aggregate=%q\n' \
    "$nuts" "$mclmc" "$finalize" "$aggregate"
} | tee "$log"
echo "monitor: squeue -j $nuts,$mclmc,$finalize,$aggregate"
echo "submission_log=$log"
