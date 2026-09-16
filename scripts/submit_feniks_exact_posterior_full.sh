#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_exact_posterior_v1}"
SMOKE_ROOT="${SMOKE_ROOT:?Set SMOKE_ROOT to the completed smoke output}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "$REPO_DIR"
mkdir -p outputs/logs
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
test -f "$SMOKE_ROOT/DONE" || {
  echo "[exact-full][error] smoke is not complete: $SMOKE_ROOT/DONE"
  exit 2
}
for path in "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" "$FEATURE_STATS"; do
  test -s "$path" || { echo "[exact-full][error] missing $path"; exit 2; }
done
test ! -e "$ROOT_DIR" || {
  echo "[exact-full][error] output already exists: $ROOT_DIR"
  exit 2
}

JAX_PLATFORMS=cpu python scripts/run_feniks_exact_posterior_benchmark.py \
  prepare-cohort --config "$CONFIG" --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" --feature-stats "$FEATURE_STATS" \
  --out "$ROOT_DIR" --mode full

selection="$ROOT_DIR/pilot/selected_mclmc.json"
common_export="ALL,ROOT_DIR=$ROOT_DIR,CONFIG=$CONFIG,DATASET=$DATASET,MODEL_ROOT=$MODEL_ROOT,CHECKPOINT=$CHECKPOINT,FEATURE_STATS=$FEATURE_STATS"

pilot_prep=$(sbatch --parsable --array=0-1%2 \
  --export="$common_export,MODE=pilot,N_GALAXIES=2" \
  scripts/feniks_exact_prepare_h100.slurm)
pilot_prep="${pilot_prep%%;*}"
pilot_nuts=$(sbatch --parsable --array=0-7%8 --time=06:00:00 \
  --dependency="afterok:$pilot_prep" \
  --export="$common_export,MODE=pilot,SAMPLER=nuts,CHAINS=4,N_GALAXIES=2" \
  scripts/feniks_exact_chain_h100.slurm)
pilot_nuts="${pilot_nuts%%;*}"
pilot_grid=$(sbatch --parsable --array=0-6%7 \
  --dependency="afterok:$pilot_prep" \
  --export="$common_export" scripts/feniks_exact_pilot_h100.slurm)
pilot_grid="${pilot_grid%%;*}"
pilot_select=$(sbatch --parsable \
  --dependency="afterok:$pilot_nuts:$pilot_grid" \
  --export="$common_export,ACTION=select,SELECTION=$selection" \
  scripts/feniks_exact_pilot_select_h100.slurm)
pilot_select="${pilot_select%%;*}"
pilot_validate=$(sbatch --parsable --array=0-1%2 --time=06:00:00 \
  --dependency="afterok:$pilot_select" \
  --export="$common_export,MODE=pilot,SAMPLER=mclmc,CHAINS=2,N_GALAXIES=1,GALAXY_OFFSET=1,PILOT_SELECTION=$selection" \
  scripts/feniks_exact_chain_h100.slurm)
pilot_validate="${pilot_validate%%;*}"
pilot_gate=$(sbatch --parsable \
  --dependency="afterok:$pilot_validate" \
  --export="$common_export,ACTION=validate,SELECTION=$selection" \
  scripts/feniks_exact_pilot_select_h100.slurm)
pilot_gate="${pilot_gate%%;*}"

full_prep=$(sbatch --parsable --array=0-6%7 \
  --dependency="afterok:$pilot_gate" \
  --export="$common_export,MODE=full,N_GALAXIES=7" \
  scripts/feniks_exact_prepare_h100.slurm)
full_prep="${full_prep%%;*}"
full_nuts=$(sbatch --parsable --array=0-27%8 \
  --dependency="afterok:$full_prep" \
  --export="$common_export,MODE=full,SAMPLER=nuts,CHAINS=4,N_GALAXIES=7" \
  scripts/feniks_exact_chain_h100.slurm)
full_nuts="${full_nuts%%;*}"
full_mclmc=$(sbatch --parsable --array=0-27%8 \
  --dependency="afterok:$full_prep" \
  --export="$common_export,MODE=full,SAMPLER=mclmc,CHAINS=4,N_GALAXIES=7,PILOT_SELECTION=$selection" \
  scripts/feniks_exact_chain_h100.slurm)
full_mclmc="${full_mclmc%%;*}"
finalize=$(sbatch --parsable --array=0-6%7 \
  --dependency="afterok:$full_nuts:$full_mclmc" \
  --export="$common_export,MODE=full" \
  scripts/feniks_exact_finalize_h100.slurm)
finalize="${finalize%%;*}"
aggregate=$(sbatch --parsable \
  --dependency="afterok:$finalize" \
  --export="$common_export" scripts/feniks_exact_aggregate_h100.slurm)
aggregate="${aggregate%%;*}"

log="outputs/logs/submit_feniks_exact_full_${STAMP}.log"
{
  printf 'root=%q\nselection=%q\n' "$ROOT_DIR" "$selection"
  printf 'pilot_prep=%q pilot_nuts=%q pilot_grid=%q pilot_select=%q pilot_validate=%q pilot_gate=%q\n' \
    "$pilot_prep" "$pilot_nuts" "$pilot_grid" "$pilot_select" "$pilot_validate" "$pilot_gate"
  printf 'full_prep=%q full_nuts=%q full_mclmc=%q finalize=%q aggregate=%q\n' \
    "$full_prep" "$full_nuts" "$full_mclmc" "$finalize" "$aggregate"
} | tee "$log"
echo "monitor: squeue -j $pilot_prep,$pilot_nuts,$pilot_grid,$pilot_select,$pilot_validate,$pilot_gate,$full_prep,$full_nuts,$full_mclmc,$finalize,$aggregate"
echo "requested_upper_bound_h100_hours=825.00 peak_h100=16"
echo "submission_log=$log"
