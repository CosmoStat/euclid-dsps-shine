#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
source "${SMOKE_ENV:-outputs/logs/feniks_adaptive_smcwake_smoke_latest.env}"

CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_adaptive_smcwake_parentprior_selection_r25.yaml}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
BIG_RUN_TAG="${BIG_RUN_TAG:-feniks_adaptive_smcwake_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/$BIG_RUN_TAG}"
TRAIN_ROOT="$RUN_ROOT/train"
MANIFEST_ROOT="$RUN_ROOT/manifests"

cd "$REPO_DIR"
mkdir -p outputs/logs
test -e "$SMOKE_TRAIN_ROOT/DONE" || {
  echo "[feniks-asmc-big][error] smoke is incomplete: $SMOKE_TRAIN_ROOT" >&2; exit 2;
}
python scripts/validate_feniks_adaptive_smc_training.py \
  --train "$SMOKE_TRAIN_ROOT" --expect-smoke
for path in "$CONFIG" "$CATALOG_DIR/train.parquet" "$CATALOG_DIR/test.parquet"; do
  test -s "$path" || { echo "[feniks-asmc-big][error] missing: $path" >&2; exit 2; }
done
test ! -e "$RUN_ROOT" || {
  echo "[feniks-asmc-big][error] output exists: $RUN_ROOT" >&2; exit 2;
}

JAX_PLATFORMS=cpu python scripts/build_feniks_parentprior_r25_manifests.py \
  --train-catalog "$CATALOG_DIR/train.parquet" \
  --test-catalog "$CATALOG_DIR/test.parquet" \
  --out "$MANIFEST_ROOT" --validation-fraction 0.10 \
  --n-exact 32 --seed 260821

python scripts/estimate_feniks_adaptive_smc_cost.py \
  --config "$CONFIG" --manifest "$MANIFEST_ROOT/manifest.json" \
  | tee "$RUN_ROOT/cost_estimate.json"

export REPO_DIR MINICONDA_PATH CONDA_ENV CONFIG CATALOG_DIR
train_raw=$(sbatch --parsable \
  --export="ALL,SMOKE=0,TRAIN_ROOT=$TRAIN_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT" \
  scripts/feniks_adaptive_smcwake_h100.slurm)
TRAIN_JOB="${train_raw%%;*}"

latest=outputs/logs/feniks_adaptive_smcwake_latest.env
printf 'export TRAIN_JOB=%q\nexport RUN_ROOT=%q\nexport TRAIN_ROOT=%q\nexport MANIFEST_ROOT=%q\nexport CONFIG=%q\nexport CATALOG_DIR=%q\nexport VALIDATED_SMOKE_ROOT=%q\n' \
  "$TRAIN_JOB" "$RUN_ROOT" "$TRAIN_ROOT" "$MANIFEST_ROOT" "$CONFIG" \
  "$CATALOG_DIR" "$SMOKE_ROOT" > "$latest"

echo "train_job=$TRAIN_JOB"
echo "run_root=$RUN_ROOT"
echo "validated_smoke_root=$SMOKE_ROOT"
echo "monitor: squeue -j $TRAIN_JOB"
echo "latest_env=$latest"
