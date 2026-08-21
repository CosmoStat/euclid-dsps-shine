#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
source "${TRAIN_ENV:-outputs/logs/feniks_parentprior_sleepnpe_latest.env}"

DATASET="${DATASET:-$CATALOG_DIR/test.parquet}"
CHECKPOINT="${CHECKPOINT:-$TRAIN_ROOT/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$TRAIN_ROOT/feature_stats.json}"
EXACT_ROOT="${EXACT_ROOT:-$RUN_ROOT/exact_posterior_32}"
COHORT_FILE="$MANIFEST_ROOT/exact_cohort.csv"
N_GALAXIES="${N_GALAXIES:-32}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

cd "$REPO_DIR"
mkdir -p outputs/logs
test -e "$TRAIN_ROOT/DONE" || {
  echo "[feniks-parent-exact-submit][error] training is incomplete" >&2; exit 2;
}
for path in "$TRAIN_ROOT/parentprior_training_validation.json" "$CONFIG" \
  "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" "$FEATURE_STATS" "$COHORT_FILE"; do
  test -s "$path" || { echo "[feniks-parent-exact-submit][error] missing: $path" >&2; exit 2; }
done
python - "$TRAIN_ROOT/parentprior_training_validation.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if payload.get("status") != "PASS":
    raise SystemExit("training validation did not pass")
PY
test ! -e "$EXACT_ROOT" || {
  echo "[feniks-parent-exact-submit][error] output exists: $EXACT_ROOT" >&2; exit 2;
}

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
JAX_PLATFORMS=cpu python scripts/run_feniks_exact_posterior_benchmark.py \
  prepare-cohort --config "$CONFIG" --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" --feature-stats "$FEATURE_STATS" \
  --out "$EXACT_ROOT" --mode full --cohort-file "$COHORT_FILE"

actual=$(python - "$EXACT_ROOT/cohort.csv" <<'PY'
import pandas as pd, sys
print(len(pd.read_csv(sys.argv[1])))
PY
)
[[ "$actual" == "$N_GALAXIES" ]] || {
  echo "[feniks-parent-exact-submit][error] cohort has $actual rows, expected $N_GALAXIES" >&2
  exit 2
}

export REPO_DIR MINICONDA_PATH CONDA_ENV CONFIG DATASET CHECKPOINT FEATURE_STATS
exact_raw=$(sbatch --parsable \
  --array="0-$((N_GALAXIES - 1))%${MAX_CONCURRENT}" \
  --export="ALL,EXACT_ROOT=$EXACT_ROOT,N_GALAXIES=$N_GALAXIES" \
  scripts/feniks_parentprior_exact_h100.slurm)
EXACT_JOB="${exact_raw%%;*}"
final_raw=$(sbatch --parsable --dependency="afterok:${EXACT_JOB}" \
  --export="ALL,EXACT_ROOT=$EXACT_ROOT" \
  scripts/feniks_parentprior_exact_finalize.slurm)
EXACT_FINALIZER_JOB="${final_raw%%;*}"

latest=outputs/logs/feniks_parentprior_exact_latest.env
printf 'export EXACT_JOB=%q\nexport EXACT_FINALIZER_JOB=%q\nexport EXACT_ROOT=%q\nexport N_GALAXIES=%q\nexport CONFIG=%q\nexport DATASET=%q\nexport CHECKPOINT=%q\nexport FEATURE_STATS=%q\n' \
  "$EXACT_JOB" "$EXACT_FINALIZER_JOB" "$EXACT_ROOT" "$N_GALAXIES" \
  "$CONFIG" "$DATASET" "$CHECKPOINT" "$FEATURE_STATS" > "$latest"

echo "exact_job=$EXACT_JOB"
echo "exact_finalizer_job=$EXACT_FINALIZER_JOB"
echo "exact_root=$EXACT_ROOT"
echo "array=0-$((N_GALAXIES - 1))%${MAX_CONCURRENT} one_galaxy_per_H100"
echo "each_task=encoder+raw_IS+defensive_IS+MAP+4_chain_batched_NUTS+diagnostics"
echo "latest_env=$latest"
