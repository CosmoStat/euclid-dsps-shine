#!/bin/bash
#SBATCH --job-name=popcosmos_chaincal
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536}"
STAGE="${STAGE:-full_cont120}"
RWS26_INFERENCE="${RWS26_INFERENCE:-$RUN_ROOT/bands26/$STAGE/inference}"
RWS24_INFERENCE="${RWS24_INFERENCE:-$RUN_ROOT/bands24_no_irac/$STAGE/inference}"
ASSET_ROOT="${ASSET_ROOT:-${SCRATCH:?Set SCRATCH or ASSET_ROOT}/popcosmos_v1p1_chains}"
POPCOSMOS_POSTERIOR="${POPCOSMOS_POSTERIOR:-$ASSET_ROOT/redshift_posterior_1395}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/popcosmos_chain_redshift_calibration_v1}"
EXPECTED_SPECZ="${EXPECTED_SPECZ:-1395}"
SAMPLES_PER_OBJECT="${SAMPLES_PER_OBJECT:-128}"
NUM_REGIONS="${NUM_REGIONS:-100}"
NUM_BOOTSTRAP="${NUM_BOOTSTRAP:-1000}"
SEED="${SEED:-260806}"
RUN_ALL_DRAW_SENSITIVITY="${RUN_ALL_DRAW_SENSITIVITY:-1}"

TRUTH="$POPCOSMOS_POSTERIOR/inference_truth.parquet"
PRIMARY="$POPCOSMOS_POSTERIOR/posterior_samples.parquet"
ALL_DRAWS="$POPCOSMOS_POSTERIOR/posterior_samples_all.parquet"
MIRA_OUT="$OUT_DIR/mira_128"
TARP_OUT="$OUT_DIR/tarp_128"
PIT_OUT="$OUT_DIR/pit_coverage_128"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export JAX_PLATFORMS=cuda
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID:-manual}-popcosmos-chaincal"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

for path in \
  "$ASSET_ROOT/PREPOST_COMPLETE.json" \
  "$POPCOSMOS_POSTERIOR/DONE" \
  "$TRUTH" \
  "$PRIMARY" \
  "$ALL_DRAWS" \
  "$RWS26_INFERENCE/inference_truth.parquet" \
  "$RWS24_INFERENCE/inference_truth.parquet"; do
  test -s "$path" || test -e "$path" || {
    echo "[popcosmos-chaincal][error] missing: $path"
    exit 2
  }
done
for source in "$RWS26_INFERENCE" "$RWS24_INFERENCE"; do
  compgen -G "$source/posterior_samples/*.parquet" >/dev/null || {
    echo "[popcosmos-chaincal][error] no posterior shards under: $source"
    exit 2
  }
done
if [[ -e "$OUT_DIR" ]]; then
  echo "[popcosmos-chaincal][error] output already exists: $OUT_DIR"
  exit 2
fi

export TRUTH RWS26_INFERENCE RWS24_INFERENCE EXPECTED_SPECZ
python - <<'PY'
import os
from pathlib import Path

import numpy as np
import pandas as pd

reference = pd.read_parquet(os.environ["TRUTH"])
if len(reference) != int(os.environ["EXPECTED_SPECZ"]):
    raise SystemExit("[popcosmos-chaincal][error] wrong Pop-COSMOS truth row count")
reference_ids = reference["object_id"].to_numpy(np.int64)
for variable in ("RWS26_INFERENCE", "RWS24_INFERENCE"):
    path = Path(os.environ[variable]) / "inference_truth.parquet"
    frame = pd.read_parquet(path)
    finite = np.isfinite(frame["redshift_true"].to_numpy(float))
    selected = frame.loc[finite].reset_index(drop=True)
    if not np.array_equal(selected["object_id"].to_numpy(np.int64), reference_ids):
        raise SystemExit(f"[popcosmos-chaincal][error] object order differs: {path}")
    if not np.array_equal(
        selected["redshift_true"].to_numpy(float),
        reference["redshift_true"].to_numpy(float),
    ):
        raise SystemExit(f"[popcosmos-chaincal][error] truth differs: {path}")
print(f"[popcosmos-chaincal] exact shared cohort: {len(reference)} objects")
PY

python - <<'PY'
import jax

print("[popcosmos-chaincal] backend:", jax.default_backend())
print("[popcosmos-chaincal] devices:", jax.devices())
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
    raise SystemExit("[popcosmos-chaincal][error] expected exactly one H100")
PY
nvidia-smi

mkdir -p "$OUT_DIR"

python scripts/evaluate_feniks_mira.py \
  --truth "$TRUTH" \
  --posterior "rws26=$RWS26_INFERENCE" \
  --posterior "rws24=$RWS24_INFERENCE" \
  --posterior "popcosmos=$PRIMARY" \
  --out "$MIRA_OUT" \
  --parameter z_obs \
  --truth-column z_obs=redshift_true \
  --drop-nonfinite-truth \
  --num-regions "$NUM_REGIONS" \
  --num-bootstrap "$NUM_BOOTSTRAP" \
  --samples-per-object "$SAMPLES_PER_OBJECT" \
  --seed "$SEED"

python scripts/evaluate_feniks_tarp.py \
  --truth "$TRUTH" \
  --posterior "rws26=$RWS26_INFERENCE" \
  --posterior "rws24=$RWS24_INFERENCE" \
  --posterior "popcosmos=$PRIMARY" \
  --out "$TARP_OUT" \
  --parameter z_obs \
  --truth-column z_obs=redshift_true \
  --drop-nonfinite-truth \
  --num-bootstrap "$NUM_BOOTSTRAP" \
  --samples-per-object "$SAMPLES_PER_OBJECT" \
  --seed "$SEED"

python scripts/evaluate_redshift_pit_coverage.py \
  --truth "$TRUTH" \
  --posterior "rws26=$RWS26_INFERENCE" \
  --posterior "rws24=$RWS24_INFERENCE" \
  --posterior "popcosmos=$PRIMARY" \
  --out "$PIT_OUT" \
  --samples-per-object "$SAMPLES_PER_OBJECT" \
  --bootstrap "$NUM_BOOTSTRAP" \
  --seed "$SEED" \
  --expected-objects "$EXPECTED_SPECZ"

if [[ "$RUN_ALL_DRAW_SENSITIVITY" == "1" ]]; then
  python scripts/evaluate_feniks_mira.py \
    --truth "$TRUTH" \
    --posterior "popcosmos=$ALL_DRAWS" \
    --out "$OUT_DIR/mira_popcosmos_all2560" \
    --parameter z_obs \
    --truth-column z_obs=redshift_true \
    --drop-nonfinite-truth \
    --num-regions "$NUM_REGIONS" \
    --num-bootstrap "$NUM_BOOTSTRAP" \
    --samples-per-object 0 \
    --seed "$SEED"
  python scripts/evaluate_feniks_tarp.py \
    --truth "$TRUTH" \
    --posterior "popcosmos=$ALL_DRAWS" \
    --out "$OUT_DIR/tarp_popcosmos_all2560" \
    --parameter z_obs \
    --truth-column z_obs=redshift_true \
    --drop-nonfinite-truth \
    --num-bootstrap "$NUM_BOOTSTRAP" \
    --samples-per-object 0 \
    --seed "$SEED"
  python scripts/evaluate_redshift_pit_coverage.py \
    --truth "$TRUTH" \
    --posterior "popcosmos=$ALL_DRAWS" \
    --out "$OUT_DIR/pit_popcosmos_all2560" \
    --samples-per-object 0 \
    --bootstrap "$NUM_BOOTSTRAP" \
    --seed "$SEED" \
    --expected-objects "$EXPECTED_SPECZ"
fi

for path in \
  "$MIRA_OUT/DONE" \
  "$MIRA_OUT/mira_summary.json" \
  "$TARP_OUT/DONE" \
  "$TARP_OUT/tarp_summary.json" \
  "$PIT_OUT/DONE" \
  "$PIT_OUT/redshift_calibration_summary.json"; do
  test -e "$path" || {
    echo "[popcosmos-chaincal][error] missing output: $path"
    exit 3
  }
done
if [[ "$RUN_ALL_DRAW_SENSITIVITY" == "1" ]]; then
  for path in \
    "$OUT_DIR/mira_popcosmos_all2560/DONE" \
    "$OUT_DIR/tarp_popcosmos_all2560/DONE" \
    "$OUT_DIR/pit_popcosmos_all2560/DONE"; do
    test -e "$path" || {
      echo "[popcosmos-chaincal][error] missing sensitivity output: $path"
      exit 3
    }
  done
fi

cat > "$OUT_DIR/run_contract.json" <<EOF
{
  "status": "complete",
  "scope": "same_1395_public_specz_redshift_only",
  "models": ["rws26", "rws24", "popcosmos"],
  "primary_samples_per_object": $SAMPLES_PER_OBJECT,
  "popcosmos_all_draw_sensitivity": $RUN_ALL_DRAW_SENSITIVITY,
  "num_regions": $NUM_REGIONS,
  "num_bootstrap": $NUM_BOOTSTRAP,
  "seed": $SEED
}
EOF
touch "$OUT_DIR/DONE"
echo "[popcosmos-chaincal] complete -> $OUT_DIR"
