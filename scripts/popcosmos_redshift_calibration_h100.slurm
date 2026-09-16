#!/bin/bash
#SBATCH --job-name=popcosmos_zcal
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
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
TRUTH="${TRUTH:-$RWS26_INFERENCE/inference_truth.parquet}"
MODEL_RWS26="${MODEL_RWS26:-rws26}"
MODEL_RWS24="${MODEL_RWS24:-rws24}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/redshift_calibration_${STAGE}_v1}"
FENIKS_ROOT="${FENIKS_ROOT:-/lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine/outputs/runs/feniks_selfsup_paper_v1}"
FENIKS_MIRA="${FENIKS_MIRA:-$FENIKS_ROOT/mira_rws_k8_t2_seed2_seed3_v2}"
FENIKS_TARP="${FENIKS_TARP:-$FENIKS_ROOT/tarp_rws_k8_t2_seed2_seed3_v1}"
PHOTOZ_METRICS="${PHOTOZ_METRICS:-$RUN_ROOT/redshift_only_comparison_v1/redshift_method_metrics.csv}"
EXPECTED_OBJECTS="${EXPECTED_OBJECTS:-5000}"
EXPECTED_SPECZ="${EXPECTED_SPECZ:-1395}"
EXPECTED_TRUTH_SHA256="${EXPECTED_TRUTH_SHA256:-83116e1252ca8f51684205b6042600b824ddc96a4dda5ad7c9842a94d86a8c4a}"
NUM_REGIONS="${NUM_REGIONS:-100}"
NUM_BOOTSTRAP="${NUM_BOOTSTRAP:-1000}"
NUM_ALPHA_BINS="${NUM_ALPHA_BINS:-0}"
SAMPLES_PER_OBJECT="${SAMPLES_PER_OBJECT:-128}"
SEED="${SEED:-260805}"
LIMIT="${LIMIT:-}"

MIRA_OUT="$OUT_DIR/mira"
TARP_OUT="$OUT_DIR/tarp"
COMPARISON_OUT="$OUT_DIR/comparison_with_current"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export JAX_PLATFORMS=cuda
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID:-manual}-popcosmos-zcal"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"

module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

for path in \
  "$TRUTH" \
  "$RWS26_INFERENCE/inference_truth.parquet" \
  "$RWS24_INFERENCE/inference_truth.parquet" \
  "$RWS26_INFERENCE/posterior_shards_manifest.json" \
  "$RWS24_INFERENCE/posterior_shards_manifest.json" \
  scripts/evaluate_feniks_mira.py \
  scripts/evaluate_feniks_tarp.py \
  scripts/compare_redshift_calibration_runs.py; do
  test -s "$path" || {
    echo "[popcosmos-zcal][error] missing: $path"
    exit 2
  }
done
for marker in \
  "${RWS26_INFERENCE%/inference}/DONE" \
  "${RWS24_INFERENCE%/inference}/DONE"; do
  test -e "$marker" || {
    echo "[popcosmos-zcal][error] missing completion marker: $marker"
    exit 2
  }
done
for source in "$RWS26_INFERENCE" "$RWS24_INFERENCE"; do
  compgen -G "$source/posterior_samples/*.parquet" >/dev/null || {
    echo "[popcosmos-zcal][error] no posterior shards under: $source"
    exit 2
  }
done
if [[ -e "$OUT_DIR" ]]; then
  echo "[popcosmos-zcal][error] output path already exists: $OUT_DIR"
  exit 2
fi
if [[ -z "$LIMIT" ]]; then
  for path in \
    "$FENIKS_MIRA/mira_scores.csv" \
    "$FENIKS_TARP/tarp_summary.csv"; do
    test -s "$path" || {
      echo "[popcosmos-zcal][error] missing comparison input: $path"
      exit 2
    }
  done
  if [[ ! -s "$PHOTOZ_METRICS" ]]; then
    echo "[popcosmos-zcal][warning] photo-z metric table not found: $PHOTOZ_METRICS"
    echo "[popcosmos-zcal][warning] cross-calibration comparison will omit that table"
  fi
fi

export RWS26_INFERENCE RWS24_INFERENCE EXPECTED_OBJECTS EXPECTED_SPECZ
export EXPECTED_TRUTH_SHA256 SAMPLES_PER_OBJECT
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

truth_frames = []
truth_hashes = []
for variable in ("RWS26_INFERENCE", "RWS24_INFERENCE"):
    inference = Path(os.environ[variable])
    manifest = json.loads(
        (inference / "posterior_shards_manifest.json").read_text(encoding="utf-8")
    )
    if not manifest.get("finalized") or manifest.get("incomplete"):
        raise SystemExit(f"[popcosmos-zcal][error] incomplete manifest: {inference}")
    expected_objects = int(os.environ["EXPECTED_OBJECTS"])
    if int(manifest["expected_selected_rows"]) != expected_objects:
        raise SystemExit(f"[popcosmos-zcal][error] wrong object count: {inference}")
    samples_rows = int(manifest["samples_rows"])
    samples_per_object = samples_rows // expected_objects
    if samples_rows % expected_objects or samples_per_object != int(
        os.environ["SAMPLES_PER_OBJECT"]
    ):
        raise SystemExit(f"[popcosmos-zcal][error] wrong sample count: {inference}")
    truth_path = inference / "inference_truth.parquet"
    truth_hashes.append(hashlib.sha256(truth_path.read_bytes()).hexdigest())
    truth_frames.append(pd.read_parquet(truth_path))
    print(
        f"[popcosmos-zcal] validated {inference}: objects={expected_objects}, "
        f"samples_per_object={samples_per_object}, shards={manifest['n_shards']}"
    )

if len(set(truth_hashes)) != 1:
    raise SystemExit("[popcosmos-zcal][error] RWS truth files differ")
if truth_hashes[0] != os.environ["EXPECTED_TRUTH_SHA256"]:
    raise SystemExit(
        f"[popcosmos-zcal][error] unexpected truth SHA-256: {truth_hashes[0]}"
    )
left, right = truth_frames
if not left.equals(right):
    raise SystemExit("[popcosmos-zcal][error] RWS truth tables differ")
finite = np.isfinite(left["redshift_true"].to_numpy(float))
if int(finite.sum()) != int(os.environ["EXPECTED_SPECZ"]):
    raise SystemExit(
        f"[popcosmos-zcal][error] expected {os.environ['EXPECTED_SPECZ']} "
        f"spec-z rows, found {int(finite.sum())}"
    )
print(
    f"[popcosmos-zcal] shared truth sha256={truth_hashes[0]} "
    f"specz={int(finite.sum())}"
)
PY

python - <<'PY'
import jax

print("[popcosmos-zcal] backend:", jax.default_backend())
print("[popcosmos-zcal] devices:", jax.devices())
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
    raise SystemExit("[popcosmos-zcal][error] expected exactly one local GPU")
PY
nvidia-smi

mkdir -p "$OUT_DIR"

mira_cmd=(
  python scripts/evaluate_feniks_mira.py
  --truth "$TRUTH"
  --posterior "$MODEL_RWS26=$RWS26_INFERENCE"
  --posterior "$MODEL_RWS24=$RWS24_INFERENCE"
  --out "$MIRA_OUT"
  --parameter z_obs
  --truth-column z_obs=redshift_true
  --drop-nonfinite-truth
  --num-regions "$NUM_REGIONS"
  --num-bootstrap "$NUM_BOOTSTRAP"
  --samples-per-object "$SAMPLES_PER_OBJECT"
  --seed "$SEED"
)
tarp_cmd=(
  python scripts/evaluate_feniks_tarp.py
  --truth "$TRUTH"
  --posterior "$MODEL_RWS26=$RWS26_INFERENCE"
  --posterior "$MODEL_RWS24=$RWS24_INFERENCE"
  --out "$TARP_OUT"
  --parameter z_obs
  --truth-column z_obs=redshift_true
  --drop-nonfinite-truth
  --num-alpha-bins "$NUM_ALPHA_BINS"
  --num-bootstrap "$NUM_BOOTSTRAP"
  --samples-per-object "$SAMPLES_PER_OBJECT"
  --seed "$SEED"
)
if [[ -n "$LIMIT" ]]; then
  mira_cmd+=(--limit "$LIMIT")
  tarp_cmd+=(--limit "$LIMIT")
fi

echo "[popcosmos-zcal] MIRA command:"
printf ' %q' "${mira_cmd[@]}"
echo
"${mira_cmd[@]}"

echo "[popcosmos-zcal] TARP command:"
printf ' %q' "${tarp_cmd[@]}"
echo
"${tarp_cmd[@]}"

if [[ -z "$LIMIT" ]]; then
  comparison_cmd=(
    python scripts/compare_redshift_calibration_runs.py
    --feniks-mira "$FENIKS_MIRA"
    --feniks-tarp "$FENIKS_TARP"
    --cosmos-mira "$MIRA_OUT"
    --cosmos-tarp "$TARP_OUT"
    --out "$COMPARISON_OUT"
  )
  if [[ -s "$PHOTOZ_METRICS" ]]; then
    comparison_cmd+=(--photoz-metrics "$PHOTOZ_METRICS")
  fi
  "${comparison_cmd[@]}"
fi

for artifact in \
  "$MIRA_OUT/DONE" \
  "$MIRA_OUT/mira_summary.json" \
  "$MIRA_OUT/mira_pairwise_differences.csv" \
  "$TARP_OUT/DONE" \
  "$TARP_OUT/tarp_summary.json" \
  "$TARP_OUT/tarp_pairwise_differences.csv"; do
  test -e "$artifact" || {
    echo "[popcosmos-zcal][error] missing output: $artifact"
    exit 3
  }
done
if [[ -z "$LIMIT" ]]; then
  test -e "$COMPARISON_OUT/DONE" || {
    echo "[popcosmos-zcal][error] missing comparison DONE"
    exit 3
  }
fi
touch "$OUT_DIR/DONE"
echo "[popcosmos-zcal] end: $(date --iso-8601=seconds)"
