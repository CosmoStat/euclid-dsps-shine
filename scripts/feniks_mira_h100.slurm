#!/bin/bash
#SBATCH --job-name=feniks_mira
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
PAPER_ROOT="${PAPER_ROOT:-outputs/runs/feniks_selfsup_paper_v1}"
SEED2_INFERENCE="${SEED2_INFERENCE:-$PAPER_ROOT/rws_k8_t2_seed2/inference}"
SEED3_INFERENCE="${SEED3_INFERENCE:-$PAPER_ROOT/rws_k8_t2_seed3/inference}"
TRUTH="${TRUTH:-$SEED2_INFERENCE/inference_truth.parquet}"
MODEL_SEED2="${MODEL_SEED2:-rws_k8_t2_seed2}"
MODEL_SEED3="${MODEL_SEED3:-rws_k8_t2_seed3}"
OUT_DIR="${OUT_DIR:-$PAPER_ROOT/mira_rws_k8_t2_seed2_seed3_v1}"
NUM_REGIONS="${NUM_REGIONS:-100}"
NUM_BOOTSTRAP="${NUM_BOOTSTRAP:-1000}"
SAMPLES_PER_OBJECT="${SAMPLES_PER_OBJECT:-128}"
SEED="${SEED:-260730}"
LIMIT="${LIMIT:-}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export JAX_PLATFORMS=cuda
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID:-manual}-feniks-mira"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"

# H100 nodes use a different CPU architecture and module tree than login nodes.
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

for path in \
  "$TRUTH" \
  "$SEED2_INFERENCE/inference_truth.parquet" \
  "$SEED3_INFERENCE/inference_truth.parquet" \
  "$SEED2_INFERENCE/posterior_shards_manifest.json" \
  "$SEED3_INFERENCE/posterior_shards_manifest.json" \
  scripts/evaluate_feniks_mira.py; do
  test -s "$path" || {
    echo "[mira][error] missing: $path"
    exit 2
  }
done
for marker in \
  "${SEED2_INFERENCE%/inference}/DONE" \
  "${SEED3_INFERENCE%/inference}/DONE"; do
  test -e "$marker" || {
    echo "[mira][error] missing completion marker: $marker"
    exit 2
  }
done
for source in "$SEED2_INFERENCE" "$SEED3_INFERENCE"; do
  posterior_found=false
  if [[ -s "$source/posterior_samples.parquet" ]]; then
    posterior_found=true
  elif compgen -G "$source/posterior_samples/*.parquet" >/dev/null; then
    posterior_found=true
  fi
  if [[ "$posterior_found" != true ]]; then
    echo "[mira][error] no posterior parquet under: $source"
    exit 2
  fi
done
if [[ -e "$OUT_DIR" ]]; then
  echo "[mira][error] output path already exists: $OUT_DIR"
  exit 2
fi

export SEED2_INFERENCE SEED3_INFERENCE SAMPLES_PER_OBJECT LIMIT
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

truth_hashes = []
for variable in ("SEED2_INFERENCE", "SEED3_INFERENCE"):
    inference = Path(os.environ[variable])
    manifest = json.loads(
        (inference / "posterior_shards_manifest.json").read_text(encoding="utf-8")
    )
    if not manifest.get("finalized") or manifest.get("incomplete"):
        raise SystemExit(f"[mira][error] incomplete posterior manifest: {inference}")
    expected_objects = int(manifest["expected_selected_rows"])
    actual_samples = int(manifest["samples_rows"])
    if actual_samples % expected_objects:
        raise SystemExit(
            f"[mira][error] sample rows are not divisible by objects: {inference}"
        )
    actual_per_object = actual_samples // expected_objects
    requested = int(os.environ["SAMPLES_PER_OBJECT"])
    if requested and actual_per_object < requested:
        raise SystemExit(
            f"[mira][error] {inference} has only {actual_per_object} samples per "
            f"object, fewer than requested {requested}"
        )
    truth = inference / "inference_truth.parquet"
    truth_hashes.append(hashlib.sha256(truth.read_bytes()).hexdigest())
    print(
        f"[mira] validated {inference}: objects={expected_objects}, "
        f"samples={actual_samples}, samples_per_object={actual_per_object}, "
        f"shards={manifest['n_shards']}"
    )
if len(set(truth_hashes)) != 1:
    raise SystemExit("[mira][error] seed2 and seed3 truth files differ")
print(f"[mira] shared truth sha256={truth_hashes[0]}")
PY

python - <<'PY'
import jax

print("[mira] backend:", jax.default_backend())
print("[mira] devices:", jax.devices())
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
    raise SystemExit("[mira][error] expected exactly one local GPU")
PY
nvidia-smi

cmd=(
  python scripts/evaluate_feniks_mira.py
  --truth "$TRUTH"
  --posterior "$MODEL_SEED2=$SEED2_INFERENCE"
  --posterior "$MODEL_SEED3=$SEED3_INFERENCE"
  --out "$OUT_DIR"
  --num-regions "$NUM_REGIONS"
  --num-bootstrap "$NUM_BOOTSTRAP"
  --samples-per-object "$SAMPLES_PER_OBJECT"
  --seed "$SEED"
)
if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

echo "[mira] truth=$TRUTH"
echo "[mira] posterior=$MODEL_SEED2=$SEED2_INFERENCE"
echo "[mira] posterior=$MODEL_SEED3=$SEED3_INFERENCE"
echo "[mira] out=$OUT_DIR"
echo "[mira] command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

for artifact in \
  "$OUT_DIR/DONE" \
  "$OUT_DIR/mira_manifest.json" \
  "$OUT_DIR/mira_summary.json" \
  "$OUT_DIR/mira_scores.parquet" \
  "$OUT_DIR/mira_object_contributions.parquet" \
  "$OUT_DIR/mira_scores.png"; do
  test -e "$artifact" || {
    echo "[mira][error] missing output: $artifact"
    exit 3
  }
done

echo "[mira] end: $(date --iso-8601=seconds)"
