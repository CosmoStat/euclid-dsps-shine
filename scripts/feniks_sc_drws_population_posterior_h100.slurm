#!/bin/bash
#SBATCH --job-name=feniks_sc_poppost
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
POSTERIOR_ROOT="${POSTERIOR_ROOT:?POSTERIOR_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
MANIFEST="$POSTERIOR_ROOT/RUN_MANIFEST.json"
OUT="$POSTERIOR_ROOT/shards/shard_$(printf '%05d' "$TASK")"
INFERENCE="$OUT/inference"
PARENT_IW="$OUT/projected_parent_iw"
SOURCE_IW="$OUT/source_prior_iw"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1 EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"

cd "$REPO_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$OUT" "$MPLCONFIGDIR" "$CACHE_ROOT/jax"

readarray -t VALUES < <(python - "$MANIFEST" "$TASK" "$REPO_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from euclid_dsps.amortized.population_vem import require_git_commit

manifest_path = Path(sys.argv[1])
task = int(sys.argv[2])
repo = Path(sys.argv[3]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
require_git_commit(repo, manifest["code_commit"])
shard = manifest["cohort"]["shards"][task]
records = (
    (manifest["model"]["checkpoint"], manifest["model"]["checkpoint_sha256"]),
    (manifest["model"]["checkpoint_sidecar"], manifest["model"]["checkpoint_sidecar_sha256"]),
    (manifest["model"]["feature_stats"], manifest["model"]["feature_stats_sha256"]),
    (manifest["model"]["config"], manifest["model"]["config_sha256"]),
    (manifest["source_model"]["checkpoint"], manifest["source_model"]["checkpoint_sha256"]),
    (manifest["source_model"]["checkpoint_sidecar"], manifest["source_model"]["checkpoint_sidecar_sha256"]),
    (manifest["source_model"]["config"], manifest["source_model"]["config_sha256"]),
    (manifest["dataset"]["path"], manifest["dataset"]["sha256"]),
    (shard["path"], shard["sha256"]),
)
for raw_path, expected in records:
    path = Path(raw_path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing input: {path}")
    digest_state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_state.update(chunk)
    digest = digest_state.hexdigest()
    if digest != expected:
        raise SystemExit(f"SHA256 mismatch: {path}")
print(manifest["model"]["config"])
print(manifest["source_model"]["config"])
print(manifest["model"]["checkpoint"])
print(manifest["source_model"]["checkpoint"])
print(manifest["model"]["feature_stats"])
print(manifest["dataset"]["path"])
print(shard["path"])
print(shard["objects"])
print(manifest["inference"]["posterior_draws_per_object"])
print(manifest["inference"]["psis_resample_draws_per_object"])
print(manifest["inference"]["prior_draws_per_shard"])
print(manifest["inference"]["object_batch_size"])
PY
)
CONFIG="${VALUES[0]}"
SOURCE_CONFIG="${VALUES[1]}"
CHECKPOINT="${VALUES[2]}"
SOURCE_CHECKPOINT="${VALUES[3]}"
FEATURE_STATS="${VALUES[4]}"
DATASET="${VALUES[5]}"
INDICES="${VALUES[6]}"
OBJECTS="${VALUES[7]}"
POSTERIOR_DRAWS="${VALUES[8]}"
RESAMPLE_DRAWS="${VALUES[9]}"
PRIOR_DRAWS="${VALUES[10]}"
OBJECT_BATCH_SIZE="${VALUES[11]}"

test ! -e "$OUT/DONE" || {
  echo "[population-posterior] already complete: $OUT"
  exit 0
}
python - "$CONFIG" <<'PY'
import sys
from euclid_dsps.config import load_config

config = load_config(sys.argv[1])
inference = config["amortized"]["inference"]
if inference.get("write_truth_snapshot") or inference.get("write_truth_diagnostics"):
    raise SystemExit("worker inference config must be truth-free")
if (config.get("truth", {}) or {}).get("parameter_columns"):
    raise SystemExit("worker inference config exposes truth parameters")
import jax
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
    raise SystemExit(f"expected exactly one GPU, got {jax.devices()}")
print(f"[population-posterior] backend={jax.default_backend()} devices={jax.devices()}")
PY

COMMON=(--dataset "$DATASET" --checkpoint "$CHECKPOINT"
  --feature-stats "$FEATURE_STATS" --row-indices-file "$INDICES"
  --selection-mode sequential --limit "$OBJECTS" --batch-size "$OBJECT_BATCH_SIZE"
  --jax-batch-size "$OBJECT_BATCH_SIZE" --posterior-samples "$POSTERIOR_DRAWS"
  --prior-samples "$PRIOR_DRAWS" --decoder-sample-chunk-size 1
  --prior-predictive-batch-size 64 --shard-outputs)

echo "[population-posterior] shard=$TASK objects=$OBJECTS K=$POSTERIOR_DRAWS"
nvidia-smi
python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-diffsky \
  "${COMMON[@]}" --seed "$((290400 + TASK))" --out "$INFERENCE"
python -m euclid_dsps.cli --config "$CONFIG" amortized-finalize-inference \
  --out "$INFERENCE" --limit "$OBJECTS" --quiet

if [[ ! -e "$PARENT_IW/DONE" ]]; then
  rm -rf "$PARENT_IW"
fi
python scripts/importance_correct_posterior.py \
  --posterior "$INFERENCE" --out "$PARENT_IW" \
  --resample-count "$RESAMPLE_DRAWS" --seed "$((291400 + TASK))" \
  --min-median-ess-fraction 0.05 --max-fraction-pareto-k-gt-0p7 0.20
if [[ ! -e "$SOURCE_IW/DONE" ]]; then
  rm -rf "$SOURCE_IW"
fi
python scripts/importance_correct_posterior.py \
  --posterior "$INFERENCE" --out "$SOURCE_IW" \
  --config "$SOURCE_CONFIG" --target-checkpoint "$SOURCE_CHECKPOINT" \
  --resample-count "$RESAMPLE_DRAWS" --seed "$((292400 + TASK))" \
  --min-median-ess-fraction 0.05 --max-fraction-pareto-k-gt-0p7 0.20

for forbidden in inference_truth.parquet redshift_pit.parquet \
  corner_full_latent_truth_prior_posterior.png; do
  test ! -e "$INFERENCE/$forbidden" || {
    echo "[population-posterior][error] truth artifact created during inference: $forbidden" >&2
    exit 2
  }
done
python - "$OUT/SHARD_COMPLETE.json" "$TASK" "$OBJECTS" "$POSTERIOR_DRAWS" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "status": "COMPLETE",
    "shard": int(sys.argv[2]),
    "objects": int(sys.argv[3]),
    "posterior_draws_per_object": int(sys.argv[4]),
    "truth_used": False,
    "targets": ["projected_parent_prior", "source_prior_baseline"],
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
touch "$OUT/DONE"
echo "[population-posterior] complete shard=$TASK"
