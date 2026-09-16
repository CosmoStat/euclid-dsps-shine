#!/bin/bash
#SBATCH --job-name=feniks_sc_topoint
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
PILOT_ROOT="${PILOT_ROOT:?PILOT_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
case "$TASK" in
  0) ARM=A ;;
  1) ARM=B ;;
  2) ARM=C ;;
  *) echo "invalid validation task: $TASK" >&2; exit 2 ;;
esac

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
mkdir -p "$CACHE_ROOT/jax" "$MPLCONFIGDIR"

readarray -t VALUES < <(python - "$PILOT_ROOT/RUN_MANIFEST.json" "$ARM" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).parent
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
arm = sys.argv[2]
if arm == "A":
    print(manifest["source"]["config"])
    print(manifest["source"]["checkpoint"])
    print(manifest["source"]["feature_stats"])
else:
    receipt = json.loads((root / "arms" / arm / "ARM_COMPLETE.json").read_text(encoding="utf-8"))
    print(receipt["config"])
    print(receipt["checkpoint"])
    print(receipt["feature_stats"])
print(manifest["dataset"]["path"])
print(manifest["cohorts"]["validation_pilot"]["path"])
print(manifest["cohorts"]["validation_pilot"]["objects"])
PY
)
CONFIG="${VALUES[0]}"
CHECKPOINT="${VALUES[1]}"
FEATURE_STATS="${VALUES[2]}"
DATASET="${VALUES[3]}"
TRACKING_INDICES="${VALUES[4]}"
TRACKING_OBJECTS="${VALUES[5]}"
OUT="$PILOT_ROOT/validation/$ARM"

for receipt in \
  "$OUT/tracking_k256/summary/TRUTH_FREE_POSTERIOR_VALIDATION.json" \
  "$OUT/support_k1024/summary/TRUTH_FREE_POSTERIOR_VALIDATION.json"; do
  test -s "$receipt" || {
    echo "[topology-npe-internal-recovery][error] missing completed support receipt: $receipt" >&2
    exit 2
  }
done

nvidia-smi
python scripts/evaluate_feniks_sc_drws_topology_npe_internal.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" --dataset "$DATASET" \
  --row-indices "$TRACKING_INDICES" --out "$OUT/internal" \
  --objects "$TRACKING_OBJECTS" --posterior-draws 64 \
  --decoder-sample-chunk-size 1 --seed "$((264000 + TASK))"

python - "$OUT" "$ARM" "$TOPOLOGY_INTERNAL_RECOVERY_AUTHORIZATION" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "status": "COMPLETE",
    "arm": sys.argv[2],
    "truth_used": False,
    "tracking": json.loads((out / "tracking_k256/summary/TRUTH_FREE_POSTERIOR_VALIDATION.json").read_text()),
    "support": json.loads((out / "support_k1024/summary/TRUTH_FREE_POSTERIOR_VALIDATION.json").read_text()),
    "internal": json.loads((out / "internal/INTERNAL_TRUTH_FREE_VALIDATION.json").read_text()),
    "recovery_authorization": sys.argv[3],
    "reused_completed_inference": True,
    "scientific_promotion": False,
}
(out / "VALIDATION_COMPLETE.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY
echo "[topology-npe-internal-recovery] validation complete arm=$ARM"
