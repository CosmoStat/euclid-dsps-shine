#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:?Set DIAGNOSTIC_ROOT}"
UPDATED_CHECKPOINT="${UPDATED_CHECKPOINT:?Set UPDATED_CHECKPOINT}"
UPDATED_FEATURE_STATS="${UPDATED_FEATURE_STATS:?Set UPDATED_FEATURE_STATS}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$DIAGNOSTIC_ROOT/warm_start_adapter_${RUN_TAG}}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-12}"

cd "$REPO_DIR"
for path in \
  "$DIAGNOSTIC_ROOT/cohorts/diagnostic_panel.parquet" \
  "$DIAGNOSTIC_ROOT/floor_0p05/seed_260817/DONE" \
  "$DIAGNOSTIC_ROOT/floor_0p05/seed_260818/DONE" \
  "$UPDATED_CHECKPOINT" \
  "${UPDATED_CHECKPOINT}.json" \
  "$UPDATED_FEATURE_STATS"; do
  test -e "$path" || { echo "[proposal-adapter-submit][error] missing: $path" >&2; exit 2; }
done
test ! -e "$ADAPTER_ROOT" || {
  echo "[proposal-adapter-submit][error] output exists: $ADAPTER_ROOT" >&2
  exit 2
}
mkdir -p outputs/logs "$ADAPTER_ROOT"

python - "$ADAPTER_ROOT/experiment_manifest.json" "$DIAGNOSTIC_ROOT" "$UPDATED_CHECKPOINT" "$UPDATED_FEATURE_STATS" "${ADAPTER_PROPOSAL_SAMPLES:-2048}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, diagnostic_root, checkpoint, feature_stats = map(Path, sys.argv[1:5])
proposal_samples = int(sys.argv[5])

def receipt(path):
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

payload = {
    "status": "submitted",
    "contract": (
        "current q plus zero-initialized residual context; source encoder frozen "
        "exactly; SMC-A fit and SMC-B validation; no catalog truth"
    ),
    "candidates": [
        "current_compressed",
        "free_context_adapter",
        "direct_photometry_adapter",
        "band_token_adapter",
    ],
    "seeds": [260820, 260821, 260822],
    "proposal_samples_per_object": proposal_samples,
    "support_thresholds": {
        "min_median_raw_ess_fraction": 0.05,
        "max_fraction_pareto_k_gt_0p7": 0.2,
    },
    "diagnostic_root": str(diagnostic_root),
    "checkpoint": receipt(checkpoint),
    "feature_stats": receipt(feature_stats),
}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

array_tasks=12
array_max=$((array_tasks - 1))
raw=$(sbatch --parsable --array="0-${array_max}%${ARRAY_CONCURRENCY}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",DIAGNOSTIC_ROOT="$DIAGNOSTIC_ROOT",ADAPTER_ROOT="$ADAPTER_ROOT",UPDATED_CHECKPOINT="$UPDATED_CHECKPOINT",UPDATED_FEATURE_STATS="$UPDATED_FEATURE_STATS",ADAPTER_EPOCHS="${ADAPTER_EPOCHS:-120}",ADAPTER_OBJECT_BATCH_SIZE="${ADAPTER_OBJECT_BATCH_SIZE:-8}",ADAPTER_LEARNING_RATE="${ADAPTER_LEARNING_RATE:-2e-4}",ADAPTER_WEIGHT_DECAY="${ADAPTER_WEIGHT_DECAY:-1e-6}",ADAPTER_PROPOSAL_SAMPLES="${ADAPTER_PROPOSAL_SAMPLES:-2048}",ADAPTER_DECODER_SAMPLE_CHUNK_SIZE="${ADAPTER_DECODER_SAMPLE_CHUNK_SIZE:-1}",ADAPTER_GEOMETRY_DRAWS="${ADAPTER_GEOMETRY_DRAWS:-256}",ADAPTER_GEOMETRY_PROJECTIONS="${ADAPTER_GEOMETRY_PROJECTIONS:-64}" \
  scripts/popcosmos_proposal_adapter_h100.slurm)
PROPOSAL_ADAPTER_JOB="${raw%%;*}"

final_raw=$(sbatch --parsable --dependency="afterok:${PROPOSAL_ADAPTER_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",ADAPTER_ROOT="$ADAPTER_ROOT" \
  scripts/popcosmos_proposal_adapter_finalize.slurm)
PROPOSAL_ADAPTER_FINALIZER_JOB="${final_raw%%;*}"

env_file=outputs/logs/popcosmos_proposal_adapter_latest.env
printf 'export PROPOSAL_ADAPTER_JOB=%q\nexport PROPOSAL_ADAPTER_FINALIZER_JOB=%q\nexport ADAPTER_ROOT=%q\nexport DIAGNOSTIC_ROOT=%q\nexport UPDATED_CHECKPOINT=%q\nexport UPDATED_FEATURE_STATS=%q\n' \
  "$PROPOSAL_ADAPTER_JOB" "$PROPOSAL_ADAPTER_FINALIZER_JOB" \
  "$ADAPTER_ROOT" "$DIAGNOSTIC_ROOT" "$UPDATED_CHECKPOINT" \
  "$UPDATED_FEATURE_STATS" > "$env_file"

echo "proposal_adapter_job=$PROPOSAL_ADAPTER_JOB"
echo "proposal_adapter_finalizer_job=$PROPOSAL_ADAPTER_FINALIZER_JOB"
echo "adapter_root=$ADAPTER_ROOT"
echo "array_tasks=$array_tasks concurrency=$ARRAY_CONCURRENCY"
echo "latest_env=$env_file"
