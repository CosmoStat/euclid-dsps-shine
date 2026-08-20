#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:?Set DIAGNOSTIC_ROOT to the completed proposal diagnostic}"
UPDATED_CHECKPOINT="${UPDATED_CHECKPOINT:?Set UPDATED_CHECKPOINT}"
UPDATED_FEATURE_STATS="${UPDATED_FEATURE_STATS:?Set UPDATED_FEATURE_STATS}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ARCHITECTURE_ROOT="${ARCHITECTURE_ROOT:-$DIAGNOSTIC_ROOT/posterior_architecture_phase01_${RUN_TAG}}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-19}"

cd "$REPO_DIR"
for path in \
  "$DIAGNOSTIC_ROOT/cohorts/diagnostic_panel.parquet" \
  "$DIAGNOSTIC_ROOT/floor_0p05/seed_260817/DONE" \
  "$DIAGNOSTIC_ROOT/floor_0p05/seed_260818/DONE" \
  "$UPDATED_CHECKPOINT" \
  "${UPDATED_CHECKPOINT}.json" \
  "$UPDATED_FEATURE_STATS"; do
  test -e "$path" || { echo "[proposal-architecture-submit][error] missing: $path" >&2; exit 2; }
done
test ! -e "$ARCHITECTURE_ROOT" || {
  echo "[proposal-architecture-submit][error] output exists: $ARCHITECTURE_ROOT" >&2
  exit 2
}
mkdir -p outputs/logs "$ARCHITECTURE_ROOT"

python - "$ARCHITECTURE_ROOT/experiment_manifest.json" "$DIAGNOSTIC_ROOT" "$UPDATED_CHECKPOINT" "$UPDATED_FEATURE_STATS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, diagnostic_root, checkpoint, feature_stats = map(Path, sys.argv[1:])

def receipt(path):
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

payload = {
    "status": "submitted",
    "self_supervision_contract": (
        "no catalog truth; weighted SMC-A fit, independent SMC-B validation, "
        "observed photometry conditioning"
    ),
    "frozen_contract": (
        "population prior, likelihood, feature stats, object panel, SMC banks "
        "and ordinary-IS thresholds"
    ),
    "phase0_candidates": [
        "current_compressed",
        "oracle_kde",
        "free_context_rqspline",
        "direct_context_realnvp",
    ],
    "phase1_candidates": [
        "current_compressed",
        "direct_context_realnvp",
        "direct_context_rqspline_medium",
        "direct_context_rqspline_large",
        "band_token_rqspline",
    ],
    "phase1_seeds": [260820, 260821, 260822],
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

# Four Phase-0 tasks and five Phase-1 architectures times three seeds.
array_tasks=19
array_max=$((array_tasks - 1))
raw=$(sbatch --parsable --array="0-${array_max}%${ARRAY_CONCURRENCY}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",DIAGNOSTIC_ROOT="$DIAGNOSTIC_ROOT",ARCHITECTURE_ROOT="$ARCHITECTURE_ROOT",UPDATED_CHECKPOINT="$UPDATED_CHECKPOINT",UPDATED_FEATURE_STATS="$UPDATED_FEATURE_STATS",EPOCHS="${EPOCHS:-80}",OBJECT_BATCH_SIZE="${OBJECT_BATCH_SIZE:-8}",LEARNING_RATE="${LEARNING_RATE:-5e-5}",WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}",PROPOSAL_SAMPLES="${PROPOSAL_SAMPLES:-2048}",DECODER_SAMPLE_CHUNK_SIZE="${DECODER_SAMPLE_CHUNK_SIZE:-1}",GEOMETRY_DRAWS="${GEOMETRY_DRAWS:-256}",GEOMETRY_PROJECTIONS="${GEOMETRY_PROJECTIONS:-64}",KDE_CENTERS="${KDE_CENTERS:-128}",KDE_BANDWIDTH="${KDE_BANDWIDTH:-0.3}" \
  scripts/popcosmos_proposal_architecture_h100.slurm)
PROPOSAL_ARCHITECTURE_JOB="${raw%%;*}"

final_raw=$(sbatch --parsable --dependency="afterok:${PROPOSAL_ARCHITECTURE_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",ARCHITECTURE_ROOT="$ARCHITECTURE_ROOT" \
  scripts/popcosmos_proposal_architecture_finalize.slurm)
PROPOSAL_ARCHITECTURE_FINALIZER_JOB="${final_raw%%;*}"

env_file=outputs/logs/popcosmos_proposal_architecture_latest.env
printf 'export PROPOSAL_ARCHITECTURE_JOB=%q\nexport PROPOSAL_ARCHITECTURE_FINALIZER_JOB=%q\nexport ARCHITECTURE_ROOT=%q\nexport DIAGNOSTIC_ROOT=%q\nexport UPDATED_CHECKPOINT=%q\nexport UPDATED_FEATURE_STATS=%q\n' \
  "$PROPOSAL_ARCHITECTURE_JOB" "$PROPOSAL_ARCHITECTURE_FINALIZER_JOB" \
  "$ARCHITECTURE_ROOT" "$DIAGNOSTIC_ROOT" "$UPDATED_CHECKPOINT" \
  "$UPDATED_FEATURE_STATS" > "$env_file"

echo "proposal_architecture_job=$PROPOSAL_ARCHITECTURE_JOB"
echo "proposal_architecture_finalizer_job=$PROPOSAL_ARCHITECTURE_FINALIZER_JOB"
echo "architecture_root=$ARCHITECTURE_ROOT"
echo "array_tasks=$array_tasks concurrency=$ARRAY_CONCURRENCY"
echo "latest_env=$env_file"
