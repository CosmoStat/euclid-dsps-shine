#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_DIR/outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536/bands26/full_cont120}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/runs/posthoc_smc_${RUN_TAG}}"
LIMIT="${LIMIT:-256}"
PROBE_LIMIT="${PROBE_LIMIT:-256}"
N_SHARDS="${N_SHARDS:-4}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-12}"
SMC_VARIANTS_CSV="${SMC_VARIANTS_CSV:-floor_0p00,floor_0p02,floor_0p05}"
SMC_SEEDS_CSV="${SMC_SEEDS_CSV:-260817,260818}"
export SMC_VARIANTS_CSV SMC_SEEDS_CSV
IFS=',' read -r -a SMC_VARIANTS <<< "$SMC_VARIANTS_CSV"
IFS=',' read -r -a SMC_SEEDS <<< "$SMC_SEEDS_CSV"
for variant in "${SMC_VARIANTS[@]}"; do
  case "$variant" in
    floor_0p00|floor_0p02|floor_0p05) ;;
    *) echo "[posthoc-smc-submit][error] unsupported variant: $variant" >&2; exit 2 ;;
  esac
done
if (( ${#SMC_VARIANTS[@]} < 1 || ${#SMC_SEEDS[@]} != 2 )); then
  echo "[posthoc-smc-submit][error] require at least one variant and exactly two seeds" >&2
  exit 2
fi

cd "$REPO_DIR"
test ! -e "$OUTPUT_ROOT" || {
  echo "[posthoc-smc-submit][error] output already exists: $OUTPUT_ROOT" >&2
  exit 2
}
mkdir -p outputs/logs "$OUTPUT_ROOT/cohorts"
CALIBRATION_INDICES="$SOURCE_ROOT/train/validation_indices.npy"
EVALUATION_INDICES="$SOURCE_ROOT/inference/inference_indices.npy"
for path in "$CALIBRATION_INDICES" "$EVALUATION_INDICES"; do
  test -s "$path" || { echo "[posthoc-smc-submit][error] missing: $path" >&2; exit 2; }
done
python - "$CALIBRATION_INDICES" "$EVALUATION_INDICES" "$OUTPUT_ROOT/cohorts" "$LIMIT" "$PROBE_LIMIT" "$N_SHARDS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

calibration_path, evaluation_path, out_path, limit, probe_limit, n_shards = sys.argv[1:]
limit = int(limit)
probe_limit = int(probe_limit)
n_shards = int(n_shards)
if min(limit, probe_limit, n_shards) <= 0:
    raise SystemExit("LIMIT, PROBE_LIMIT and N_SHARDS must be positive")
calibration = np.asarray(np.load(calibration_path), dtype=np.int64)
evaluation = np.asarray(np.load(evaluation_path), dtype=np.int64)
if len(calibration) < limit + probe_limit:
    raise SystemExit("Not enough calibration rows for disjoint SMC and probe cohorts")
rng = np.random.default_rng(260817)
selected = rng.permutation(calibration)[: limit + probe_limit]
smc = selected[:limit]
probe = selected[limit:]
if np.intersect1d(smc, probe).size or np.intersect1d(selected, evaluation).size:
    raise SystemExit("Training/probe/evaluation cohorts are not disjoint")
out = Path(out_path)
np.save(out / "smc_calibration_indices.npy", smc)
np.save(out / "proposal_probe_indices.npy", probe)
for index, shard in enumerate(np.array_split(smc, n_shards)):
    np.save(out / f"smc_calibration_indices_shard_{index:03d}.npy", shard)

def receipt(path):
    path = Path(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

payload = {
    "status": "complete",
    "selection": "seeded random disjoint subsets of the frozen validation rows",
    "seed": 260817,
    "smc_objects": limit,
    "proposal_probe_objects": probe_limit,
    "n_shards": n_shards,
    "spectroscopic_evaluation_overlap": 0,
    "source_calibration_indices": receipt(calibration_path),
    "source_evaluation_indices": receipt(evaluation_path),
}
(out / "cohort_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

array_tasks=$((${#SMC_VARIANTS[@]} * ${#SMC_SEEDS[@]} * N_SHARDS))
array_max=$((array_tasks - 1))
pilot_raw=$(sbatch --parsable --array="0-${array_max}%${ARRAY_CONCURRENCY}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",SOURCE_ROOT="$SOURCE_ROOT",OUTPUT_ROOT="$OUTPUT_ROOT",LIMIT="$LIMIT",N_SHARDS="$N_SHARDS",PARTICLES="${PARTICLES:-1024}",OBJECT_BATCH_SIZE="${OBJECT_BATCH_SIZE:-4}",TARGET_ESS_FRACTION="${TARGET_ESS_FRACTION:-0.5}",MAX_STAGES="${MAX_STAGES:-64}",MALA_STEPS="${MALA_STEPS:-2}",MALA_STEP_SIZE="${MALA_STEP_SIZE:-0.02}",MALA_PARTICLE_CHUNK_SIZE="${MALA_PARTICLE_CHUNK_SIZE:-64}" \
  scripts/popcosmos_posthoc_smc_h100.slurm)
pilot_job="${pilot_raw%%;*}"

final_raw=$(sbatch --parsable --dependency="afterok:${pilot_job}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",OUTPUT_ROOT="$OUTPUT_ROOT",N_SHARDS="$N_SHARDS",LIMIT="$LIMIT" \
  scripts/popcosmos_posthoc_smc_finalize.slurm)
final_job="${final_raw%%;*}"

env_file=outputs/logs/popcosmos_posthoc_smc_latest.env
printf 'export SMC_PILOT_JOB=%q\nexport SMC_FINALIZER_JOB=%q\nexport SMC_OUTPUT_ROOT=%q\nexport SOURCE_ROOT=%q\nexport SMC_CALIBRATION_INDICES=%q\nexport SMC_PROBE_INDICES=%q\nexport SMC_N_SHARDS=%q\nexport SMC_VARIANTS_CSV=%q\nexport SMC_SEEDS_CSV=%q\n' \
  "$pilot_job" "$final_job" "$OUTPUT_ROOT" "$SOURCE_ROOT" \
  "$OUTPUT_ROOT/cohorts/smc_calibration_indices.npy" \
  "$OUTPUT_ROOT/cohorts/proposal_probe_indices.npy" "$N_SHARDS" \
  "$SMC_VARIANTS_CSV" "$SMC_SEEDS_CSV" > "$env_file"
echo "smc_pilot_job=$pilot_job"
echo "smc_finalizer_job=$final_job"
echo "smc_output_root=$OUTPUT_ROOT"
echo "variants=$SMC_VARIANTS_CSV seeds=$SMC_SEEDS_CSV"
echo "array_tasks=$array_tasks concurrency=$ARRAY_CONCURRENCY shards_per_seed=$N_SHARDS"
echo "monitor: squeue -j $pilot_job,$final_job"
echo "logs: outputs/logs/cosmos_smc-${pilot_job}_<taskid>.out"
echo "latest_env=$env_file"
