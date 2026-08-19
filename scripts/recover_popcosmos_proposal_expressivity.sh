#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:?Source the proposal-expressivity receipt first}"
UPDATED_CHECKPOINT="${UPDATED_CHECKPOINT:?Source the proposal-expressivity receipt first}"
UPDATED_FEATURE_STATS="${UPDATED_FEATURE_STATS:?Source the proposal-expressivity receipt first}"
PROPOSAL_REFRESH_OUT="${PROPOSAL_REFRESH_OUT:?Source the proposal-expressivity receipt first}"
ORIGINAL_SMC_DIAGNOSTIC_JOB="${SMC_DIAGNOSTIC_JOB:-}"

cd "$REPO_DIR"
PANEL_SUMMARY="$DIAGNOSTIC_ROOT/cohorts/diagnostic_panel_summary.json"
test -s "$PANEL_SUMMARY" || {
  echo "[proposal-expressivity-recover][error] missing: $PANEL_SUMMARY" >&2
  exit 2
}

read -r LIMIT N_SHARDS < <(
  python - "$PANEL_SUMMARY" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["n_objects"], payload["n_shards"])
PY
)

SMC_VARIANTS_CSV="floor_0p05"
SMC_SEEDS_CSV="260817,260818"
export SMC_VARIANTS_CSV SMC_SEEDS_CSV

missing_tasks=$(
  python - "$DIAGNOSTIC_ROOT" "$N_SHARDS" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
n_shards = int(sys.argv[2])
seeds = (260817, 260818)
missing = []
for task in range(len(seeds) * n_shards):
    group, shard = divmod(task, n_shards)
    done = root / "floor_0p05" / f"seed_{seeds[group]}" / f"shard_{shard:03d}" / "DONE"
    if not done.is_file():
        missing.append(str(task))
print(",".join(missing))
PY
)
if [[ -z "$missing_tasks" ]]; then
  echo "[proposal-expressivity-recover][error] no missing SMC task found" >&2
  exit 2
fi

if [[ -n "$ORIGINAL_SMC_DIAGNOSTIC_JOB" ]]; then
  scancel "${SMC_DIAGNOSTIC_FINALIZER_JOB:-}" "${PROPOSAL_EXPRESSIVITY_JOB:-}" 2>/dev/null || true
fi

stamp=$(date +%Y%m%d_%H%M%S)
python - "$DIAGNOSTIC_ROOT" "$N_SHARDS" "$missing_tasks" "$stamp" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
n_shards = int(sys.argv[2])
tasks = [int(value) for value in sys.argv[3].split(",")]
stamp = sys.argv[4]
seeds = (260817, 260818)
for task in tasks:
    group, shard = divmod(task, n_shards)
    path = root / "floor_0p05" / f"seed_{seeds[group]}" / f"shard_{shard:03d}"
    if path.exists() and not (path / "DONE").is_file():
        path.rename(path.with_name(f"{path.name}.failed_{stamp}"))
PY

retry_raw=$(sbatch --parsable --array="${missing_tasks}%${ARRAY_CONCURRENCY:-16}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",SOURCE_ROOT="$PROPOSAL_REFRESH_OUT",SMC_CHECKPOINT="$UPDATED_CHECKPOINT",SMC_FEATURE_STATS="$UPDATED_FEATURE_STATS",OUTPUT_ROOT="$DIAGNOSTIC_ROOT",LIMIT="$LIMIT",N_SHARDS="$N_SHARDS",PARTICLES="${PARTICLES:-1024}",OBJECT_BATCH_SIZE="${SMC_OBJECT_BATCH_SIZE:-4}",TARGET_ESS_FRACTION="${TARGET_ESS_FRACTION:-0.5}",MAX_STAGES="${MAX_STAGES:-64}",MALA_STEPS="${MALA_STEPS:-2}",MALA_STEP_SIZE="${MALA_STEP_SIZE:-0.005}",MALA_PARTICLE_CHUNK_SIZE="${MALA_PARTICLE_CHUNK_SIZE:-64}" \
  scripts/popcosmos_posthoc_smc_h100.slurm)
SMC_DIAGNOSTIC_JOB="${retry_raw%%;*}"

final_raw=$(sbatch --parsable --dependency="afterok:${SMC_DIAGNOSTIC_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",OUTPUT_ROOT="$DIAGNOSTIC_ROOT",N_SHARDS="$N_SHARDS",LIMIT="$LIMIT" \
  scripts/popcosmos_posthoc_smc_finalize.slurm)
SMC_DIAGNOSTIC_FINALIZER_JOB="${final_raw%%;*}"

expr_raw=$(sbatch --parsable --dependency="afterok:${SMC_DIAGNOSTIC_FINALIZER_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",DIAGNOSTIC_ROOT="$DIAGNOSTIC_ROOT",UPDATED_CHECKPOINT="$UPDATED_CHECKPOINT",UPDATED_FEATURE_STATS="$UPDATED_FEATURE_STATS",EPOCHS="${EPOCHS:-40}",PROPOSAL_SAMPLES="${PROPOSAL_SAMPLES:-2048}",DECODER_SAMPLE_CHUNK_SIZE="${DECODER_SAMPLE_CHUNK_SIZE:-1}",EXPERTS="${EXPERTS:-2}",OBJECT_BATCH_SIZE="${OBJECT_BATCH_SIZE:-8}",LEARNING_RATE="${LEARNING_RATE:-2e-5}",WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}",MIXTURE_MEAN_OFFSET="${MIXTURE_MEAN_OFFSET:-0.05}",GEOMETRY_DRAWS="${GEOMETRY_DRAWS:-256}",GEOMETRY_PROJECTIONS="${GEOMETRY_PROJECTIONS:-64}",SEED="${SEED:-260819}" \
  scripts/popcosmos_proposal_expressivity_h100.slurm)
PROPOSAL_EXPRESSIVITY_JOB="${expr_raw%%;*}"

env_file=outputs/logs/popcosmos_proposal_expressivity_latest.env
printf 'export ORIGINAL_SMC_DIAGNOSTIC_JOB=%q\nexport SMC_DIAGNOSTIC_JOB=%q\nexport SMC_DIAGNOSTIC_FINALIZER_JOB=%q\nexport PROPOSAL_EXPRESSIVITY_JOB=%q\nexport DIAGNOSTIC_ROOT=%q\nexport PROPOSAL_EXPRESSIVITY_OUT=%q\nexport PROPOSAL_REFRESH_OUT=%q\nexport UPDATED_CHECKPOINT=%q\nexport UPDATED_FEATURE_STATS=%q\nexport IMPORTANCE_DIAGNOSTICS=%q\n' \
  "$ORIGINAL_SMC_DIAGNOSTIC_JOB" "$SMC_DIAGNOSTIC_JOB" \
  "$SMC_DIAGNOSTIC_FINALIZER_JOB" "$PROPOSAL_EXPRESSIVITY_JOB" \
  "$DIAGNOSTIC_ROOT" "$DIAGNOSTIC_ROOT/proposal_expressivity" \
  "$PROPOSAL_REFRESH_OUT" "$UPDATED_CHECKPOINT" "$UPDATED_FEATURE_STATS" \
  "${IMPORTANCE_DIAGNOSTICS:-}" > "$env_file"

echo "missing_tasks=$missing_tasks"
echo "smc_retry_job=$SMC_DIAGNOSTIC_JOB"
echo "smc_finalizer_job=$SMC_DIAGNOSTIC_FINALIZER_JOB"
echo "proposal_expressivity_job=$PROPOSAL_EXPRESSIVITY_JOB"
echo "diagnostic_root=$DIAGNOSTIC_ROOT"
echo "latest_env=$env_file"
