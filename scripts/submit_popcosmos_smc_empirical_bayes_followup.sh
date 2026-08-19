#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_DIR/outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536/bands26/full_cont120}"
PARENT_SMC_ROOT="${PARENT_SMC_ROOT:?Set PARENT_SMC_ROOT to the completed SMC512 root}"
SMC_EM_OUT="${SMC_EM_OUT:?Set SMC_EM_OUT to the selected direct-SMC M-step output}"
CONFIRM_OBJECTS="${CONFIRM_OBJECTS:-256}"
PROBE_OBJECTS="${PROBE_OBJECTS:-256}"
OBJECTS_PER_SHARD="${OBJECTS_PER_SHARD:-16}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-32}"
RUN_TAG="${RUN_TAG:-smc_em_confirm${CONFIRM_OBJECTS}_$(date +%Y%m%d_%H%M%S)}"

if (( CONFIRM_OBJECTS <= 0 || PROBE_OBJECTS <= 0 || OBJECTS_PER_SHARD <= 0 )); then
  echo "[smc-em-followup][error] cohort and shard sizes must be positive" >&2
  exit 2
fi
if (( CONFIRM_OBJECTS % OBJECTS_PER_SHARD != 0 )); then
  echo "[smc-em-followup][error] CONFIRM_OBJECTS must be divisible by OBJECTS_PER_SHARD" >&2
  exit 2
fi

cd "$REPO_DIR"
SOURCE_CHECKPOINT="$SOURCE_ROOT/train/checkpoints/best.eqx"
SOURCE_STATS="$SOURCE_ROOT/train/feature_stats.json"
CANDIDATE_CHECKPOINT="$SMC_EM_OUT/checkpoints/best.eqx"
EM_SUMMARY="$SMC_EM_OUT/smc_empirical_bayes_summary.json"
PARENT_SELECTION="$PARENT_SMC_ROOT/pilot_selection/selection_summary.json"
PARENT_SMC_INDICES="$PARENT_SMC_ROOT/cohorts/smc_calibration_indices.npy"
PARENT_PROBE_INDICES="$PARENT_SMC_ROOT/cohorts/proposal_probe_indices.npy"

for path in \
  "$SOURCE_CHECKPOINT" \
  "$SOURCE_STATS" \
  "$CANDIDATE_CHECKPOINT" \
  "$EM_SUMMARY" \
  "$PARENT_SELECTION" \
  "$PARENT_SMC_INDICES" \
  "$PARENT_PROBE_INDICES"; do
  test -s "$path" || {
    echo "[smc-em-followup][error] missing prerequisite: $path" >&2
    exit 2
  }
done
for marker in "$SMC_EM_OUT/DONE" "$PARENT_SMC_ROOT/pilot_selection/DONE"; do
  test -e "$marker" || {
    echo "[smc-em-followup][error] missing completion marker: $marker" >&2
    exit 2
  }
done

python - "$EM_SUMMARY" "$PARENT_SELECTION" "$SMC_EM_OUT/checkpoints/candidate.eqx" "$CANDIDATE_CHECKPOINT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

summary_path, parent_path, candidate_path, best_path = map(Path, sys.argv[1:])
payload = json.loads(summary_path.read_text(encoding="utf-8"))
if payload.get("selection_status") != "PASS":
    raise SystemExit("Direct SMC M-step selection did not pass")
if payload.get("selected_candidate") != "updated_prior":
    raise SystemExit("Direct SMC M-step did not select the updated prior")
parent = json.loads(parent_path.read_text(encoding="utf-8"))
if parent.get("selection_status") != "PASS":
    raise SystemExit("Parent SMC selection did not pass")
if parent.get("selected_variant") != "floor_0p05":
    raise SystemExit("Updated-prior confirmation is frozen to floor_0p05")
for path in (candidate_path, best_path):
    if not path.is_file():
        raise SystemExit(f"Missing selected checkpoint: {path}")
digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (candidate_path, best_path)]
if digests[0] != digests[1]:
    raise SystemExit("candidate.eqx and best.eqx differ after a PASS selection")
PY

confirmation_root="outputs/runs/posthoc_smc_${RUN_TAG}"
test ! -e "$confirmation_root" || {
  echo "[smc-em-followup][error] output already exists: $confirmation_root" >&2
  exit 2
}

# Build a completely new pair of cohorts. The generic pilot must not consume
# its progressive-parent mode here because this is an independent confirmation.
exclusion_parent_root="$PARENT_SMC_ROOT"
export REPO_DIR MINICONDA_PATH CONDA_ENV SOURCE_ROOT RUN_TAG
export OUTPUT_ROOT="$confirmation_root"
export LIMIT="$CONFIRM_OBJECTS"
export PROBE_LIMIT="$PROBE_OBJECTS"
export PARTICLES=1024
export OBJECT_BATCH_SIZE=4
export MALA_PARTICLE_CHUNK_SIZE=64
export TARGET_ESS_FRACTION=0.5
export MAX_STAGES=64
export MALA_STEPS=2
export MALA_STEP_SIZE=0.005
export N_SHARDS=$((CONFIRM_OBJECTS / OBJECTS_PER_SHARD))
export ARRAY_CONCURRENCY
export SMC_VARIANTS_CSV=floor_0p05
export SMC_SEEDS_CSV=260817,260818
export SMC_CHECKPOINT="$CANDIDATE_CHECKPOINT"
export SMC_FEATURE_STATS="$SOURCE_STATS"
export EXCLUDE_INDICES_CSV="$PARENT_SMC_INDICES,$PARENT_PROBE_INDICES"
export PARENT_SMC_ROOT=""

bash scripts/submit_popcosmos_posthoc_smc_pilot.sh
source outputs/logs/popcosmos_posthoc_smc_latest.env
confirmation_smc_job="$SMC_PILOT_JOB"
confirmation_finalizer_job="$SMC_FINALIZER_JOB"
confirmation_root="$SMC_OUTPUT_ROOT"

prior_confirmation_out="$confirmation_root/prior_confirmation"
confirm_raw=$(sbatch --parsable \
  --dependency="afterok:${confirmation_finalizer_job}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT",CANDIDATE_CHECKPOINT="$CANDIDATE_CHECKPOINT",SMC_ROOT="$confirmation_root",OUT="$prior_confirmation_out" \
  scripts/popcosmos_smc_empirical_bayes_confirm.slurm)
prior_confirmation_job="${confirm_raw%%;*}"

export SMC_ROOT="$confirmation_root"
export OUT="$confirmation_root/proposal_refresh_k2048"
export BASE_COMPONENTS=1
export REFRESH_EPOCHS="${REFRESH_EPOCHS:-20}"
export PROBE_SAMPLES="${PROBE_SAMPLES:-2048}"
export MIN_MEDIAN_ESS_FRACTION="${MIN_MEDIAN_ESS_FRACTION:-0.05}"
export MAX_FRACTION_PARETO_K_GT_0P7="${MAX_FRACTION_PARETO_K_GT_0P7:-0.2}"
export REFRESH_DEPENDENCY="$prior_confirmation_job"
export REFRESH_SOURCE_CHECKPOINT="$CANDIDATE_CHECKPOINT"
export REFRESH_FEATURE_STATS="$SOURCE_STATS"
export PRIOR_CONFIRMATION_SUMMARY="$prior_confirmation_out/prior_confirmation_summary.json"
export REFRESH_ROLE=post-em-fast-encoder-recovery
unset BASELINE_REFRESH_OUT

bash scripts/submit_popcosmos_posthoc_smc_refresh.sh
source outputs/logs/popcosmos_posthoc_smc_refresh_latest.env
refresh_job="$SMC_REFRESH_JOB"
refresh_out="$SMC_REFRESH_OUT"

receipt=outputs/logs/popcosmos_smc_empirical_bayes_followup_latest.env
printf 'export SMC_CONFIRM_JOB=%q\nexport SMC_CONFIRM_FINALIZER_JOB=%q\nexport PRIOR_CONFIRMATION_JOB=%q\nexport SMC_REFRESH_JOB=%q\nexport SMC_CONFIRM_ROOT=%q\nexport PRIOR_CONFIRMATION_OUT=%q\nexport SMC_REFRESH_OUT=%q\nexport SOURCE_ROOT=%q\nexport SOURCE_CHECKPOINT=%q\nexport CANDIDATE_CHECKPOINT=%q\nexport PARENT_SMC_ROOT=%q\nexport SMC_EM_OUT=%q\n' \
  "$confirmation_smc_job" "$confirmation_finalizer_job" \
  "$prior_confirmation_job" "$refresh_job" "$confirmation_root" \
  "$prior_confirmation_out" "$refresh_out" "$SOURCE_ROOT" \
  "$SOURCE_CHECKPOINT" "$CANDIDATE_CHECKPOINT" \
  "$exclusion_parent_root" "$SMC_EM_OUT" > "$receipt"

echo "smc_confirmation_job=$confirmation_smc_job"
echo "smc_finalizer_job=$confirmation_finalizer_job"
echo "prior_confirmation_job=$prior_confirmation_job"
echo "encoder_refresh_job=$refresh_job"
echo "smc_confirmation_root=$confirmation_root"
echo "prior_confirmation_out=$prior_confirmation_out"
echo "encoder_refresh_out=$refresh_out"
echo "receipt=$receipt"
echo "monitor: squeue -r -j $confirmation_smc_job,$confirmation_finalizer_job,$prior_confirmation_job,$refresh_job"
echo "report: python scripts/report_popcosmos_smc_empirical_bayes_followup.py --root $confirmation_root"
