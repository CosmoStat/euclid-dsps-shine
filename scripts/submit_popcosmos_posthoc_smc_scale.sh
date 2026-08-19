#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_DIR/outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536/bands26/full_cont120}"
PARENT_SMC_ROOT="${PARENT_SMC_ROOT:?Set PARENT_SMC_ROOT to the completed previous tier}"
SCALE_OBJECTS="${SCALE_OBJECTS:-512}"
OBJECTS_PER_SHARD="${OBJECTS_PER_SHARD:-16}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-32}"
RUN_TAG="${RUN_TAG:-smc_floor05_teacher${SCALE_OBJECTS}_$(date +%Y%m%d_%H%M%S)}"

case "$SCALE_OBJECTS" in
  512|1024) ;;
  *)
    echo "[posthoc-smc-scale][error] SCALE_OBJECTS must be 512 or 1024" >&2
    exit 2
    ;;
esac
if (( SCALE_OBJECTS % OBJECTS_PER_SHARD != 0 )); then
  echo "[posthoc-smc-scale][error] SCALE_OBJECTS must be divisible by OBJECTS_PER_SHARD" >&2
  exit 2
fi

cd "$REPO_DIR"
for path in \
  "$SOURCE_ROOT/train/checkpoints/best.eqx" \
  "$SOURCE_ROOT/train/feature_stats.json" \
  "$PARENT_SMC_ROOT/pilot_selection/DONE" \
  "$PARENT_SMC_ROOT/pilot_selection/selection_summary.json" \
  "$PARENT_SMC_ROOT/cohorts/smc_calibration_indices.npy" \
  "$PARENT_SMC_ROOT/cohorts/proposal_probe_indices.npy"; do
  test -e "$path" || {
    echo "[posthoc-smc-scale][error] missing prerequisite: $path" >&2
    exit 2
  }
done

python - "$PARENT_SMC_ROOT/pilot_selection/selection_summary.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("selection_status") != "PASS":
    raise SystemExit("Parent SMC selection did not pass")
if payload.get("selected_variant") != "floor_0p05":
    raise SystemExit("Progressive scaling is frozen to floor_0p05")
PY

if [[ "$SCALE_OBJECTS" == 1024 ]]; then
  parent_decision="$PARENT_SMC_ROOT/proposal_refresh_k2048/refresh_validation_summary.json"
  test -s "$parent_decision" || {
    echo "[posthoc-smc-scale][error] missing 512-tier refresh decision: $parent_decision" >&2
    exit 2
  }
  python - "$parent_decision" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("encoder_refresh_gate") != "PASS":
    raise SystemExit("Parent encoder refresh did not pass")
if payload.get("ordinary_importance_support_gate") != "PASS":
    raise SystemExit("Parent ordinary-IS support did not pass; stop before 1024")
PY
fi

export REPO_DIR MINICONDA_PATH CONDA_ENV SOURCE_ROOT PARENT_SMC_ROOT RUN_TAG
export OUTPUT_ROOT="outputs/runs/posthoc_smc_${RUN_TAG}"
export LIMIT="$SCALE_OBJECTS"
export PROBE_LIMIT="$SCALE_OBJECTS"
export PARTICLES=1024
export OBJECT_BATCH_SIZE=4
export MALA_PARTICLE_CHUNK_SIZE=64
export TARGET_ESS_FRACTION=0.5
export MAX_STAGES=64
export MALA_STEPS=2
export MALA_STEP_SIZE=0.005
export N_SHARDS=$((SCALE_OBJECTS / OBJECTS_PER_SHARD))
export ARRAY_CONCURRENCY
export SMC_VARIANTS_CSV=floor_0p05
export SMC_SEEDS_CSV=260817,260818

bash scripts/submit_popcosmos_posthoc_smc_pilot.sh
source outputs/logs/popcosmos_posthoc_smc_latest.env

scale_pilot_job="$SMC_PILOT_JOB"
scale_finalizer_job="$SMC_FINALIZER_JOB"
scale_root="$SMC_OUTPUT_ROOT"

export SMC_ROOT="$scale_root"
export OUT="$scale_root/proposal_refresh_k2048"
export BASE_COMPONENTS=1
export REFRESH_EPOCHS="${REFRESH_EPOCHS:-20}"
export PROBE_SAMPLES="${PROBE_SAMPLES:-2048}"
export MIN_MEDIAN_ESS_FRACTION="${MIN_MEDIAN_ESS_FRACTION:-0.05}"
export MAX_FRACTION_PARETO_K_GT_0P7="${MAX_FRACTION_PARETO_K_GT_0P7:-0.2}"
export REFRESH_DEPENDENCY="$scale_finalizer_job"
unset BASELINE_REFRESH_OUT

bash scripts/submit_popcosmos_posthoc_smc_refresh.sh
source outputs/logs/popcosmos_posthoc_smc_refresh_latest.env

receipt=outputs/logs/popcosmos_posthoc_smc_scale_latest.env
printf 'export SMC_SCALE_OBJECTS=%q\nexport SMC_PILOT_JOB=%q\nexport SMC_FINALIZER_JOB=%q\nexport SMC_REFRESH_JOB=%q\nexport SMC_OUTPUT_ROOT=%q\nexport SMC_REFRESH_OUT=%q\nexport SOURCE_ROOT=%q\nexport PARENT_SMC_ROOT=%q\n' \
  "$SCALE_OBJECTS" "$scale_pilot_job" "$scale_finalizer_job" \
  "$SMC_REFRESH_JOB" "$scale_root" "$SMC_REFRESH_OUT" \
  "$SOURCE_ROOT" "$PARENT_SMC_ROOT" > "$receipt"

echo "smc_scale_objects=$SCALE_OBJECTS"
echo "smc_pilot_job=$scale_pilot_job"
echo "smc_finalizer_job=$scale_finalizer_job"
echo "smc_refresh_job=$SMC_REFRESH_JOB"
echo "smc_output_root=$scale_root"
echo "smc_refresh_out=$SMC_REFRESH_OUT"
echo "receipt=$receipt"
echo "monitor: squeue -r -j $scale_pilot_job,$scale_finalizer_job,$SMC_REFRESH_JOB"
echo "report: python scripts/report_popcosmos_posthoc_smc_scale.py --root $scale_root"
