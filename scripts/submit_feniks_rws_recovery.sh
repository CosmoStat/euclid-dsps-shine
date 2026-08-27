#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
RUN_TAG="${RUN_TAG:-feniks_sc_drws_r27p5_$(date +%Y%m%d_%H%M%S)}"
RECOVERY_ROOT="${RECOVERY_ROOT:-${SCRATCH:?Set SCRATCH}/$RUN_TAG}"
SMOKE_ROOT="${RECOVERY_ROOT}_smoke"
MANIFEST_ROOT="$RECOVERY_ROOT/manifests"
SMOKE_MANIFEST_ROOT="$SMOKE_ROOT/manifests"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH}/feniks_rws_recovery_runtime}"
LOG_ROOT="${LOG_ROOT:-$CACHE_ROOT/slurm_logs/$RUN_TAG}"
TRAIN_CATALOG="$CATALOG_DIR/train.parquet"
TEST_CATALOG="$CATALOG_DIR/test.parquet"

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs
for path in "$TRAIN_CATALOG" "$TEST_CATALOG" \
  configs/experiments/feniks_sc_drws_r27p5_historical.yaml \
  configs/experiments/feniks_sc_drws_r27p5_current.yaml \
  scripts/feniks_rws_recovery_pilot_h100.slurm \
  scripts/feniks_rws_recovery_confirm_h100.slurm; do
  test -s "$path" || { echo "[rws-recovery-submit][error] missing: $path" >&2; exit 2; }
done
test ! -e "$RECOVERY_ROOT" || { echo "output exists: $RECOVERY_ROOT" >&2; exit 2; }
test ! -e "$SMOKE_ROOT" || { echo "smoke output exists: $SMOKE_ROOT" >&2; exit 2; }

JAX_PLATFORMS=cpu python scripts/build_feniks_rws_recovery_manifests.py \
  --train-catalog "$TRAIN_CATALOG" --test-catalog "$TEST_CATALOG" \
  --out "$MANIFEST_ROOT" --validation-objects 614 --pilot-objects 512 \
  --confirmation-objects 2000 --seed 260826
JAX_PLATFORMS=cpu python scripts/build_feniks_rws_recovery_manifests.py \
  --train-catalog "$TRAIN_CATALOG" --test-catalog "$TEST_CATALOG" \
  --out "$SMOKE_MANIFEST_ROOT" --validation-objects 64 --pilot-objects 128 \
  --confirmation-objects 16 --seed 260826

EXPORTS="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,CATALOG_DIR=$CATALOG_DIR,CACHE_ROOT=$CACHE_ROOT"
SMOKE_RAW=$(sbatch --parsable --array=0-3%4 --time=00:30:00 \
  --output="$LOG_ROOT/smoke-%A_%a.out" --error="$LOG_ROOT/smoke-%A_%a.err" \
  --export="$EXPORTS,SMOKE=1,RECOVERY_ROOT=$SMOKE_ROOT,MANIFEST_ROOT=$SMOKE_MANIFEST_ROOT" \
  scripts/feniks_rws_recovery_pilot_h100.slurm)
SMOKE_JOB="${SMOKE_RAW%%;*}"
PILOT_RAW=$(sbatch --parsable --dependency="afterok:$SMOKE_JOB" \
  --output="$LOG_ROOT/pilot-%A_%a.out" --error="$LOG_ROOT/pilot-%A_%a.err" \
  --export="$EXPORTS,SMOKE=0,RECOVERY_ROOT=$RECOVERY_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT" \
  scripts/feniks_rws_recovery_pilot_h100.slurm)
PILOT_JOB="${PILOT_RAW%%;*}"
PILOT_GATE_RAW=$(sbatch --parsable --dependency="afterok:$PILOT_JOB" \
  --output="$LOG_ROOT/pilot-gate-%j.out" --error="$LOG_ROOT/pilot-gate-%j.err" \
  --export="$EXPORTS,RECOVERY_ROOT=$RECOVERY_ROOT" \
  scripts/feniks_rws_recovery_pilot_finalize.slurm)
PILOT_GATE_JOB="${PILOT_GATE_RAW%%;*}"
CONFIRM_RAW=$(sbatch --parsable --dependency="afterok:$PILOT_GATE_JOB" \
  --output="$LOG_ROOT/confirm-%A_%a.out" --error="$LOG_ROOT/confirm-%A_%a.err" \
  --export="$EXPORTS,RECOVERY_ROOT=$RECOVERY_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT" \
  scripts/feniks_rws_recovery_confirm_h100.slurm)
CONFIRM_JOB="${CONFIRM_RAW%%;*}"
FINAL_RAW=$(sbatch --parsable --dependency="afterok:$CONFIRM_JOB" \
  --output="$LOG_ROOT/final-gate-%j.out" --error="$LOG_ROOT/final-gate-%j.err" \
  --export="$EXPORTS,RECOVERY_ROOT=$RECOVERY_ROOT" \
  scripts/feniks_rws_recovery_confirm_finalize.slurm)
FINAL_JOB="${FINAL_RAW%%;*}"

LATEST=outputs/logs/feniks_rws_recovery_latest.env
printf 'export SMOKE_JOB=%q\nexport PILOT_JOB=%q\nexport PILOT_GATE_JOB=%q\nexport CONFIRM_JOB=%q\nexport FINAL_JOB=%q\nexport RECOVERY_ROOT=%q\nexport MANIFEST_ROOT=%q\nexport CACHE_ROOT=%q\nexport LOG_ROOT=%q\n' \
  "$SMOKE_JOB" "$PILOT_JOB" "$PILOT_GATE_JOB" "$CONFIRM_JOB" "$FINAL_JOB" \
  "$RECOVERY_ROOT" "$MANIFEST_ROOT" "$CACHE_ROOT" "$LOG_ROOT" > "$LATEST"

echo "smoke_job=$SMOKE_JOB"
echo "pilot_job=$PILOT_JOB"
echo "pilot_gate_job=$PILOT_GATE_JOB"
echo "confirmation_job=$CONFIRM_JOB"
echo "final_gate_job=$FINAL_JOB"
echo "recovery_root=$RECOVERY_ROOT"
echo "resources=4 pilot tasks x 4 H100, then 2 independent confirmation tasks x 4 H100"
echo "full_dataset_not_submitted=1"
echo "monitor: source $LATEST && squeue -r -j $SMOKE_JOB,$PILOT_JOB,$PILOT_GATE_JOB,$CONFIRM_JOB,$FINAL_JOB"
echo "detailed_monitor: bash scripts/monitor_feniks_rws_recovery.sh"
echo "latest_env=$LATEST"
