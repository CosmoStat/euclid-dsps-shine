#!/bin/bash
set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
RECOVERY_ROOT="${RECOVERY_ROOT:?Set RECOVERY_ROOT to the passed confirmation root}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$RECOVERY_ROOT/manifests}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
LOG_ROOT="${LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$RECOVERY_ROOT")-full}"
cd "$REPO_DIR"
test -s "$RECOVERY_ROOT/RWS_RECOVERY_PASS.json" || { echo "confirmation gate missing" >&2; exit 2; }
python - "$RECOVERY_ROOT/RWS_RECOVERY_PASS.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if r.get("status") != "PASS" or not r.get("ready_for_full_catalogue"):
    raise SystemExit("confirmation did not authorize full training")
PY
mkdir -p "$LOG_ROOT" outputs/logs "$CACHE_ROOT/jax"
EXPORTS="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,CATALOG_DIR=$CATALOG_DIR,RECOVERY_ROOT=$RECOVERY_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT,CACHE_ROOT=$CACHE_ROOT"
RAW=$(sbatch --parsable --array=0 --output="$LOG_ROOT/full-%A_%a.out" \
  --error="$LOG_ROOT/full-%A_%a.err" --export="$EXPORTS" scripts/feniks_sc_drws_full_h100.slurm)
FULL_JOB="${RAW%%;*}"
GATE_RAW=$(sbatch --parsable --dependency="afterok:$FULL_JOB" \
  --output="$LOG_ROOT/full-gate-%j.out" --error="$LOG_ROOT/full-gate-%j.err" \
  --export="$EXPORTS" scripts/feniks_sc_drws_full_finalize.slurm)
FULL_GATE_JOB="${GATE_RAW%%;*}"
LATEST=outputs/logs/feniks_sc_drws_full_latest.env
printf 'export FULL_JOB=%q\nexport FULL_GATE_JOB=%q\nexport RECOVERY_ROOT=%q\nexport MANIFEST_ROOT=%q\nexport LOG_ROOT=%q\n' \
  "$FULL_JOB" "$FULL_GATE_JOB" "$RECOVERY_ROOT" "$MANIFEST_ROOT" "$LOG_ROOT" > "$LATEST"
echo "full_job=$FULL_JOB"
echo "full_gate_job=$FULL_GATE_JOB"
echo "latest_env=$LATEST"
echo "monitor: bash scripts/monitor_feniks_sc_drws_full.sh"
echo "final_inference_not_submitted=1"
