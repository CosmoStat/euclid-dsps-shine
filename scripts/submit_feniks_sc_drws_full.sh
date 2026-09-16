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
ALLOW_UNCONFIRMED_FULL="${ALLOW_UNCONFIRMED_FULL:-0}"
cd "$REPO_DIR"
PASS_RECEIPT="$RECOVERY_ROOT/RWS_RECOVERY_PASS.json"
AUTHORIZATION_RECEIPT="$PASS_RECEIPT"
if [[ -s "$PASS_RECEIPT" ]]; then
  python - "$PASS_RECEIPT" <<'PY'
import json, sys

receipt = json.load(open(sys.argv[1]))
if receipt.get("status") != "PASS" or not receipt.get("ready_for_full_catalogue"):
    raise SystemExit("confirmation did not authorize full training")
PY
elif [[ "$ALLOW_UNCONFIRMED_FULL" == "1" ]]; then
  SELECTED_CANDIDATE="${SELECTED_CANDIDATE:?Set SELECTED_CANDIDATE for explicit unconfirmed full training}"
  case "$SELECTED_CANDIDATE" in
    historical_4x128|current_residual_6x256) ;;
    *) echo "invalid SELECTED_CANDIDATE=$SELECTED_CANDIDATE" >&2; exit 2 ;;
  esac
  AUTHORIZATION_RECEIPT="$RECOVERY_ROOT/FULL_LAUNCH_AUTHORIZATION.json"
  python - "$AUTHORIZATION_RECEIPT" "$SELECTED_CANDIDATE" \
    "$(git rev-parse HEAD)" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": "EXPLICIT_UNCONFIRMED_FULL_OVERRIDE",
    "selected_candidate": sys.argv[2],
    "git_commit": sys.argv[3],
    "reason": "population-first full profile differs from the legacy pilot profile",
    "pilot_thresholds_used_as_training_gate": False,
    "truth_used_for_training_or_authorization": False,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
else
  echo "confirmation gate missing; set ALLOW_UNCONFIRMED_FULL=1 and SELECTED_CANDIDATE explicitly" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT" outputs/logs "$CACHE_ROOT/jax"
EXPORTS="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,CATALOG_DIR=$CATALOG_DIR,RECOVERY_ROOT=$RECOVERY_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT,CACHE_ROOT=$CACHE_ROOT,FULL_AUTHORIZATION_RECEIPT=$AUTHORIZATION_RECEIPT"
RAW=$(sbatch --parsable --array=0 --output="$LOG_ROOT/full-%A_%a.out" \
  --error="$LOG_ROOT/full-%A_%a.err" --export="$EXPORTS" scripts/feniks_sc_drws_full_h100.slurm)
FULL_JOB="${RAW%%;*}"
GATE_RAW=$(sbatch --parsable --dependency="afterok:$FULL_JOB" \
  --output="$LOG_ROOT/full-gate-%j.out" --error="$LOG_ROOT/full-gate-%j.err" \
  --export="$EXPORTS" scripts/feniks_sc_drws_full_finalize.slurm)
FULL_GATE_JOB="${GATE_RAW%%;*}"
LATEST=outputs/logs/feniks_sc_drws_full_latest.env
printf 'export FULL_JOB=%q\nexport FULL_GATE_JOB=%q\nexport RECOVERY_ROOT=%q\nexport MANIFEST_ROOT=%q\nexport LOG_ROOT=%q\nexport FULL_AUTHORIZATION_RECEIPT=%q\n' \
  "$FULL_JOB" "$FULL_GATE_JOB" "$RECOVERY_ROOT" "$MANIFEST_ROOT" "$LOG_ROOT" \
  "$AUTHORIZATION_RECEIPT" > "$LATEST"
echo "full_job=$FULL_JOB"
echo "full_gate_job=$FULL_GATE_JOB"
echo "latest_env=$LATEST"
echo "authorization_receipt=$AUTHORIZATION_RECEIPT"
echo "monitor: bash scripts/monitor_feniks_sc_drws_full.sh"
echo "final_inference_not_submitted=1"
