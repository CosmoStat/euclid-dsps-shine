#!/bin/bash
set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
SOURCE_ENV="${1:-outputs/logs/feniks_sc_drws_balanced_npe_latest.env}"
source "$SOURCE_ENV"
REPO_DIR="$(pwd -P)"
SOURCE_ROOT="${BALANCED_ROOT:?balanced experiment environment required}"
export DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:-$(dirname "$SOURCE_ROOT")/frozen_parent_local_vi_diagnostic_v1}"
export CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?}/feniks_sc_drws_runtime}"
export MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?}/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-shine}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
command -v git >/dev/null || { echo 'git missing on submission host'; exit 2; }
command -v sbatch >/dev/null || { echo 'sbatch missing: submit from Jean-Zay front end'; exit 2; }
git diff --quiet
git diff --cached --quiet
COMMIT="$(git rev-parse HEAD)"
EXTRA=()
WALLTIME="03:00:00"
if [[ "${LOCAL_VI_GRADIENT_ISOLATION:-0}" == "1" ]]; then
  EXTRA+=(--gradient-isolation)
  WALLTIME="00:20:00"
fi
if [[ "${LOCAL_VI_REDSHIFT_DECOMPOSITION:-0}" == "1" ]]; then
  EXTRA+=(--redshift-decomposition)
  WALLTIME="00:45:00"
fi
if [[ "${LOCAL_VI_PHOTOMETRY_REFERENCE:-0}" == "1" ]]; then
  EXTRA+=(--photometry-reference)
  WALLTIME="00:45:00"
fi
if [[ -n "${LOCAL_VI_FULL_DECODER_REFERENCE:-}" ]]; then
  EXTRA+=(--full-decoder-reference "$LOCAL_VI_FULL_DECODER_REFERENCE")
  WALLTIME="01:30:00"
fi
if [[ -n "${LOCAL_VI_MDF_PRECISION_REFERENCE:-}" ]]; then
  EXTRA+=(--mdf-precision-reference "$LOCAL_VI_MDF_PRECISION_REFERENCE")
  WALLTIME="01:30:00"
fi
if [[ -n "${LOCAL_VI_TARGET_RESOLUTION_REFERENCE:-}" ]]; then
  EXTRA+=(--target-resolution-reference "$LOCAL_VI_TARGET_RESOLUTION_REFERENCE")
  WALLTIME="00:45:00"
fi
if [[ -n "${LOCAL_VI_REDSHIFT_PRECISION_REFERENCE:-}" ]]; then
  EXTRA+=(--redshift-precision-reference "$LOCAL_VI_REDSHIFT_PRECISION_REFERENCE")
  WALLTIME="00:45:00"
fi
if [[ -n "${LOCAL_VI_PRECISION_NIGHT_REFERENCE:-}" ]]; then
  EXTRA+=(--precision-night-reference "$LOCAL_VI_PRECISION_NIGHT_REFERENCE")
  WALLTIME="10:00:00"
fi
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu EUCLID_DSPS_REQUIRE_GPU=0 \
  python scripts/run_feniks_sc_drws_local_vi_diagnostic.py prepare \
  --source-root "$SOURCE_ROOT" --root "$DIAGNOSTIC_ROOT" \
  --objects "${LOCAL_VI_OBJECTS:-16}" --steps "${LOCAL_VI_STEPS:-64}" \
  --draws "${LOCAL_VI_DRAWS:-128}" "${EXTRA[@]}"
SNAPSHOT="$CACHE_ROOT/code/local-vi-diagnostic-${COMMIT:0:12}"
export DIAGNOSTIC_LOG_ROOT="$CACHE_ROOT/slurm_logs/$(basename "$DIAGNOSTIC_ROOT")"
mkdir -p "$(dirname "$SNAPSHOT")" "$DIAGNOSTIC_LOG_ROOT" outputs/logs
if [[ ! -e "$SNAPSHOT" ]]; then
  git worktree add --detach "$SNAPSHOT" "$COMMIT"
fi
test "$(git -C "$SNAPSHOT" rev-parse HEAD)" = "$COMMIT"
git -C "$SNAPSHOT" diff --quiet
git -C "$SNAPSHOT" diff --cached --quiet
if [[ ! -e "$SNAPSHOT/Data/diffsky" ]]; then
  mkdir -p "$SNAPSHOT/Data"
  ln -s "$REPO_DIR/Data/diffsky" "$SNAPSHOT/Data/diffsky"
fi
LOCAL_REPO="$REPO_DIR"
export REPO_DIR="$SNAPSHOT"
touch "$DIAGNOSTIC_ROOT/SUBMISSION_INFLIGHT"
JOB="$(sbatch --parsable --export=ALL \
  --time="$WALLTIME" \
  --output="$DIAGNOSTIC_LOG_ROOT/diagnostic-%j.out" \
  --error="$DIAGNOSTIC_LOG_ROOT/diagnostic-%j.err" \
  "$REPO_DIR/scripts/feniks_sc_drws_local_vi_diagnostic_h100.slurm")"
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo 'Ambiguous submission: inspect squeue, do not resubmit'; exit 1; }
export DIAGNOSTIC_JOB="$JOB"
export DIAGNOSTIC_ENV="$LOCAL_REPO/outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env"
python - <<'PY'
import json
import os
from pathlib import Path
import shlex
root = Path(os.environ['DIAGNOSTIC_ROOT'])
payload = {key: os.environ[key] for key in ('DIAGNOSTIC_ROOT','DIAGNOSTIC_LOG_ROOT','DIAGNOSTIC_JOB','REPO_DIR')}
manifest = json.loads((root/'RUN_MANIFEST.json').read_text())
(root/'SUBMISSION.json').write_text(json.dumps({'status':'SUBMITTED', 'jobs':payload, 'gpus':1, 'nodes':1, 'gpu_hours_ceiling':manifest['allocation_gpu_hours'], 'mode':manifest['mode']}, indent=2)+'\n')
body = ''.join(f'export {key}={shlex.quote(value)}\n' for key,value in payload.items())
(root/'submission.env').write_text(body)
Path(os.environ['DIAGNOSTIC_ENV']).write_text(body)
(root/'SUBMISSION_INFLIGHT').unlink()
PY
echo "diagnostic_job=$DIAGNOSTIC_JOB"
echo "root=$DIAGNOSTIC_ROOT"
echo "monitor: bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh"
