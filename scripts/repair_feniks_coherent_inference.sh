#!/bin/bash
# Repair the failed pre-reference run using a NEW code snapshot, then resume it.
set -Eeuo pipefail
REPO=$(pwd -P)
ROOT=$(realpath "${1:?existing coherent inference root}")
command -v sbatch >/dev/null
command -v squeue >/dev/null
ALL_JOBS=''
[[ ! -f "$ROOT/JOBS.env" ]] || source "$ROOT/JOBS.env"
if [[ -n "$ALL_JOBS" ]]; then
  # Query by user rather than completed IDs; a scheduler error must fail closed.
  LIVE=$(squeue --noheader --user="$(id -un)" --format='%i')
  for job in ${ALL_JOBS//,/ }; do
    if [[ ! $job =~ ^[0-9]+$ ]]; then
      echo "Invalid saved job ID: $job" >&2; exit 1
    fi
    if awk -v j="$job" '{split($1,a,"_"); if(a[1]==j) found=1} END {exit !found}' <<< "$LIVE"; then
      echo "Job $job still active; wait before repairing." >&2; exit 1
    fi
  done
fi
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m scripts.repair_feniks_coherent_inference "$ROOT" --check
TAG=$(date +%Y%m%d_%H%M%S)
ARCHIVE="$ROOT.repair_$TAG.code.tar"
test ! -e "$ARCHIVE"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" euclid_dsps scripts configs pyproject.toml
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/coherent-inference-$DIGEST"
mkdir -p "$CODE"
tar -xf "$ARCHIVE" -C "$CODE"
[[ -e "$CODE/Data" ]] || ln -s "$REPO/Data" "$CODE/Data"
python -m scripts.repair_feniks_coherent_inference "$ROOT" --code "$CODE" --archive "$ARCHIVE"
bash scripts/submit_feniks_coherent_inference.sh --resume "$ROOT"
