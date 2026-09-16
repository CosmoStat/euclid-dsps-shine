#!/bin/bash
# Usage: submit...sh BASE NEW_ROOT [concurrency]; resume: ... resume ROOT [array]
set -Eeuo pipefail
trap 'echo "AVI submission stopped at line $LINENO: $BASH_COMMAND" >&2' ERR
if [[ "${1:-}" = resume ]]; then
  export AVI_ROOT="$(realpath "${2:?existing root}")"
  export AVI_CODE="$(cat "$AVI_ROOT/CODE_DIR")" AVI_MODE=train
  ARRAY="${3:-0-6%7}"
  test -s "$AVI_ROOT/MANIFEST.json"
  JOB=$(sbatch --parsable --array="$ARRAY" --export=ALL \
    --output="$AVI_ROOT/logs/train-%A_%a.out" --error="$AVI_ROOT/logs/train-%A_%a.err" \
    "$AVI_CODE/scripts/feniks_avi_experiments.slurm")
  printf '%s\n' "${JOB%%;*}" >> "$AVI_ROOT/train_jobs.txt"
  printf 'resume_array=%s\nroot=%s\n' "$JOB" "$AVI_ROOT"
  exit 0
fi
BASE="$(realpath "${1:?FENIKS base directory}")"
export AVI_ROOT="$(realpath -m "${2:?NEW output directory}")"
CONCURRENCY="${3:-7}"
[[ "$CONCURRENCY" =~ ^[1-7]$ ]] || { echo 'concurrency must be 1..7'; exit 1; }
test ! -e "$AVI_ROOT"
REPO="$(pwd -P)"
test -d "$REPO/filters"
ARCHIVE="$AVI_ROOT.code.tar"
test ! -e "$ARCHIVE"
mkdir -p "$(dirname "$AVI_ROOT")"
# Snapshot actual runtime files, including any local runtime changes.
# Generated artifacts, user documentation and unrelated notebooks are excluded.
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" \
  euclid_dsps scripts configs pyproject.toml
# Copy curve contents even when filters/ is a site-local symlink.
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export AVI_CODE="$SCRATCH/feniks_sc_drws_runtime/code/avi-$DIGEST"
mkdir -p "$AVI_CODE"
tar -xf "$ARCHIVE" -C "$AVI_CODE"
if [[ ! -e "$AVI_CODE/Data" && ! -L "$AVI_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$AVI_CODE/Data"
fi
cd "$AVI_CODE"
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m scripts.feniks_avi_experiments prepare --root "$AVI_ROOT" \
  --reference "$BASE/frozen_parent_wake_holdout_v1" \
  --manifest-root "${MANIFEST_ROOT:-$BASE/manifests}" \
  --nuts-root "$BASE/frozen_geometry_nuts_observed8_dense_depth6_v1"
printf '%s\n' "$AVI_CODE" > "$AVI_ROOT/CODE_DIR"
export AVI_MODE=preflight
PREFLIGHT=$(sbatch --parsable --time=02:00:00 --array="0-6%$CONCURRENCY" --export=ALL \
  --output="$AVI_ROOT/logs/preflight-%A_%a.out" --error="$AVI_ROOT/logs/preflight-%A_%a.err" \
  "$AVI_CODE/scripts/feniks_avi_experiments.slurm")
printf '%s\n' "${PREFLIGHT%%;*}" > "$AVI_ROOT/preflight_job.txt"
export AVI_MODE=train
TRAIN=$(sbatch --parsable --array="0-6%$CONCURRENCY" --dependency="afterok:${PREFLIGHT%%;*}" --export=ALL \
  --output="$AVI_ROOT/logs/train-%A_%a.out" --error="$AVI_ROOT/logs/train-%A_%a.err" \
  "$AVI_CODE/scripts/feniks_avi_experiments.slurm")
printf '%s\n' "${TRAIN%%;*}" > "$AVI_ROOT/train_jobs.txt"
printf 'preflight_array=%s\ntraining_array=%s\nroot=%s\nGPUs_per_task=4 peak_GPUs=%s\n' \
  "$PREFLIGHT" "$TRAIN" "$AVI_ROOT" "$((4*CONCURRENCY))"
