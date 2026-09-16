#!/bin/bash
# Usage: submit...sh TRAINING_ROOT NEW_ROOT [concurrency]
# Resume: submit...sh resume ROOT [array]
set -Eeuo pipefail
trap 'echo "AVI-next submission stopped at line $LINENO: $BASH_COMMAND" >&2' ERR
if [[ "${1:-}" = resume ]]; then
  export AVI_NEXT_ROOT="$(realpath "${2:?existing root}")"
  export AVI_NEXT_CODE="$(cat "$AVI_NEXT_ROOT/CODE_DIR")"
  ARRAY="${3:-0-3%4}"
  test -s "$AVI_NEXT_ROOT/MANIFEST.json"
  export AVI_NEXT_MODE=train
  JOB=$(sbatch --parsable --array="$ARRAY" --export=ALL \
    --output="$AVI_NEXT_ROOT/logs/train-%A_%a.out" \
    --error="$AVI_NEXT_ROOT/logs/train-%A_%a.err" \
    "$AVI_NEXT_CODE/scripts/feniks_avi_next.slurm")
  printf '%s\n' "${JOB%%;*}" >> "$AVI_NEXT_ROOT/train_job.txt"
  printf 'resume_array=%s\nroot=%s\n' "$JOB" "$AVI_NEXT_ROOT"
  exit 0
fi
TRAINING_ROOT="$(realpath "${1:?completed avi_encoder_experiments root}")"
export AVI_NEXT_ROOT="$(realpath -m "${2:?new output root}")"
CONCURRENCY="${3:-4}"
[[ "$CONCURRENCY" =~ ^[1-4]$ ]] || { echo 'concurrency must be 1..4'; exit 1; }
test ! -e "$AVI_NEXT_ROOT"
REPO="$(pwd -P)"
test -d "$REPO/filters"
ARCHIVE="$AVI_NEXT_ROOT.code.tar"
test ! -e "$ARCHIVE"
mkdir -p "$(dirname "$AVI_NEXT_ROOT")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export AVI_NEXT_CODE="$SCRATCH/feniks_sc_drws_runtime/code/avi-next-$DIGEST"
mkdir -p "$AVI_NEXT_CODE"
tar -xf "$ARCHIVE" -C "$AVI_NEXT_CODE"
if [[ ! -e "$AVI_NEXT_CODE/Data" && ! -L "$AVI_NEXT_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$AVI_NEXT_CODE/Data"
fi
cd "$AVI_NEXT_CODE"
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m scripts.feniks_avi_next_experiments prepare \
  --training "$TRAINING_ROOT" --root "$AVI_NEXT_ROOT"
printf '%s\n' "$AVI_NEXT_CODE" > "$AVI_NEXT_ROOT/CODE_DIR"
export AVI_NEXT_MODE=preflight
PREFLIGHT=$(sbatch --parsable --time=02:00:00 --array="0-3%$CONCURRENCY" --export=ALL \
  --output="$AVI_NEXT_ROOT/logs/preflight-%A_%a.out" \
  --error="$AVI_NEXT_ROOT/logs/preflight-%A_%a.err" \
  "$AVI_NEXT_CODE/scripts/feniks_avi_next.slurm")
printf '%s\n' "${PREFLIGHT%%;*}" > "$AVI_NEXT_ROOT/preflight_job.txt"
export AVI_NEXT_MODE=train
TRAIN=$(sbatch --parsable --array="0-3%$CONCURRENCY" \
  --dependency="afterok:${PREFLIGHT%%;*}" --export=ALL \
  --output="$AVI_NEXT_ROOT/logs/train-%A_%a.out" \
  --error="$AVI_NEXT_ROOT/logs/train-%A_%a.err" \
  "$AVI_NEXT_CODE/scripts/feniks_avi_next.slurm")
printf '%s\n' "${TRAIN%%;*}" > "$AVI_NEXT_ROOT/train_job.txt"
AUDIT=$(sbatch --parsable --dependency="afterok:${TRAIN%%;*}" --export=ALL \
  --output="$AVI_NEXT_ROOT/logs/audit-%j.out" \
  --error="$AVI_NEXT_ROOT/logs/audit-%j.err" \
  "$AVI_NEXT_CODE/scripts/feniks_avi_expert_audit.slurm")
printf '%s\n' "${AUDIT%%;*}" > "$AVI_NEXT_ROOT/audit_job.txt"
printf 'preflight_array=%s\ntraining_array=%s\naudit_job=%s\nroot=%s\nGPUs_per_array_task=4\npeak_GPUs=%s\n' \
  "$PREFLIGHT" "$TRAIN" "$AUDIT" "$AVI_NEXT_ROOT" "$((4*CONCURRENCY))"
