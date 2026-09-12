#!/bin/bash
# Usage: bash scripts/submit_feniks_avi_inference.sh TRAINING_ROOT NEW_ROOT [concurrency]
set -Eeuo pipefail
TRAINING_ROOT=$(realpath "${1:?training root}")
export INFERENCE_ROOT=$(realpath -m "${2:?new inference root}")
CONCURRENCY=${3:-8}
[[ "$CONCURRENCY" =~ ^[1-8]$ ]]
test ! -e "$INFERENCE_ROOT"
REPO=$(pwd -P)
test -d filters
ARCHIVE="$INFERENCE_ROOT.code.tar"
test ! -e "$ARCHIVE"
mkdir -p "$(dirname "$INFERENCE_ROOT")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export AVI_CODE="$SCRATCH/feniks_sc_drws_runtime/code/avi-inference-$DIGEST"
mkdir -p "$AVI_CODE"
tar -xf "$ARCHIVE" -C "$AVI_CODE"
if [[ ! -e "$AVI_CODE/Data" && ! -L "$AVI_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$AVI_CODE/Data"
fi
cd "$AVI_CODE"
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m scripts.feniks_avi_inference prepare --training "$TRAINING_ROOT" --root "$INFERENCE_ROOT"
mkdir -p "$INFERENCE_ROOT/logs"
printf '%s\n' "$AVI_CODE" > "$INFERENCE_ROOT/CODE_DIR"
export AVI_MODE=infer
JOB=$(sbatch --parsable --array="0-7%$CONCURRENCY" --export=ALL \
  --output="$INFERENCE_ROOT/logs/infer-%A_%a.out" --error="$INFERENCE_ROOT/logs/infer-%A_%a.err" scripts/feniks_avi_inference.slurm)
printf '%s\n' "${JOB%%;*}" > "$INFERENCE_ROOT/inference_job.txt"
export AVI_MODE=mira
MIRA=$(sbatch --parsable --dependency="afterok:${JOB%%;*}" --gres=gpu:1 --cpus-per-task=12 --export=ALL \
  --output="$INFERENCE_ROOT/logs/mira-%j.out" --error="$INFERENCE_ROOT/logs/mira-%j.err" scripts/feniks_avi_inference.slurm)
printf '%s\n' "${MIRA%%;*}" > "$INFERENCE_ROOT/mira_job.txt"
printf 'inference=%s\nmira=%s\nroot=%s\nGPUs_per_task=4 peak_GPUs=%s\n' "$JOB" "$MIRA" "$INFERENCE_ROOT" "$((4*CONCURRENCY))"
