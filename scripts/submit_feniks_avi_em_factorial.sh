#!/bin/bash
# Usage: submit_feniks_avi_em_factorial.sh EM_ROOT FACTORIAL_ROOT [concurrency]
set -Eeuo pipefail
trap 'echo "AVI factorial submission stopped at line $LINENO: $BASH_COMMAND" >&2' ERR

EM_ROOT="$(realpath "${1:?completed AVI EM root}")"
export AVI_FACTORIAL_ROOT="$(realpath -m "${2:?new factorial inference root}")"
CONCURRENCY="${3:-4}"
[[ "$CONCURRENCY" =~ ^[1-4]$ ]] || {
  echo "concurrency must lie between 1 and 4" >&2
  exit 2
}
test ! -e "$AVI_FACTORIAL_ROOT"
test -s "$EM_ROOT/MANIFEST.json"

REPO="$(pwd -P)"
test -d "$REPO/filters"
ARCHIVE="$AVI_FACTORIAL_ROOT.code.tar"
test ! -e "$ARCHIVE"
mkdir -p "$(dirname "$AVI_FACTORIAL_ROOT")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export AVI_FACTORIAL_CODE="$SCRATCH/feniks_sc_drws_runtime/code/avi-factorial-$DIGEST"
mkdir -p "$AVI_FACTORIAL_CODE"
tar -xf "$ARCHIVE" -C "$AVI_FACTORIAL_CODE"
if [[ ! -e "$AVI_FACTORIAL_CODE/Data" && ! -L "$AVI_FACTORIAL_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$AVI_FACTORIAL_CODE/Data"
fi
cd "$AVI_FACTORIAL_CODE"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python -m scripts.feniks_avi_em_factorial prepare \
    --em-root "$EM_ROOT" --root "$AVI_FACTORIAL_ROOT" --particles 4096
printf '%s\n' "$AVI_FACTORIAL_CODE" > "$AVI_FACTORIAL_ROOT/CODE_DIR"

export AVI_FACTORIAL_MODE=infer
RAW=$(sbatch --parsable --time=04:00:00 \
  --array="0-3%${CONCURRENCY}" --export=ALL \
  --output="$AVI_FACTORIAL_ROOT/logs/infer-%A_%a.out" \
  --error="$AVI_FACTORIAL_ROOT/logs/infer-%A_%a.err" \
  "$AVI_FACTORIAL_CODE/scripts/feniks_avi_em_factorial.slurm")
INFERENCE_JOB="${RAW%%;*}"

export AVI_FACTORIAL_MODE=report
RAW=$(sbatch --parsable --time=03:00:00 --gres=gpu:1 --cpus-per-task=24 \
  --dependency="afterok:$INFERENCE_JOB" --export=ALL \
  --output="$AVI_FACTORIAL_ROOT/logs/report-%j.out" \
  --error="$AVI_FACTORIAL_ROOT/logs/report-%j.err" \
  "$AVI_FACTORIAL_CODE/scripts/feniks_avi_em_factorial.slurm")
REPORT_JOB="${RAW%%;*}"

ALL_JOBS="$INFERENCE_JOB,$REPORT_JOB"
printf 'export AVI_FACTORIAL_ROOT=%q\nexport AVI_FACTORIAL_CODE=%q\nexport INFERENCE_JOB=%q\nexport REPORT_JOB=%q\nexport ALL_JOBS=%q\n' \
  "$AVI_FACTORIAL_ROOT" "$AVI_FACTORIAL_CODE" "$INFERENCE_JOB" \
  "$REPORT_JOB" "$ALL_JOBS" > "$AVI_FACTORIAL_ROOT/JOBS.env"

printf 'inference=%s\nreport=%s\nroot=%s\npeak_h100s=%s\nwatch=bash scripts/watch_feniks_avi_em_factorial.sh %q\n' \
  "$INFERENCE_JOB" "$REPORT_JOB" "$AVI_FACTORIAL_ROOT" \
  "$((4 * CONCURRENCY))" "$AVI_FACTORIAL_ROOT"
