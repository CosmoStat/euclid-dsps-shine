#!/bin/bash
# Usage: submit_feniks_avi_next_validation.sh OVERNIGHT_ROOT NUTS_ROOT NEW_ROOT
set -Eeuo pipefail
trap 'echo "AVI validation submission stopped at line $LINENO: $BASH_COMMAND" >&2' ERR

export AVI_OVERNIGHT_ROOT="$(realpath "${1:?completed overnight inference root}")"
export AVI_NUTS_ROOT="$(realpath "${2:?completed observed-eight NUTS root}")"
export AVI_VALIDATION_ROOT="$(realpath -m "${3:?new validation root}")"
test ! -e "$AVI_VALIDATION_ROOT"

REPO="$(pwd -P)"
test -d "$REPO/filters"
ARCHIVE="$AVI_VALIDATION_ROOT.code.tar"
test ! -e "$ARCHIVE"
mkdir -p "$(dirname "$AVI_VALIDATION_ROOT")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export AVI_VALIDATION_CODE="$SCRATCH/feniks_sc_drws_runtime/code/avi-validation-$DIGEST"
mkdir -p "$AVI_VALIDATION_CODE"
tar -xf "$ARCHIVE" -C "$AVI_VALIDATION_CODE"
if [[ ! -e "$AVI_VALIDATION_CODE/Data" && ! -L "$AVI_VALIDATION_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$AVI_VALIDATION_CODE/Data"
fi
cd "$AVI_VALIDATION_CODE"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m scripts.feniks_avi_next_validation prepare \
  --overnight "$AVI_OVERNIGHT_ROOT" --nuts-root "$AVI_NUTS_ROOT" \
  --root "$AVI_VALIDATION_ROOT"
printf '%s\n' "$AVI_VALIDATION_CODE" > "$AVI_VALIDATION_ROOT/CODE_DIR"
mkdir -p "$AVI_VALIDATION_ROOT/logs"

export AVI_VALIDATION_MODE=preflight
PREFLIGHT=$(sbatch --parsable --time=01:00:00 --export=ALL \
  --output="$AVI_VALIDATION_ROOT/logs/preflight-%j.out" \
  --error="$AVI_VALIDATION_ROOT/logs/preflight-%j.err" \
  "$AVI_VALIDATION_CODE/scripts/feniks_avi_next_validation.slurm")
PREFLIGHT=${PREFLIGHT%%;*}

export AVI_VALIDATION_MODE=infer
INFERENCE=$(sbatch --parsable --time=03:00:00 \
  --dependency="afterok:$PREFLIGHT" --export=ALL \
  --output="$AVI_VALIDATION_ROOT/logs/inference-%j.out" \
  --error="$AVI_VALIDATION_ROOT/logs/inference-%j.err" \
  "$AVI_VALIDATION_CODE/scripts/feniks_avi_next_validation.slurm")
INFERENCE=${INFERENCE%%;*}

export AVI_VALIDATION_MODE=audit
AUDIT=$(sbatch --parsable --time=03:00:00 \
  --dependency="afterok:$INFERENCE" --export=ALL \
  --output="$AVI_VALIDATION_ROOT/logs/audit-%j.out" \
  --error="$AVI_VALIDATION_ROOT/logs/audit-%j.err" \
  "$AVI_VALIDATION_CODE/scripts/feniks_avi_next_validation.slurm")
AUDIT=${AUDIT%%;*}

export AVI_VALIDATION_MODE=report
REPORT=$(sbatch --parsable --time=02:00:00 \
  --dependency="afterok:$AUDIT" --export=ALL \
  --output="$AVI_VALIDATION_ROOT/logs/report-%j.out" \
  --error="$AVI_VALIDATION_ROOT/logs/report-%j.err" \
  "$AVI_VALIDATION_CODE/scripts/feniks_avi_next_validation.slurm")
REPORT=${REPORT%%;*}

printf 'export AVI_OVERNIGHT_ROOT=%q\nexport AVI_NUTS_ROOT=%q\nexport AVI_VALIDATION_ROOT=%q\nexport AVI_VALIDATION_CODE=%q\nexport PREFLIGHT_JOB=%q\nexport INFERENCE_JOB=%q\nexport AUDIT_JOB=%q\nexport REPORT_JOB=%q\n' \
  "$AVI_OVERNIGHT_ROOT" "$AVI_NUTS_ROOT" "$AVI_VALIDATION_ROOT" \
  "$AVI_VALIDATION_CODE" "$PREFLIGHT" "$INFERENCE" "$AUDIT" "$REPORT" \
  > "$AVI_VALIDATION_ROOT/JOBS.env"

printf 'preflight=%s\ninference=%s\naudit=%s\nreport=%s\nroot=%s\npeak_h100s=4\nwatch=bash scripts/watch_feniks_avi_next_validation.sh %q\n' \
  "$PREFLIGHT" "$INFERENCE" "$AUDIT" "$REPORT" \
  "$AVI_VALIDATION_ROOT" "$AVI_VALIDATION_ROOT"
