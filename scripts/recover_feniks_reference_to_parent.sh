#!/bin/bash
# Keep completed banks/classifier; freeze corrected code, refit weights and report.
set -Eeuo pipefail
REPO=$(pwd -P)
ROOT=$(realpath "${1:?existing reference-to-parent run}")
command -v sbatch >/dev/null
command -v squeue >/dev/null
test -s "$ROOT/JOBS.env"
test -s "$ROOT/INPUT.env"
exec 9>"$ROOT/.recovery.lock"
flock -n 9 || { echo 'Another recovery submission is active' >&2; exit 1; }
source "$ROOT/JOBS.env"
LIVE=$(squeue -h -u "${USER:?}" -o '%i')
: "${ALL_JOBS:?}"
for job in ${ALL_JOBS//,/ }; do
  [[ $job =~ ^[0-9]+$ ]] || { echo 'Invalid saved job ID' >&2; exit 1; }
  if awk -v j="$job" '{split($1,a,"_"); if(a[1]==j) found=1} END {exit !found}' <<< "$LIVE"; then
    echo "Job $job still active; wait before recovery." >&2; exit 1
  fi
done
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export EUCLID_DSPS_REQUIRE_GPU=0 EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLBACKEND=Agg
CHECK=$(python -m scripts.feniks_reference_to_parent check-recovery --root "$ROOT")
mkdir -p "$ROOT/recovery"
RECOVERY=$(mktemp -d "$ROOT/recovery/population_$(date +%Y%m%d_%H%M%S)_XXXXXX")
printf '%s\n' "$CHECK" > "$RECOVERY/PRESERVED.json"
for name in JOBS.env INPUT.env CODE_SHA256; do
  cp "$ROOT/$name" "$RECOVERY/$name.previous"
done
ARCHIVE="$RECOVERY/code.tar"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" euclid_dsps scripts configs pyproject.toml
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/reference-to-parent-$DIGEST"
mkdir -p "$CODE"
tar -xf "$ARCHIVE" -C "$CODE"
[[ -e "$CODE/Data" ]] || ln -s "$REPO/Data" "$CODE/Data"
printf '%s\n' "$DIGEST" > "$RECOVERY/CODE_SHA256"
printf '%s\n' "$CODE" > "$RECOVERY/CODE_DIR"
printf '%s\n' "$DIGEST" > "$ROOT/CODE_SHA256.next"
mv "$ROOT/CODE_SHA256.next" "$ROOT/CODE_SHA256"
printf 'export FENIKS_RTP_CODE=%q\n' "$CODE" > "$ROOT/INPUT.next.env"
mv "$ROOT/INPUT.next.env" "$ROOT/INPUT.env"
echo 'Completed banks and stopped classifier retained. KKT tolerance remains 2e-6.'
bash scripts/submit_feniks_reference_to_parent.sh --resume "$ROOT"
cp "$ROOT/JOBS.env" "$RECOVERY/JOBS.submitted.env"
