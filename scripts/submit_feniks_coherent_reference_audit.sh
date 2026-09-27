#!/bin/bash
# SOURCE_INFERENCE NEW_AUDIT_ROOT, or --resume EXISTING_AUDIT_ROOT
set -Eeuo pipefail
command -v sbatch >/dev/null
REPO=$(pwd -P)
if [[ ${1:-} == --resume ]]; then
  export FENIKS_CRA_ROOT=$(realpath "${2:?existing audit root}")
  source "$FENIKS_CRA_ROOT/INPUT.env"
  if [[ -f "$FENIKS_CRA_ROOT/JOBS.env" ]]; then
    source "$FENIKS_CRA_ROOT/JOBS.env"
    ACTIVE=$(squeue -h -j "$AUDIT_JOB" -o '%i' 2>/dev/null || true)
    [[ -z "$ACTIVE" ]] || { echo "Job still active: $ACTIVE" >&2; exit 1; }
  fi
else
  export FENIKS_CRA_SOURCE=$(realpath "${1:?completed coherent inference root}")
  export FENIKS_CRA_ROOT=$(realpath -m "${2:?new audit root}")
  test -f "$FENIKS_CRA_SOURCE/report/FINAL.json"
  test -f "$FENIKS_CRA_SOURCE/reference/basis.npz"
  test ! -e "$FENIKS_CRA_ROOT"
  test ! -e "$FENIKS_CRA_ROOT.code.tar"
  mkdir -p "$FENIKS_CRA_ROOT/logs"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf "$FENIKS_CRA_ROOT.code.tar" euclid_dsps scripts configs pyproject.toml
  DIGEST=$(sha256sum "$FENIKS_CRA_ROOT.code.tar" | cut -d' ' -f1)
  export FENIKS_CRA_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/coherent-reference-audit-$DIGEST"
  mkdir -p "$FENIKS_CRA_CODE"
  tar -xf "$FENIKS_CRA_ROOT.code.tar" -C "$FENIKS_CRA_CODE"
  [[ -e "$FENIKS_CRA_CODE/Data" ]] || ln -s "$REPO/Data" "$FENIKS_CRA_CODE/Data"
  printf '%s\n' "$DIGEST" > "$FENIKS_CRA_ROOT/CODE_SHA256"
  printf 'export FENIKS_CRA_SOURCE=%q\nexport FENIKS_CRA_CODE=%q\n' \
    "$FENIKS_CRA_SOURCE" "$FENIKS_CRA_CODE" > "$FENIKS_CRA_ROOT/INPUT.env"
fi
export FENIKS_CRA_SOURCE FENIKS_CRA_CODE
if [[ -f "$FENIKS_CRA_ROOT/report/FINAL.json" ]]; then
  echo "Audit already completed: $FENIKS_CRA_ROOT"; exit 0
fi
ATTEMPT=$(date +%Y%m%d_%H%M%S)
RAW=$(sbatch --parsable --account="${FENIKS_CPU_ACCOUNT:-jrx@cpu}" \
  --partition="${FENIKS_CPU_PARTITION:-cpu_p1}" --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --hint=nomultithread --time=00:30:00 \
  --job-name=feniks_reference_audit --export=ALL \
  --output="$FENIKS_CRA_ROOT/logs/audit-$ATTEMPT-%j.out" \
  --error="$FENIKS_CRA_ROOT/logs/audit-$ATTEMPT-%j.err" \
  "$FENIKS_CRA_CODE/scripts/feniks_coherent_reference_audit.slurm")
AUDIT_JOB=${RAW%%;*}
[[ $AUDIT_JOB =~ ^[0-9]+$ ]]
printf 'export AUDIT_JOB=%q\n' "$AUDIT_JOB" > "$FENIKS_CRA_ROOT/JOBS.env"
printf 'export AUDIT=%q\n' "$FENIKS_CRA_ROOT" > "$(dirname "$FENIKS_CRA_ROOT")/avi_coherent_reference_audit_latest.env"
echo "audit=$AUDIT_JOB; CPU only, 4 threads, 30 minute ceiling, no training or DSPS."
printf 'watch=bash scripts/watch_feniks_coherent_reference_audit.sh %q\n' "$FENIKS_CRA_ROOT"
