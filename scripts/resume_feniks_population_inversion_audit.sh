#!/bin/bash
# Usage: bash scripts/resume_feniks_population_inversion_audit.sh EXISTING_ROOT
set -Eeuo pipefail

export FENIKS_POPULATION_INVERSION_ROOT=$(realpath "${1:?existing audit root}")
REPO=$(pwd -P)
test -s "$FENIKS_POPULATION_INVERSION_ROOT/JOBS.env"
test -d "$REPO/filters"
exec 9>"$FENIKS_POPULATION_INVERSION_ROOT/.recovery.lock"
flock -n 9 || { echo 'Another recovery submission is active' >&2; exit 2; }
source "$FENIKS_POPULATION_INVERSION_ROOT/JOBS.env"

declare -A CANCEL=()
while read -r id state; do
  [[ -n "$id" ]] || continue
  base=${id%%_*}
  if [[ "$state" == RUNNING ]]; then
    echo "Refusing recovery while old job $id is RUNNING" >&2
    exit 2
  fi
  [[ "$base" =~ ^[0-9]+$ ]] && CANCEL["$base"]=1
done < <(squeue -h -j "${ALL_JOBS:?}" -o '%i %T' || true)

mapfile -t RESOURCE < <(python - "$FENIKS_POPULATION_INVERSION_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
for arm in manifest["arms"]:
    if (root / arm / "FINAL.json").exists():
        raise SystemExit(f"{arm} is already final; refuse mixed recovery")
    if not (root / arm / "checkpoint_stability.csv").is_file():
        raise SystemExit(f"{arm} did not reach the recoverable checkpoint")
    completed = all(
        (root / arm / name).is_file()
        for name in ("regularization_path.csv", "bootstrap.csv", "selected_weights.csv")
    )
    if not completed:
        raise SystemExit(f"{arm} is missing completed path/bootstrap tables")
r = manifest["settings"]["resources"]
for key in ("arm_hours", "report_hours", "arm_concurrency"):
    value = r[key]
    assert isinstance(value, int) and value > 0
    print(value)
PY
)
ARM_HOURS=${RESOURCE[0]}
REPORT_HOURS=${RESOURCE[1]}
ARM_CONCURRENCY=${RESOURCE[2]}

TAG=$(date +%Y%m%d_%H%M%S)
RECOVERY="$FENIKS_POPULATION_INVERSION_ROOT/recovery/$TAG"
mkdir -p "$RECOVERY"
cp "$FENIKS_POPULATION_INVERSION_ROOT/JOBS.env" "$RECOVERY/JOBS.previous.env"
cp "$FENIKS_POPULATION_INVERSION_ROOT/CODE_DIR" "$RECOVERY/CODE_DIR.previous"
cp "$FENIKS_POPULATION_INVERSION_ROOT/CODE_SHA256" "$RECOVERY/CODE_SHA256.previous"

ARCHIVE="$RECOVERY/code.tar"
tar --exclude='__pycache__' --exclude='*.pyc' \
  -cf "$ARCHIVE" euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export FENIKS_POPULATION_INVERSION_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/population-inversion-$DIGEST"
mkdir -p "$FENIKS_POPULATION_INVERSION_CODE"
tar -xf "$ARCHIVE" -C "$FENIKS_POPULATION_INVERSION_CODE"
[[ -e "$FENIKS_POPULATION_INVERSION_CODE/Data" ]] || \
  ln -s "$REPO/Data" "$FENIKS_POPULATION_INVERSION_CODE/Data"

if [[ ${#CANCEL[@]} -gt 0 ]]; then
  scancel "${!CANCEL[@]}"
fi
printf '%s\n' "$FENIKS_POPULATION_INVERSION_CODE" \
  > "$FENIKS_POPULATION_INVERSION_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FENIKS_POPULATION_INVERSION_ROOT/CODE_SHA256"

export FENIKS_POPULATION_INVERSION_MODE=finalize
RAW=$(sbatch --parsable --time="1:00:00" \
  --array="0-1%$ARM_CONCURRENCY" --export=ALL \
  --output="$FENIKS_POPULATION_INVERSION_ROOT/logs/arm-%A_%a.out" \
  --error="$FENIKS_POPULATION_INVERSION_ROOT/logs/arm-%A_%a.err" \
  "$FENIKS_POPULATION_INVERSION_CODE/scripts/feniks_population_inversion_audit.slurm")
ARM_JOB=${RAW%%;*}

export FENIKS_POPULATION_INVERSION_MODE=report
RAW=$(sbatch --parsable --time="$REPORT_HOURS:00:00" \
  --dependency="afterok:$ARM_JOB" --export=ALL \
  --output="$FENIKS_POPULATION_INVERSION_ROOT/logs/report-%j.out" \
  --error="$FENIKS_POPULATION_INVERSION_ROOT/logs/report-%j.err" \
  "$FENIKS_POPULATION_INVERSION_CODE/scripts/feniks_population_inversion_audit.slurm")
REPORT_JOB=${RAW%%;*}
ALL_JOBS="$ARM_JOB,$REPORT_JOB"
printf 'export ARM_JOB=%q\nexport REPORT_JOB=%q\nexport ALL_JOBS=%q\n' \
  "$ARM_JOB" "$REPORT_JOB" "$ALL_JOBS" \
  > "$RECOVERY/JOBS.env"
cp "$RECOVERY/JOBS.env" "$FENIKS_POPULATION_INVERSION_ROOT/JOBS.next.env"
mv "$FENIKS_POPULATION_INVERSION_ROOT/JOBS.next.env" \
  "$FENIKS_POPULATION_INVERSION_ROOT/JOBS.env"

echo "arms=$ARM_JOB report=$REPORT_JOB"
echo 'Reused completed regularization paths and bootstraps; only receipts and report are rerun.'
printf 'watch=bash scripts/watch_feniks_population_inversion_audit.sh %q\n' \
  "$FENIKS_POPULATION_INVERSION_ROOT"
