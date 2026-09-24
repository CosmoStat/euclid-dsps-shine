#!/bin/bash
# Resume a failed ratio-convergence array in its existing experiment root.
# Usage: bash scripts/resume_feniks_ratio_convergence.sh EXISTING_ROOT
set -Eeuo pipefail

export FENIKS_RATIO_CONVERGENCE_ROOT=$(realpath "${1:?existing convergence root}")
REPO=$(pwd -P)
test -s "$FENIKS_RATIO_CONVERGENCE_ROOT/JOBS.env"
test -d "$REPO/filters"
exec 9>"$FENIKS_RATIO_CONVERGENCE_ROOT/.recovery.lock"
flock -n 9 || { echo 'Another recovery submission is active' >&2; exit 2; }
source "$FENIKS_RATIO_CONVERGENCE_ROOT/JOBS.env"

if [[ -f "$FENIKS_RATIO_CONVERGENCE_ROOT/RECOVERY_PENDING" ]]; then
  echo 'An incomplete recovery submission exists; inspect RECOVERY_PENDING' >&2
  exit 2
fi

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

mapfile -t RESOURCE < <(
  JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu \
    EUCLID_DSPS_REQUIRE_GPU=0 JAX_ENABLE_X64=true \
    python - "$FENIKS_RATIO_CONVERGENCE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
cfg = manifest["settings"]
start = cfg["classifier"]["epochs"]
for arm in manifest["arms"]:
    out = root / arm
    if (out / "FINAL.json").exists():
        raise SystemExit(f"{arm} is already final; refuse mixed partial recovery")
    resume = json.loads((out / "RESUME.json").read_text())
    if int(resume["epoch"]) < start:
        raise SystemExit(f"{arm} checkpoint precedes the source epoch")
    if not (out / "best.eqx").is_file():
        raise SystemExit(f"{arm} is missing best.eqx")
    trajectory = out / "trajectory.csv"
    if not trajectory.is_file():
        raise SystemExit(f"{arm} is missing its saved epoch-{start} diagnostic")
resources = cfg["convergence"]["resources"]
for key in ("classifier_hours", "report_hours", "classifier_concurrency"):
    value = resources[key]
    if not isinstance(value, int) or value <= 0:
        raise SystemExit(f"invalid resource {key}")
    print(value)
PY
)
[[ ${#RESOURCE[@]} == 3 ]] || { echo 'Recovery validation failed' >&2; exit 2; }
CLASSIFIER_HOURS=${RESOURCE[0]}
REPORT_HOURS=${RESOURCE[1]}
CLASSIFIER_CONCURRENCY=${RESOURCE[2]}

TAG=$(date +%Y%m%d_%H%M%S)
RECOVERY="$FENIKS_RATIO_CONVERGENCE_ROOT/recovery/$TAG"
mkdir -p "$RECOVERY"
cp "$FENIKS_RATIO_CONVERGENCE_ROOT/JOBS.env" "$RECOVERY/JOBS.previous.env"
cp "$FENIKS_RATIO_CONVERGENCE_ROOT/CODE_DIR" "$RECOVERY/CODE_DIR.previous"
cp "$FENIKS_RATIO_CONVERGENCE_ROOT/CODE_SHA256" "$RECOVERY/CODE_SHA256.previous"

ARCHIVE="$RECOVERY/code.tar"
tar --exclude='__pycache__' --exclude='*.pyc' \
  -cf "$ARCHIVE" euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export FENIKS_RATIO_CONVERGENCE_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/ratio-convergence-$DIGEST"
mkdir -p "$FENIKS_RATIO_CONVERGENCE_CODE"
tar -xf "$ARCHIVE" -C "$FENIKS_RATIO_CONVERGENCE_CODE"
if [[ ! -e "$FENIKS_RATIO_CONVERGENCE_CODE/Data" && ! -L "$FENIKS_RATIO_CONVERGENCE_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$FENIKS_RATIO_CONVERGENCE_CODE/Data"
fi

printf '%s\n' "$RECOVERY" > "$FENIKS_RATIO_CONVERGENCE_ROOT/RECOVERY_PENDING"
if [[ ${#CANCEL[@]} -gt 0 ]]; then
  scancel "${!CANCEL[@]}"
fi
printf '%s\n' "$FENIKS_RATIO_CONVERGENCE_CODE" \
  > "$FENIKS_RATIO_CONVERGENCE_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FENIKS_RATIO_CONVERGENCE_ROOT/CODE_SHA256"

: > "$RECOVERY/JOBS.env"
export FENIKS_RATIO_CONVERGENCE_MODE=run
RAW=$(sbatch --parsable --time="$CLASSIFIER_HOURS:00:00" \
  --array="0-1%$CLASSIFIER_CONCURRENCY" --export=ALL \
  --output="$FENIKS_RATIO_CONVERGENCE_ROOT/logs/classifier-%A_%a.out" \
  --error="$FENIKS_RATIO_CONVERGENCE_ROOT/logs/classifier-%A_%a.err" \
  "$FENIKS_RATIO_CONVERGENCE_CODE/scripts/feniks_ratio_convergence.slurm")
CLASSIFIER_JOB=${RAW%%;*}

export FENIKS_RATIO_CONVERGENCE_MODE=report
RAW=$(sbatch --parsable --time="$REPORT_HOURS:00:00" \
  --dependency="afterok:$CLASSIFIER_JOB" --export=ALL \
  --output="$FENIKS_RATIO_CONVERGENCE_ROOT/logs/report-%j.out" \
  --error="$FENIKS_RATIO_CONVERGENCE_ROOT/logs/report-%j.err" \
  "$FENIKS_RATIO_CONVERGENCE_CODE/scripts/feniks_ratio_convergence.slurm")
REPORT_JOB=${RAW%%;*}
ALL_JOBS="$CLASSIFIER_JOB,$REPORT_JOB"
printf 'export CLASSIFIER_JOB=%q\nexport REPORT_JOB=%q\nexport ALL_JOBS=%q\n' \
  "$CLASSIFIER_JOB" "$REPORT_JOB" "$ALL_JOBS" > "$RECOVERY/JOBS.env"
cp "$RECOVERY/JOBS.env" "$FENIKS_RATIO_CONVERGENCE_ROOT/JOBS.next.env"
mv "$FENIKS_RATIO_CONVERGENCE_ROOT/JOBS.next.env" \
  "$FENIKS_RATIO_CONVERGENCE_ROOT/JOBS.env"
rm "$FENIKS_RATIO_CONVERGENCE_ROOT/RECOVERY_PENDING"

echo "classifier=$CLASSIFIER_JOB report=$REPORT_JOB"
echo 'Reused both epoch-200 checkpoints, histories, splits and simulation banks.'
printf 'watch=bash scripts/watch_feniks_ratio_convergence.sh %q\n' \
  "$FENIKS_RATIO_CONVERGENCE_ROOT"
