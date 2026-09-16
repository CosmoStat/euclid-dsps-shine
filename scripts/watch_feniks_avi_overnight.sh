#!/bin/bash
# Usage: watch_feniks_avi_overnight.sh TRAIN_ROOT [refresh_seconds]
set -Eeuo pipefail
ROOT="$(realpath "${1:?q-refresh root}")"
INTERVAL="${2:-30}"
source "$ROOT/JOBS.env"

while true; do
  clear || true
  date
  echo "===== SLURM ====="
  squeue -r -j "$PREFLIGHT_JOB,$TRAIN_JOB,$PREPARE_JOB,$INFERENCE_JOB,$REPORT_JOB" \
    -o "%.22i %.18T %.12M %R" || true
  sacct -n -X -j "$PREFLIGHT_JOB,$TRAIN_JOB,$PREPARE_JOB,$INFERENCE_JOB,$REPORT_JOB" \
    --format=JobID%22,State%18,Elapsed,ExitCode | sed '/^[[:space:]]*$/d' || true

  echo
  echo "===== Q REFRESH ====="
  python - "$AVI_TRAIN_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
for stage in ("preflight", "arms"):
    for arm in ("Q_source_refresh", "Q_latest_refresh"):
        progress = root / stage / arm / "PROGRESS.json"
        final = root / stage / arm / "FINAL.json"
        if final.exists():
            data = json.loads(final.read_text())
            print(stage, arm, data.get("status"))
        elif progress.exists():
            data = json.loads(progress.read_text())
            print(stage, arm, data.get("stage"), f"{data.get('progress_percent', 0):.2f}%", "loss=", data.get("loss"))
        else:
            print(stage, arm, "waiting")
PY

  echo
  echo "===== INFERENCE ====="
  if [[ -s "$AVI_INFERENCE_ROOT/MANIFEST.json" ]]; then
    python - "$AVI_INFERENCE_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
for item in manifest["variants"]:
    arm = root / "arms" / item["name"]
    final, progress = arm / "FINAL.json", arm / "PROGRESS.json"
    if final.exists():
        print(item["name"], json.loads(final.read_text()).get("status"))
    elif progress.exists():
        data = json.loads(progress.read_text())
        print(item["name"], data.get("stage"), f"{data.get('progress_percent', 0):.2f}%")
    else:
        print(item["name"], "waiting")
report = root / "report/FINAL.json"
print("report", json.loads(report.read_text()).get("status") if report.exists() else "waiting")
PY
  else
    echo "inference preparation waiting"
  fi
  echo "Ctrl-C stops only this display. Refresh in $INTERVAL seconds."
  sleep "$INTERVAL"
done
