#!/bin/bash
# Usage: watch_feniks_avi_em_factorial.sh FACTORIAL_ROOT [refresh_seconds]
set -Eeuo pipefail

ROOT="$(realpath "${1:?factorial root}")"
INTERVAL="${2:-30}"
source "$ROOT/JOBS.env"

while true; do
  clear || true
  date
  echo "===== SLURM ====="
  squeue -r -j "$ALL_JOBS" -o "%.22i %.18T %.12M %R" || true
  sacct -n -X -j "$ALL_JOBS" \
    --format=JobID%22,State%18,Elapsed,ExitCode | sed '/^[[:space:]]*$/d' || true

  echo
  echo "===== FACTORIAL ====="
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
for variant in manifest["variants"]:
    arm = root / "arms" / variant["name"]
    final = arm / "FINAL.json"
    progress = arm / "PROGRESS.json"
    if final.exists():
        print(variant["name"], json.loads(final.read_text()).get("status"))
    elif progress.exists():
        value = json.loads(progress.read_text())
        print(
            variant["name"],
            value.get("stage"),
            f"{value.get('progress_percent', 0):.2f}%",
        )
    else:
        print(variant["name"], "waiting")
report = root / "report/FACTORIAL_FINAL.json"
print(
    "report",
    json.loads(report.read_text()).get("status") if report.exists() else "waiting",
)
PY
  echo "Ctrl-C stops only this display. Refresh in $INTERVAL seconds."
  sleep "$INTERVAL"
done
