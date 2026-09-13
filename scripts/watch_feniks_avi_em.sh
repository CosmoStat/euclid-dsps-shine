#!/bin/bash
# Usage: watch_feniks_avi_em.sh EM_ROOT [refresh_seconds]
set -Eeuo pipefail

ROOT="$(realpath "${1:?EM root}")"
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
  echo "===== EM CYCLES ====="
  python - "$ROOT" "$CYCLES" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for cycle in range(1, int(sys.argv[2]) + 1):
    base = root / "cycles" / f"cycle_{cycle:02d}"
    final = base / "FINAL.json"
    if final.exists():
        value = json.loads(final.read_text())
        print(
            f"cycle {cycle}: {value['status']} "
            f"alpha={value.get('selection_log_alpha')} "
            f"ESS={value.get('posterior_ess_median')}"
        )
        continue
    printed = False
    for stage, arm in (("mstep", f"P_em_{cycle:02d}"), ("qstep", f"Q_em_{cycle:02d}")):
        folder = base / stage
        for phase in ("arms", "preflight"):
            progress = folder / phase / arm / "PROGRESS.json"
            receipt = folder / phase / arm / "FINAL.json"
            if receipt.exists():
                value = json.loads(receipt.read_text())
                print(f"cycle {cycle} {stage} {phase}: {value.get('status')}")
                printed = True
            elif progress.exists():
                value = json.loads(progress.read_text())
                print(
                    f"cycle {cycle} {stage} {phase}: {value.get('stage')} "
                    f"{value.get('progress_percent', 0):.2f}%"
                )
                printed = True
    if not printed:
        print(f"cycle {cycle}: waiting")
PY

  echo
  echo "===== INFERENCE ====="
  if [[ -s "$AVI_EM_INFERENCE_ROOT/MANIFEST.json" ]]; then
    python - "$AVI_EM_INFERENCE_ROOT" <<'PY'
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
        print(variant["name"], value.get("stage"), f"{value.get('progress_percent', 0):.2f}%")
    else:
        print(variant["name"], "waiting")
report = root / "report/FINAL.json"
print("report", json.loads(report.read_text()).get("status") if report.exists() else "waiting")
PY
  else
    echo "inference preparation waiting"
  fi
  echo "Ctrl-C stops only this display. Refresh in $INTERVAL seconds."
  sleep "$INTERVAL"
done
