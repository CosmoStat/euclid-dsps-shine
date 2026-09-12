#!/bin/bash
set -Eeuo pipefail
ROOT="$(realpath "${1:?AVI-next root}")"
PREFLIGHT=$(cat "$ROOT/preflight_job.txt")
TRAIN=$(tail -n 1 "$ROOT/train_job.txt")
AUDIT=$(cat "$ROOT/audit_job.txt")
while true; do
  clear
  date
  squeue -r -j "$PREFLIGHT,$TRAIN,$AUDIT" \
    -o "%.22i %.18T %.12M %N" || true
  sacct -X -j "$PREFLIGHT,$TRAIN,$AUDIT" \
    --format=JobID%22,State,Elapsed,ExitCode || true
  python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for stage in ("preflight", "arms"):
    for arm in ("H2_experts", "H8_experts", "P_latest_prior", "P_scratch_prior"):
        dest = root / stage / arm
        final, progress = dest / "FINAL.json", dest / "PROGRESS.json"
        if final.exists():
            data = json.loads(final.read_text())
            print(stage, arm, data.get("status"))
        elif progress.exists():
            data = json.loads(progress.read_text())
            print(stage, arm, data.get("stage"), f"{data.get('progress_percent', 0):.2f}%",
                  "ESS=", data.get("posterior_ess_median", data.get("ess")))
        else:
            print(stage, arm, "waiting")
audit = root / "EXPERT_AUDIT_COMPLETE.json"
print("expert audit:", json.loads(audit.read_text())["status"] if audit.exists() else "waiting")
PY
  echo "Ctrl-C stops only this display. Refresh in 30 seconds."
  sleep 30
done
