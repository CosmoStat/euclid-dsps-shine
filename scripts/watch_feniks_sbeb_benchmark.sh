#!/bin/bash
# Usage: watch_feniks_sbeb_benchmark.sh ROOT [refresh_seconds]
set -Eeuo pipefail

ROOT="$(realpath "${1:?SBEB root}")"
INTERVAL="${2:-30}"
source "$ROOT/JOBS.env"

while true; do
  clear || true
  date
  echo "===== SLURM ====="
  squeue -r -j "$ALL_JOBS" -o "%.24i %.18T %.12M %R" || true
  sacct -n -X -j "$ALL_JOBS" \
    --format=JobID%24,State%18,Elapsed,ExitCode | sed '/^[[:space:]]*$/d' || true

  echo
  echo "===== SCRATCH BOOTSTRAP ====="
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for cut in (25, 27, 29):
    arm = root / f"bootstrap/r{cut}/arms/scratch_r{cut}"
    final = arm / "FINAL.json"
    progress = arm / "PROGRESS.json"
    if final.exists():
        print(f"r<{cut}", json.loads(final.read_text()).get("status"))
    elif progress.exists():
        value = json.loads(progress.read_text())
        print(f"r<{cut}", value.get("stage"), f"{value.get('progress_percent', 0):.2f}%")
    else:
        print(f"r<{cut} waiting")
PY

  echo
  echo "===== FACTORIAL ====="
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
complete = 0
for cell in manifest["factor_cells"]:
    arm = root / "factor/cells" / cell["name"] / "arms" / f"P_{cell['name']}"
    if (arm / "FINAL.json").exists():
        complete += 1
print(f"prior fits {complete}/{len(manifest['factor_cells'])}")
inferred = sum(
    (root / f"factor/inference/r{cut}/arms" / cell["name"] / "FINAL.json").exists()
    for cut in (25, 27, 29)
    for cell in manifest["factor_cells"]
    if cell["cut"] == cut
)
print(f"inference {inferred}/{len(manifest['factor_cells'])}")
report = root / "factor/report/FINAL.json"
print("report", json.loads(report.read_text()).get("status") if report.exists() else "waiting")
PY

  echo
  echo "===== EM TRAJECTORIES ====="
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
for track in manifest["trajectory_tracks"]:
    base = root / "trajectories" / track["name"]
    cycles = sum(
        (base / "em/cycles" / f"cycle_{cycle:02d}/FINAL.json").exists()
        for cycle in range(1, int(manifest["cycles"]) + 1)
    )
    inferred = sum(
        (base / "inference/arms" / f"cycle_{cycle:02d}/FINAL.json").exists()
        for cycle in range(0, int(manifest["cycles"]) + 1)
    )
    nuts = root / "nuts" / track["name"] / "inference/FINAL.json"
    factorial = base / "endpoint_factorial/report/FACTORIAL_FINAL.json"
    print(
        track["name"],
        f"cycles={cycles}/{manifest['cycles']}",
        f"inference={inferred}/{int(manifest['cycles']) + 1}",
        f"factorial={'done' if factorial.exists() else 'waiting'}",
        f"nuts={'done' if nuts.exists() else 'waiting'}",
    )
final = root / "report/FINAL.json"
print("final report", json.loads(final.read_text()).get("status") if final.exists() else "waiting")
PY
  echo "Ctrl-C stops only this display. Refresh in $INTERVAL seconds."
  sleep "$INTERVAL"
done
