#!/bin/bash
# Usage: watch_feniks_sbeb_continuation.sh CONTINUATION_ROOT [refresh_seconds]
set -Eeuo pipefail

ROOT="$(realpath "${1:?SBEB continuation root}")"
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
  echo "===== CONTINUATION ====="
  python - "$ROOT" "$CYCLES" $TRACKS <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cycles = int(sys.argv[2])
tracks = sys.argv[3:]

for track in tracks:
    em = root / track / "em"
    completed = 0
    details = "waiting"
    for cycle in range(1, cycles + 1):
        cycle_root = em / "cycles" / f"cycle_{cycle:02d}"
        final = cycle_root / "FINAL.json"
        if final.is_file():
            value = json.loads(final.read_text())
            completed += 1
            details = (
                f"alpha={value.get('selection_log_alpha')} "
                f"ESS={value.get('posterior_ess_median')}"
            )
            continue
        for stage, arm in (
            ("mstep", f"P_em_{cycle:02d}"),
            ("qstep", f"Q_em_{cycle:02d}"),
        ):
            base = cycle_root / stage
            progress = base / "arms" / arm / "PROGRESS.json"
            receipt = base / "arms" / arm / "FINAL.json"
            if receipt.is_file():
                details = f"cycle {cycle} {stage} complete"
            elif progress.is_file():
                value = json.loads(progress.read_text())
                details = (
                    f"cycle {cycle} {stage} {value.get('stage')} "
                    f"{value.get('progress_percent', 0):.2f}%"
                )
        break

    inference = root / track / "inference"
    inferred = 0
    if (inference / "MANIFEST.json").is_file():
        manifest = json.loads((inference / "MANIFEST.json").read_text())
        inferred = sum(
            (inference / "arms" / variant["name"] / "FINAL.json").is_file()
            for variant in manifest["variants"]
        )
    report = inference / "report/FINAL.json"
    report_status = (
        json.loads(report.read_text()).get("status")
        if report.is_file()
        else "waiting"
    )
    print(
        f"{track}: cycles={completed}/{cycles} {details}; "
        f"inference={inferred}/{cycles + 1}; report={report_status}"
    )
PY

  echo "Ctrl-C stops only this display. Refresh in $INTERVAL seconds."
  sleep "$INTERVAL"
done
