#!/bin/bash
set -eu

ROOT=${1:?population low-rank audit root}
while true; do
  date
  if [[ -f "$ROOT/JOBS.env" ]]; then
    source "$ROOT/JOBS.env"
    squeue -j "${ALL_JOBS:?}" -o '%.20i %.12T %.10M %R' || true
    sacct -X -j "$ALL_JOBS" --state=FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED \
      --format=JobID,State,ExitCode || true
    FAILED_TASKS=$(sacct -X -n -j "$ARM_JOB" --format=JobIDRaw,State | \
      awk '$2 ~ /FAILED|TIMEOUT|OUT_OF_MEMORY|CANCELLED/ && $1 !~ /\./ {split($1, a, "_"); if (length(a) == 2) print a[2]}' | \
      paste -sd, -)
    BASIS_FAILED=$(sacct -X -n -j "$BASIS_JOB" --format=JobIDRaw,State | \
      awk '$2 ~ /FAILED|TIMEOUT|OUT_OF_MEMORY|CANCELLED/ && $1 !~ /\./ {print 1; exit}')
    export FAILED_TASKS BASIS_FAILED
  fi
  python - "$ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])

def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

basis = read(root / "spectral_basis/FINAL.json")
basis_state = "DONE" if basis.get("status") else (
    "FAILED" if os.environ.get("BASIS_FAILED") else "WAITING"
)
detail = f"rank={basis.get('maximum_rank', '-')} graph={basis.get('graph_components', '-')}" if basis else ""
print(f"\nSPECTRAL BASIS: {basis_state} {detail}")
print("ARM          STATE       STAGE                  R  LAMBDA  PARENT SW  BOOT SW")
failed = set(filter(None, os.environ.get("FAILED_TASKS", "").split(",")))
for task, arm in enumerate(("noiseless_photometry", "noisy_photometry")):
    short = "noiseless" if arm.startswith("noiseless") else "noisy"
    final = read(root / arm / "FINAL.json")
    if final.get("status"):
        row = final["selected"]
        print(
            f"{short:13} {'DONE':11} {'selected':21} "
            f"{final['selected_rank']:>2} {final['selected_strength']:>7.3g} "
            f"{row['parent_physical_sliced_wasserstein']:>10.4f} "
            f"{row['bootstrap_parent_sw_median']:>8.4f}"
        )
        continue
    progress = read(root / arm / "PROGRESS.json")
    stage = progress.get("stage", "waiting")
    if stage in ("bootstrap", "candidate_path"):
        stage = f"{stage} {progress.get('complete', 0)}/{progress.get('total', '?')}"
    state = "FAILED" if str(task) in failed else ("RUNNING" if progress else "WAITING")
    print(f"{short:13} {state:11} {stage:21}")

report = read(root / "report/FINAL.json")
print(f"\nREPORT: {'DONE' if report.get('status') else 'WAITING'}")
if report:
    print(f"NEXT:   {report['next_action']}")
    failed = [key for key, value in report["decisions"].items() if not value]
    print("FAIL:   " + (", ".join(failed) if failed else "none"))
PY
  [[ "${2:-}" == --once ]] && break
  echo "Ctrl-C stops only this watcher. Refresh in 30 seconds."
  sleep 30
done
