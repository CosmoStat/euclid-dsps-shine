#!/bin/bash
set -eu

ROOT=${1:?ratio-followup experiment root}
while true; do
  date
  if [[ -f "$ROOT/JOBS.env" ]]; then
    source "$ROOT/JOBS.env"
    squeue -j "${ALL_JOBS:?}" -o '%.20i %.12T %.10M %R' || true
    sacct -X -j "$ALL_JOBS" --state=FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED \
      --format=JobID,State,ExitCode || true
  fi
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

print("\nARM          SEED  STATE      EPOCH   BEST NLL  RAW SW  CAL SW  CAL RATIO")
for arm in ("noiseless_photometry", "noisy_photometry"):
    short = "noiseless" if arm.startswith("noiseless") else "noisy"
    for seed in range(2):
        directory = root / f"{arm}_seed{seed}"
        final = read(directory / "FINAL.json")
        progress = read(directory / "PROGRESS.json")
        if final.get("status"):
            training = final["training"]
            raw = final["raw"]
            calibrated = final["calibrated"]
            print(
                f"{short:13} {seed:>4}  {'DONE':9} "
                f"{training['epochs']:>5}  {training['best_nll']:>9.4f} "
                f"{raw['parent_physical_sliced_wasserstein']:>7.4f} "
                f"{calibrated['parent_physical_sliced_wasserstein']:>7.4f} "
                f"{calibrated['ratio_moment_median_abs_error']:>9.4f}"
            )
        elif progress.get("epoch"):
            print(
                f"{short:13} {seed:>4}  {'RUNNING':9} "
                f"{progress['epoch']:>5}  {progress['best_nll']:>9.4f} "
                f"{'-':>7} {'-':>7} {'-':>9}"
            )
        else:
            print(f"{short:13} {seed:>4}  {'WAITING':9} {'-':>5} {'-':>9} {'-':>7} {'-':>7} {'-':>9}")

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
