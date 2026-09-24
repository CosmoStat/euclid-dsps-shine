#!/bin/bash
set -eu

ROOT=${1:?ratio-convergence experiment root}
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

print("\nARM          STATE      EPOCH  BEST NLL  PARENT SW  ZERO WT  RATIO  PLATEAU")
for arm in ("noiseless_photometry", "noisy_photometry"):
    short = "noiseless" if arm.startswith("noiseless") else "noisy"
    final = read(root / arm / "FINAL.json")
    progress = final.get("final") or read(root / arm / "PROGRESS.json")
    if final.get("status"):
        state = "CONVERGED" if final["converged"] else "MAX_EPOCH"
    elif progress.get("epoch"):
        state = "RUNNING"
    else:
        state = "WAITING"
    if progress.get("parent_physical_sliced_wasserstein") is not None:
        print(
            f"{short:13} {state:10} {int(progress['epoch']):>5} "
            f"{progress['best_nll']:>9.4f} "
            f"{progress['parent_physical_sliced_wasserstein']:>9.4f} "
            f"{progress['zero_parent_weight_fraction']:>8.3f} "
            f"{progress['ratio_moment_median_abs_error']:>6.3f} "
            f"{str(bool(progress['nll_plateau'])):>7}"
        )
    elif progress.get("epoch") is not None:
        print(
            f"{short:13} {'TRAINING':10} {int(progress['epoch']):>5} "
            f"{progress['best_nll']:>9.4f} "
            f"{'-':>9} {'-':>8} {'-':>6} {'-':>7}"
        )
    else:
        print(f"{short:13} {state:10} {'-':>5} {'-':>9} {'-':>9} {'-':>8} {'-':>6} {'-':>7}")

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
