#!/bin/bash
set -eu

ROOT=${1:?ratio-ladder experiment root}
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

print("\nSTAGE                 STATE       DETAIL")
observation = read(root / "observation/FINAL.json")
if observation.get("status"):
    detail = (
        f"decoder={'PASS' if observation['decoder_pass'] else 'FAIL'} "
        f"noise={'PASS' if observation['conditional_noise_pass'] else 'FAIL'}"
    )
    print(f"{'observation':21} {'DONE':11} {detail}")
else:
    progress = read(root / "observation/PROGRESS.json").get("stage", "waiting")
    print(f"{'observation':21} {'RUNNING' if progress != 'waiting' else 'WAITING':11} {progress}")

manifest = read(root / "MANIFEST.json")
shards = manifest.get("settings", {}).get("reference_shards", 4)
bank_dirs = [root / "bank" / f"reference_{i:03d}" for i in range(shards)]
bank_dirs.append(root / "bank/target")
done = sum(read(path / "FINAL.json").get("status") == "RATIO_LADDER_BANK_COMPLETE" for path in bank_dirs)
active = []
for path in bank_dirs:
    progress = read(path / "PROGRESS.json")
    if "done" in progress and not read(path / "FINAL.json").get("status"):
        active.append(f"{path.name}={100 * progress['done'] / progress['total']:.0f}%")
print(f"{'simulation banks':21} {done:>2}/{len(bank_dirs):<8} {' '.join(active[:2])}")

for arm in ("exact_theta", "physical_a", "noiseless_photometry", "noisy_photometry"):
    final = read(root / arm / "FINAL.json")
    if final.get("status") == "RATIO_LADDER_ARM_COMPLETE":
        detail = (
            f"parentSW={final['parent_physical_sliced_wasserstein']:.4f} "
            f"selectedSW={final['selected_physical_sliced_wasserstein']:.4f}"
        )
        state = "DONE"
    else:
        progress = read(root / arm / "PROGRESS.json")
        if "epoch" in progress:
            state = "RUNNING"
            detail = f"epoch={progress['epoch']} bestNLL={progress['best_nll']:.4f}"
        else:
            state, detail = "WAITING", "-"
    print(f"{arm:21} {state:11} {detail}")

report = read(root / "report/FINAL.json")
print(f"{'report':21} {'DONE' if report.get('status') else 'WAITING':11} -")
PY
  [[ "${2:-}" == --once ]] && break
  echo "Ctrl-C stops only this watcher. Refresh in 30 seconds."
  sleep 30
done
