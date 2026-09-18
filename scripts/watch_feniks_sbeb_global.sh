#!/bin/bash
# Usage: watch_feniks_sbeb_global.sh PARENT_ROOT CONTINUATION_ROOT [refresh_seconds]
set -Eeuo pipefail

PARENT="$(realpath "${1:?completed/running SBEB benchmark root}")"
CONTINUATION="$(realpath "${2:?SBEB continuation root}")"
INTERVAL="${3:-30}"

source "$PARENT/JOBS.env"
PARENT_JOBS="$ALL_JOBS"
source "$CONTINUATION/JOBS.env"
CONTINUATION_JOBS="$ALL_JOBS"
WATCH_JOBS="$PARENT_JOBS,$CONTINUATION_JOBS"

while true; do
  clear || true
  date
  echo "===== SLURM: ALL CAMPAIGN JOBS ====="
  squeue -r -j "$WATCH_JOBS" -o "%.24i %.18T %.12M %R" || true

  echo
  echo "===== FAILURES ====="
  sacct -X -n -j "$WATCH_JOBS" \
    --state=FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL,CANCELLED \
    --format=JobID%24,State%18,Elapsed,ExitCode \
    | sed '/^[[:space:]]*$/d' || true

  echo
  python - "$PARENT" "$CONTINUATION" <<'PY'
import json
import sys
from pathlib import Path

parent = Path(sys.argv[1])
continuation = Path(sys.argv[2])


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def unit_state(path: Path, *, final_name="FINAL.json"):
    final = load(path / final_name)
    if final is not None:
        return "done", final.get("status", "complete")
    progress = load(path / "PROGRESS.json")
    if progress is None:
        return "waiting", ""
    percent = progress.get("progress_percent", progress.get("percent"))
    stage = progress.get("stage", "running")
    detail = stage if percent is None else f"{stage} {float(percent):.1f}%"
    return "running", detail


def section(title, entries):
    states = [(label, *unit_state(path, final_name=final)) for label, path, final in entries]
    done = sum(status == "done" for _, status, _ in states)
    running = [(label, detail) for label, status, detail in states if status == "running"]
    waiting = sum(status == "waiting" for _, status, _ in states)
    print(f"\n===== {title} =====")
    print(f"done={done}/{len(states)} running={len(running)} waiting={waiting}")
    for label, detail in running:
        print(f"  {label:<42} {detail}")


manifest = load(parent / "MANIFEST.json")
tracks = [item["name"] for item in manifest["trajectory_tracks"]]
cycles = int(manifest["cycles"])

bootstrap = [
    (
        f"scratch_r{cut}",
        parent / f"bootstrap/r{cut}/arms/scratch_r{cut}",
        "FINAL.json",
    )
    for cut in (25, 27, 29)
]
section("SCRATCH BOOTSTRAP", bootstrap)

factor_training = [
    (
        cell["name"],
        parent / "factor/cells" / cell["name"] / "arms" / f"P_{cell['name']}",
        "FINAL.json",
    )
    for cell in manifest["factor_cells"]
]
section("INITIAL FACTOR GRID: PRIOR FITS", factor_training)

factor_inference = [
    (
        cell["name"],
        parent / f"factor/inference/r{cell['cut']}/arms" / cell["name"],
        "FINAL.json",
    )
    for cell in manifest["factor_cells"]
]
section("INITIAL FACTOR GRID: INFERENCE", factor_inference)
factor_report = load(parent / "factor/report/FINAL.json")
print("factor report:", factor_report.get("status") if factor_report else "waiting")

print("\n===== EM TRAINING CYCLES 1-4 =====")
for track in tracks:
    em = parent / "trajectories" / track / "em"
    completed = sum(
        (em / "cycles" / f"cycle_{cycle:02d}/FINAL.json").is_file()
        for cycle in range(1, cycles + 1)
    )
    print(f"  {track:<24} {completed}/{cycles}")

endpoint = []
for track in tracks:
    root = parent / "trajectories" / track / "endpoint_factorial"
    endpoint_manifest = load(root / "MANIFEST.json")
    if endpoint_manifest:
        endpoint.extend(
            (
                f"{track}/{variant['name']}",
                root / "arms" / variant["name"],
                "FINAL.json",
            )
            for variant in endpoint_manifest["variants"]
        )
section("ENDPOINT Q0/Q4 x P0/P4 FACTORIALS", endpoint)

endpoint_reports = [
    (
        track,
        parent / "trajectories" / track / "endpoint_factorial/report",
        "FACTORIAL_FINAL.json",
    )
    for track in tracks
]
section("ENDPOINT FACTORIAL REPORTS", endpoint_reports)

em_inference = []
for track in tracks:
    root = parent / "trajectories" / track / "inference"
    inference_manifest = load(root / "MANIFEST.json")
    if inference_manifest:
        em_inference.extend(
            (
                f"{track}/{variant['name']}",
                root / "arms" / variant["name"],
                "FINAL.json",
            )
            for variant in inference_manifest["variants"]
        )
section("FULL EM INFERENCE BY SAVED CYCLE", em_inference)

em_reports = [
    (track, parent / "trajectories" / track / "inference/report", "FINAL.json")
    for track in tracks
]
section("FULL EM REPORTS", em_reports)

nuts = [
    (track, parent / "nuts" / track / "inference", "FINAL.json")
    for track in tracks
]
section("FINAL NUTS-COHORT INFERENCE", nuts)

final_report = load(parent / "report/FINAL.json")
print("\n===== ORIGINAL CAMPAIGN FINAL REPORT =====")
print(final_report.get("status") if final_report else "waiting")

print("\n===== CONTINUATION: GLOBAL CYCLES 5-7 =====")
for track in ("warm_iw_r27", "warm_iw_r29"):
    em = continuation / track / "em"
    completed = 0
    active = "waiting for original final report"
    for local_cycle in range(1, 4):
        cycle_root = em / "cycles" / f"cycle_{local_cycle:02d}"
        receipt = load(cycle_root / "FINAL.json")
        if receipt:
            completed += 1
            active = (
                f"global {local_cycle + 4} complete; "
                f"alpha={receipt.get('selection_log_alpha')} "
                f"ESS={receipt.get('posterior_ess_median')}"
            )
            continue
        for stage, arm_name in (
            ("mstep", f"P_em_{local_cycle:02d}"),
            ("qstep", f"Q_em_{local_cycle:02d}"),
        ):
            status, detail = unit_state(cycle_root / stage / "arms" / arm_name)
            if status != "waiting":
                active = f"global {local_cycle + 4} {stage}: {status} {detail}"
        break
    print(f"  {track:<24} cycles={completed}/3; {active}")

continuation_inference = []
continuation_reports = []
for track in ("warm_iw_r27", "warm_iw_r29"):
    root = continuation / track / "inference"
    inference_manifest = load(root / "MANIFEST.json")
    if inference_manifest:
        continuation_inference.extend(
            (
                f"{track}/global_{variant.get('global_cycle', '?')}",
                root / "arms" / variant["name"],
                "FINAL.json",
            )
            for variant in inference_manifest["variants"]
        )
    continuation_reports.append((track, root / "report", "FINAL.json"))
section("CONTINUATION INFERENCE GLOBAL CYCLES 4-7", continuation_inference)
section("CONTINUATION REPORTS", continuation_reports)
PY

  echo
  echo "Ctrl-C stops only this watcher. Refresh in $INTERVAL seconds."
  sleep "$INTERVAL"
done
