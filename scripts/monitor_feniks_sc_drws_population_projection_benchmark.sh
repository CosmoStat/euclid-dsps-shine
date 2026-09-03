#!/bin/bash
set -u

DEFAULT_ENV="outputs/logs/feniks_sc_drws_population_projection_benchmark_latest.env"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  ENV_FILE="$DEFAULT_ENV"
  INTERVAL="$1"
else
  ENV_FILE="${1:-$DEFAULT_ENV}"
  INTERVAL="${2:-${INTERVAL:-60}}"
fi
test -s "$ENV_FILE" || { echo "missing environment: $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"

while true; do
  clear
  date
  echo
  echo "===== POPULATION FLOW ARCHITECTURE BENCHMARK ====="
  echo "root=$BENCHMARK_ROOT"
  squeue -r -j "$ALL_JOBS" -o "%.18i %.27j %.2t %.10M %R" 2>/dev/null || true
  sacct -X -j "$ALL_JOBS" \
    --format=JobID,JobName%27,State,Elapsed,Timelimit,ExitCode,NodeList \
    2>/dev/null || true

  echo
  echo "===== 1. PARALLEL TRUTH-FREE FITS ====="
  python - "$BENCHMARK_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.load(open(root / "RUN_MANIFEST.json"))
for candidate in manifest["architecture_benchmark"]["trained_candidates"]:
    base = root / "candidates" / candidate["name"]
    receipt = base / "FIT_COMPLETE.json"
    progress = base / "FIT_PROGRESS.json"
    if receipt.exists():
        data = json.load(open(receipt))
        print(f"{candidate['name']:<24} complete "
              f"NLL(selected/parent)={data['selected']['best_validation_weighted_nll']:.5f}/"
              f"{data['parent']['best_validation_weighted_nll']:.5f}")
    elif progress.exists():
        data = json.load(open(progress))
        print(f"{candidate['name']:<24} running target={data.get('target')} "
              f"pass={data.get('pass')}/{data.get('passes_requested')}")
    else:
        print(f"{candidate['name']:<24} pending")
PY

  echo
  echo "===== 2. TRUTH-FREE VALIDATION AND WINNER ====="
  python - "$BENCHMARK_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.load(open(root / "RUN_MANIFEST.json"))
names = [x["name"] for x in manifest["architecture_benchmark"]["trained_candidates"]]
names.append(manifest["architecture_benchmark"]["baseline"]["name"])
for name in names:
    path = root / "candidates" / name / "TRUTH_FREE_EVALUATION.json"
    if not path.exists():
        print(f"{name:<24} pending")
        continue
    data = json.load(open(path))
    c = data["comparisons"]
    selected = c["selected_flow_vs_q_aggregate"]
    parent = c["parent_flow_vs_inverse_beta_q"]
    selected_parent = c["selected_parent_flow_vs_q_aggregate"]
    print(f"{name:<24} score={data['primary_score']:.3f} "
          f"z sel/parent/sel-parent={selected['redshift_cdf_supremum']:.3f}/"
          f"{parent['redshift_cdf_supremum']:.3f}/"
          f"{selected_parent['redshift_cdf_supremum']:.3f} "
          f"coremax={selected['maximum_core_5d_cdf_supremum']:.3f}/"
          f"{parent['maximum_core_5d_cdf_supremum']:.3f}/"
          f"{selected_parent['maximum_core_5d_cdf_supremum']:.3f}")
winner = root / "TRUTH_FREE_ARCHITECTURE_WINNER.json"
if winner.exists():
    data = json.load(open(winner))
    print(f"WINNER={data['winner']} score={data['winner_primary_score']:.3f} "
          f"all_gates={data['winner_passes_all_truth_free_distribution_gates']}")
else:
    print("winner: pending")
PY

  echo
  echo "===== 3. FROZEN WINNER CLOSURE ====="
  FINAL="$BENCHMARK_ROOT/POPULATION_PROJECTION_COMPLETE.json"
  if [[ -s "$FINAL" ]]; then
    python - "$FINAL" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
d = x["posterior_calibration"]["redshift"]
print(f"q PIT KS={d['pit_ks_uniform']:.4f} ECE={d['coverage_ece']:.4f} "
      f"C68={d['coverage_68']:.3f} C95={d['coverage_95']:.3f}")
print(f"projection_pass={x['distribution_projection']['pass']} "
      f"posterior_calibration_pass={x['posterior_calibration']['pass']}")
print("plots:", x["artifacts"]["redshift_distribution_plot"])
PY
  else
    echo "closure: pending (truth remains unread until winner freeze)"
  fi

  echo
  echo "===== RECENT ERRORS ====="
  find "$BENCHMARK_LOG_ROOT" -maxdepth 1 -type f -name "*.err" \
    -exec grep -hE \
      "Traceback|RESOURCE_EXHAUSTED|Out of memory|ValueError|RuntimeError|SystemError" \
      {} + 2>/dev/null | tail -n 16 || true
  echo
  echo "Ctrl-C stops only this monitor. Refresh in ${INTERVAL}s."
  sleep "$INTERVAL"
done
