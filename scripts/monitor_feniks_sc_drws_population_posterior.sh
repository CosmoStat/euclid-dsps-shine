#!/bin/bash
set -u

DEFAULT_ENV="outputs/logs/feniks_sc_drws_population_posterior_latest.env"
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
  echo "===== PROJECTED-PARENT INDIVIDUAL POSTERIORS ====="
  echo "root=$POSTERIOR_ROOT"
  squeue -r -j "$ALL_JOBS" -o "%.18i %.27j %.2t %.10M %R" 2>/dev/null || true
  sacct -X -j "$ALL_JOBS" \
    --format=JobID,JobName%27,State,Elapsed,Timelimit,ExitCode,NodeList \
    2>/dev/null || true

  echo
  echo "===== 1. K=1024 SHARDS ====="
  expected=$(python - "$POSTERIOR_ROOT/RUN_MANIFEST.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))["cohort"]["shards"]))
PY
)
  complete=$(find "$POSTERIOR_ROOT/shards" -mindepth 2 -maxdepth 2 \
    -name DONE -type f 2>/dev/null | wc -l)
  echo "complete=$complete/$expected"
  for gate in "$POSTERIOR_ROOT"/shards/shard_*/projected_parent_iw/support_gate.json; do
    [[ -s "$gate" ]] || continue
    python - "$gate" <<'PY'
import json, re, sys
from pathlib import Path
x = json.load(open(sys.argv[1]))
match = re.search(r"shard_(\d+)", str(Path(sys.argv[1])))
shard = int(match.group(1)) if match else -1
print(f"shard={shard:02d} parent={x['status']} "
      f"ESS/K={x['median_raw_ess_fraction']:.5f} "
      f"k>0.7={x['fraction_pareto_k_gt_0p7']:.3f}")
PY
  done

  echo
  echo "===== 2. GLOBAL SAME-DRAW SUPPORT ====="
  FINAL="$POSTERIOR_ROOT/INDIVIDUAL_POSTERIOR_DIAGNOSTIC_COMPLETE.json"
  if [[ -s "$FINAL" ]]; then
    python - "$FINAL" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
for key, label in (("source_prior_support", "source"),
                   ("projected_parent_support", "projected parent")):
    s = x[key]
    print(f"{label}: {s['status']} ESS={s['median_raw_ess']:.2f} "
          f"ESS/K={s['median_raw_ess_fraction']:.5f} "
          f"k>0.7={s['fraction_pareto_k_gt_0p7']:.3f} "
          f"maxw_p90={s['p90_max_raw_weight']:.3f}")
d = x["same_draw_support_delta_parent_minus_source"]
print(f"parent-source delta ESS={d['median_raw_ess']:+.2f} "
      f"delta k>0.7={d['fraction_pareto_k_gt_0p7']:+.3f}")
PY
  else
    echo "attente de l'agrégation des huit shards"
  fi

  echo
  echo "===== 3. REDSHIFT PIT / COVERAGE ====="
  if [[ -s "$FINAL" ]]; then
    python - "$FINAL" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
for name, row in x["redshift_calibration"].items():
    print(f"{name:<22} PIT_KS={row['pit_ks_uniform']:.4f} "
          f"ECE={row['coverage_ece']:.4f} "
          f"C68={row['coverage_68']:.3f} C95={row['coverage_95']:.3f}")
PY
  else
    echo "truth non lue avant la finalisation"
  fi

  echo
  echo "===== 4. CORNERS / PPC ====="
  if [[ -s "$FINAL" ]]; then
    python - "$FINAL" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print(f"panels={len(x['panels'])}")
print("corners:", x["artifacts"]["panel_manifest"])
print("PPC:", x["artifacts"]["ppc_plot"])
print("support:", x["artifacts"]["support_plot"])
PY
  else
    echo "attente"
  fi

  echo
  echo "===== RECENT ERRORS ====="
  find "$POSTERIOR_LOG_ROOT" -maxdepth 1 -type f -name "*.err" \
    -exec grep -hE \
      "Traceback|RESOURCE_EXHAUSTED|Out of memory|ValueError|RuntimeError|SystemError|TypeError" \
      {} + 2>/dev/null | tail -n 16 || true
  echo
  echo "Ctrl-C stops only this monitor. Refresh in ${INTERVAL}s."
  sleep "$INTERVAL"
done
