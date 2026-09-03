#!/bin/bash
set -u

DEFAULT_ENV="outputs/logs/feniks_sc_drws_population_projection_latest.env"
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
  echo "===== DIRECT POPULATION PROJECTION ====="
  echo "root=$PROJECTION_ROOT"
  squeue -r -j "$ALL_JOBS" -o "%.18i %.27j %.2t %.10M %R" 2>/dev/null || true
  sacct -X -j "$ALL_JOBS" \
    --format=JobID,JobName%27,State,Elapsed,Timelimit,ExitCode,NodeList \
    2>/dev/null || true

  echo
  echo "===== 1. BETA ON EXISTING JOINT Q DRAWS ====="
  for item in "beta_fit:16" "beta_validation:4"; do
    name="${item%%:*}"
    expected="${item##*:}"
    count=$(find "$PROJECTION_ROOT/banks/$name/shards" -mindepth 2 -maxdepth 2 \
      -name COMPLETE.json -type f 2>/dev/null | wc -l)
    printf '%-22s %2d/%-2d shards\n' "$name" "$count" "$expected"
  done
  if [[ -s "$PROJECTION_ROOT/BETA_TARGET_COMPLETE.json" ]]; then
    python - "$PROJECTION_ROOT/BETA_TARGET_COMPLETE.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print(f"status={x['status']}")
for split in ('fit', 'validation'):
    d = x[split]
    print(f"{split}: alpha_harmonic={d['alpha_harmonic']:.5f} "
          f"ESS/K={d['ess_fraction']:.4f} maxw={d['maximum_normalized_weight']:.6f}")
PY
  else
    echo "inverse-selection target: attente"
  fi

  echo
  echo "===== 2. TRUTH-FREE FLOW FITS ====="
  if [[ -s "$PROJECTION_ROOT/FIT_PROGRESS.json" ]]; then
    python - "$PROJECTION_ROOT/FIT_PROGRESS.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
if x.get('status') == 'complete':
    for name in ('selected', 'parent'):
        d = x[name]
        print(f"{name}: passes={d['passes_completed']} "
              f"validation NLL={d['best_validation_weighted_nll']:.6f}")
else:
    print(f"target={x.get('target')} pass={x.get('pass')}/{x.get('passes_requested')} "
          f"best validation NLL={x.get('best_validation_weighted_nll')}")
PY
  else
    echo "attente"
  fi

  echo
  echo "===== 3. POSTERIOR CALIBRATION ====="
  FINAL="$PROJECTION_ROOT/POPULATION_PROJECTION_COMPLETE.json"
  if [[ -s "$FINAL" ]]; then
    python - "$FINAL" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
d = x['posterior_calibration']['redshift']
print(f"redshift q PIT KS={d['pit_ks_uniform']:.4f} "
      f"coverage ECE={d['coverage_ece']:.4f} "
      f"C68={d['coverage_68']:.3f} C95={d['coverage_95']:.3f}")
print(f"calibration pass={x['posterior_calibration']['pass']}")
PY
  else
    echo "PIT/coverage: attente"
  fi

  echo
  echo "===== 4. POPULATION DISTRIBUTION PROJECTIONS ====="
  if [[ -s "$FINAL" ]]; then
    python - "$FINAL" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
for name, d in x['distribution_projection']['comparisons'].items():
    print(f"{name}: z CDFsup={d['redshift_cdf_supremum']:.4f} "
          f"z rank-KS={d['redshift_distribution_rank_uniform_ks']:.4f} "
          f"max physical5 CDFsup={d['maximum_physical_5d_cdf_supremum']:.4f}")
print(f"distribution projection pass={x['distribution_projection']['pass']}")
print("These distribution ranks are not posterior PIT; no redshift median gate is used.")
print("plots:", x['artifacts']['redshift_distribution_plot'])
PY
  else
    echo "attente"
  fi

  echo
  echo "===== RECENT ERRORS ====="
  grep -hE "Traceback|RESOURCE_EXHAUSTED|Out of memory|ValueError|RuntimeError|SystemError" \
    "$PROJECTION_LOG_ROOT"/*.err 2>/dev/null | tail -n 12 || true
  echo
  echo "Ctrl-C stops only this monitor. Refresh in ${INTERVAL}s."
  sleep "$INTERVAL"
done
