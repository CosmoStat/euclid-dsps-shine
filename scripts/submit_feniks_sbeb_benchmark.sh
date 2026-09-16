#!/bin/bash
# Usage: submit_feniks_sbeb_benchmark.sh SOURCE Q_TRAIN PRIOR SELECTION NUTS ROOT [concurrency]
set -Eeuo pipefail
trap 'echo "SBEB submission stopped at line $LINENO: $BASH_COMMAND" >&2' ERR

SOURCE="$(realpath "${1:?completed AVI source root}")"
Q_TRAIN="$(realpath "${2:?completed Q_latest_refresh root}")"
PRIOR="$(realpath "${3:?completed P_latest_prior root}")"
SELECTION="$(realpath "${4:?completed exact-selection validation root}")"
NUTS="$(realpath "${5:?historical eight-object NUTS root}")"
export FENIKS_SBEB_ROOT="$(realpath -m "${6:?new SBEB benchmark root}")"
CONCURRENCY="${7:-8}"
[[ "$CONCURRENCY" =~ ^[1-8]$ ]] || {
  echo "concurrency must be 1..8 (four H100s per task)" >&2
  exit 2
}
test ! -e "$FENIKS_SBEB_ROOT"
test -s "$SELECTION/selection/FINAL.json"
test -s "$NUTS/MANIFEST.json"

REPO="$(pwd -P)"
test -d "$REPO/filters"
ARCHIVE="$FENIKS_SBEB_ROOT.code.tar"
test ! -e "$ARCHIVE"
mkdir -p "$(dirname "$FENIKS_SBEB_ROOT")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export FENIKS_SBEB_CODE="$SCRATCH/feniks_sc_drws_runtime/code/sbeb-$DIGEST"
mkdir -p "$FENIKS_SBEB_CODE"
tar -xf "$ARCHIVE" -C "$FENIKS_SBEB_CODE"
if [[ ! -e "$FENIKS_SBEB_CODE/Data" && ! -L "$FENIKS_SBEB_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$FENIKS_SBEB_CODE/Data"
fi
cd "$FENIKS_SBEB_CODE"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m scripts.feniks_sbeb_benchmark prepare \
  --source "$SOURCE" --warm-training "$Q_TRAIN" --warm-prior "$PRIOR" \
  --selection "$SELECTION" --nuts-root "$NUTS" --root "$FENIKS_SBEB_ROOT" \
  --scratch-epochs 180 --cycles 4 --q-epochs 24 --prior-sweeps 5
printf '%s\n' "$FENIKS_SBEB_CODE" > "$FENIKS_SBEB_ROOT/CODE_DIR"

submit_array() {
  local mode="$1" range="$2" dependency="$3" walltime="$4" name="$5"
  export FENIKS_SBEB_MODE="$mode"
  local raw
  raw=$(sbatch --parsable --time="$walltime" --array="$range%$CONCURRENCY" \
    --dependency="afterok:$dependency" --export=ALL \
    --output="$FENIKS_SBEB_ROOT/logs/${name}-%A_%a.out" \
    --error="$FENIKS_SBEB_ROOT/logs/${name}-%A_%a.err" \
    "$FENIKS_SBEB_CODE/scripts/feniks_sbeb_benchmark.slurm")
  printf '%s' "${raw%%;*}"
}

submit_single() {
  local mode="$1" dependency="$2" walltime="$3" name="$4"
  export FENIKS_SBEB_MODE="$mode"
  local raw
  raw=$(sbatch --parsable --time="$walltime" --dependency="afterok:$dependency" \
    --export=ALL \
    --output="$FENIKS_SBEB_ROOT/logs/${name}-%j.out" \
    --error="$FENIKS_SBEB_ROOT/logs/${name}-%j.err" \
    "$FENIKS_SBEB_CODE/scripts/feniks_sbeb_benchmark.slurm")
  printf '%s' "${raw%%;*}"
}

# A zero-duration scheduler dependency is not available, so submit bootstrap
# without a dependency after the synchronous preparation above.
export FENIKS_SBEB_MODE=bootstrap FENIKS_SBEB_MAX_HOURS=19
RAW=$(sbatch --parsable --time=20:00:00 --array="0-2%$CONCURRENCY" --export=ALL \
  --output="$FENIKS_SBEB_ROOT/logs/bootstrap-%A_%a.out" \
  --error="$FENIKS_SBEB_ROOT/logs/bootstrap-%A_%a.err" \
  "$FENIKS_SBEB_CODE/scripts/feniks_sbeb_benchmark.slurm")
BOOTSTRAP_JOB="${RAW%%;*}"

FACTOR_JOB=$(submit_array factor 0-23 "$BOOTSTRAP_JOB" 12:00:00 factor)
FACTOR_PREP_JOB=$(submit_single prepare-factor-inference "$FACTOR_JOB" 00:30:00 factor-prep)
FACTOR_INFER_JOB=$(submit_array factor-infer 0-23 "$FACTOR_PREP_JOB" 06:00:00 factor-infer)
FACTOR_REPORT_JOB=$(submit_single factor-report "$FACTOR_INFER_JOB" 06:00:00 factor-report)
TRAJECTORY_PREP_JOB=$(submit_single prepare-trajectories "$FACTOR_REPORT_JOB" 00:30:00 trajectories-prep)

DEPENDENCY="$TRAJECTORY_PREP_JOB"
CYCLE_JOBS=()
for CYCLE in 1 2 3 4; do
  export FENIKS_SBEB_MODE=em-cycle FENIKS_SBEB_CYCLE="$CYCLE" FENIKS_SBEB_MAX_HOURS=9
  RAW=$(sbatch --parsable --time=10:00:00 --array="0-7%$CONCURRENCY" \
    --dependency="afterok:$DEPENDENCY" --export=ALL \
    --output="$FENIKS_SBEB_ROOT/logs/cycle-${CYCLE}-%A_%a.out" \
    --error="$FENIKS_SBEB_ROOT/logs/cycle-${CYCLE}-%A_%a.err" \
    "$FENIKS_SBEB_CODE/scripts/feniks_sbeb_benchmark.slurm")
  JOB="${RAW%%;*}"
  CYCLE_JOBS+=("$JOB")
  DEPENDENCY="$JOB"
done

ENDPOINT_PREP_JOB=$(submit_single prepare-endpoint-factorials "$DEPENDENCY" 00:30:00 endpoint-factorials-prep)
ENDPOINT_INFER_JOB=$(submit_array endpoint-factorial-infer 0-31 "$ENDPOINT_PREP_JOB" 06:00:00 endpoint-factorial-infer)
ENDPOINT_REPORT_JOB=$(submit_array endpoint-factorial-report 0-7 "$ENDPOINT_INFER_JOB" 06:00:00 endpoint-factorial-report)
EM_PREP_JOB=$(submit_single prepare-em-inference "$ENDPOINT_PREP_JOB" 00:30:00 em-inference-prep)
EM_INFER_JOB=$(submit_array em-infer 0-39 "$EM_PREP_JOB" 06:00:00 em-infer)
EM_REPORT_JOB=$(submit_array em-report 0-7 "$EM_INFER_JOB" 06:00:00 em-report)
NUTS_JOB=$(submit_array nuts-infer 0-7 "$EM_REPORT_JOB:$ENDPOINT_REPORT_JOB" 06:00:00 nuts-infer)
FINAL_REPORT_JOB=$(submit_single report "$NUTS_JOB" 08:00:00 final-report)

CYCLE_JOB_CSV=$(IFS=,; echo "${CYCLE_JOBS[*]}")
ALL_JOBS="$BOOTSTRAP_JOB,$FACTOR_JOB,$FACTOR_PREP_JOB,$FACTOR_INFER_JOB,$FACTOR_REPORT_JOB,$TRAJECTORY_PREP_JOB,$CYCLE_JOB_CSV,$ENDPOINT_PREP_JOB,$ENDPOINT_INFER_JOB,$ENDPOINT_REPORT_JOB,$EM_PREP_JOB,$EM_INFER_JOB,$EM_REPORT_JOB,$NUTS_JOB,$FINAL_REPORT_JOB"
printf 'export FENIKS_SBEB_ROOT=%q\nexport FENIKS_SBEB_CODE=%q\nexport BOOTSTRAP_JOB=%q\nexport FACTOR_JOB=%q\nexport FACTOR_PREP_JOB=%q\nexport FACTOR_INFER_JOB=%q\nexport FACTOR_REPORT_JOB=%q\nexport TRAJECTORY_PREP_JOB=%q\nexport CYCLE_JOBS=%q\nexport ENDPOINT_PREP_JOB=%q\nexport ENDPOINT_INFER_JOB=%q\nexport ENDPOINT_REPORT_JOB=%q\nexport EM_PREP_JOB=%q\nexport EM_INFER_JOB=%q\nexport EM_REPORT_JOB=%q\nexport NUTS_JOB=%q\nexport FINAL_REPORT_JOB=%q\nexport ALL_JOBS=%q\n' \
  "$FENIKS_SBEB_ROOT" "$FENIKS_SBEB_CODE" "$BOOTSTRAP_JOB" "$FACTOR_JOB" \
  "$FACTOR_PREP_JOB" "$FACTOR_INFER_JOB" "$FACTOR_REPORT_JOB" \
  "$TRAJECTORY_PREP_JOB" "$CYCLE_JOB_CSV" "$ENDPOINT_PREP_JOB" \
  "$ENDPOINT_INFER_JOB" "$ENDPOINT_REPORT_JOB" "$EM_PREP_JOB" "$EM_INFER_JOB" \
  "$EM_REPORT_JOB" "$NUTS_JOB" "$FINAL_REPORT_JOB" "$ALL_JOBS" \
  > "$FENIKS_SBEB_ROOT/JOBS.env"

printf 'root=%s\nbootstrap=%s\nfactor=%s\nfactor_inference=%s\nfactor_report=%s\ncycles=%s\nendpoint_factorial_inference=%s\nendpoint_factorial_reports=%s\nem_inference=%s\nem_reports=%s\nnuts=%s\nfinal_report=%s\npeak_h100s=%s\nwatch=bash scripts/watch_feniks_sbeb_benchmark.sh %q\n' \
  "$FENIKS_SBEB_ROOT" "$BOOTSTRAP_JOB" "$FACTOR_JOB" "$FACTOR_INFER_JOB" \
  "$FACTOR_REPORT_JOB" "$CYCLE_JOB_CSV" "$ENDPOINT_INFER_JOB" \
  "$ENDPOINT_REPORT_JOB" "$EM_INFER_JOB" "$EM_REPORT_JOB" \
  "$NUTS_JOB" "$FINAL_REPORT_JOB" "$((4 * CONCURRENCY))" "$FENIKS_SBEB_ROOT"
