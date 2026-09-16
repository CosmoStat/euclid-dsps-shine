#!/bin/bash
# Usage: submit_feniks_avi_em.sh SOURCE Q_TRAIN PRIOR SELECTION EM_ROOT INFERENCE_ROOT [cycles] [inference_concurrency]
set -Eeuo pipefail
trap 'echo "AVI EM submission stopped at line $LINENO: $BASH_COMMAND" >&2' ERR

SOURCE="$(realpath "${1:?completed avi_encoder_experiments root}")"
Q_TRAIN="$(realpath "${2:?completed Q_latest_refresh training root}")"
PRIOR="$(realpath "${3:?completed P_latest_prior follow-up root}")"
SELECTION="$(realpath "${4:?completed exact-selection validation root}")"
export AVI_EM_ROOT="$(realpath -m "${5:?new EM root}")"
export AVI_EM_INFERENCE_ROOT="$(realpath -m "${6:?new inference root}")"
CYCLES="${7:-4}"
INFERENCE_CONCURRENCY="${8:-5}"
[[ "$CYCLES" =~ ^[1-9][0-9]*$ ]] || { echo "cycles must be positive" >&2; exit 2; }
[[ "$INFERENCE_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || {
  echo "inference concurrency must be positive" >&2
  exit 2
}
test ! -e "$AVI_EM_ROOT"
test ! -e "$AVI_EM_INFERENCE_ROOT"
test -s "$SELECTION/selection/FINAL.json"
test -s "$SELECTION/selection/true_parent.parquet"
test -s "$SELECTION/selection/true_selected.parquet"

REPO="$(pwd -P)"
test -d "$REPO/filters"
ARCHIVE="$AVI_EM_ROOT.code.tar"
test ! -e "$ARCHIVE"
mkdir -p "$(dirname "$AVI_EM_ROOT")" "$(dirname "$AVI_EM_INFERENCE_ROOT")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export AVI_EM_CODE="$SCRATCH/feniks_sc_drws_runtime/code/avi-em-$DIGEST"
mkdir -p "$AVI_EM_CODE"
tar -xf "$ARCHIVE" -C "$AVI_EM_CODE"
if [[ ! -e "$AVI_EM_CODE/Data" && ! -L "$AVI_EM_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$AVI_EM_CODE/Data"
fi
cd "$AVI_EM_CODE"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m scripts.feniks_avi_em prepare \
  --source "$SOURCE" --training "$Q_TRAIN" --prior-followup "$PRIOR" \
  --selection "$SELECTION" --root "$AVI_EM_ROOT" \
  --inference-root "$AVI_EM_INFERENCE_ROOT" --cycles "$CYCLES" --q-epochs 3
printf '%s\n' "$AVI_EM_CODE" > "$AVI_EM_ROOT/CODE_DIR"

DEPENDENCY=""
CYCLE_JOBS=()
for CYCLE in $(seq 1 "$CYCLES"); do
  export AVI_EM_MODE=cycle AVI_EM_CYCLE="$CYCLE"
  EXTRA=()
  if [[ -n "$DEPENDENCY" ]]; then EXTRA+=(--dependency="afterok:$DEPENDENCY"); fi
  RAW=$(sbatch --parsable --time=06:00:00 "${EXTRA[@]}" --export=ALL \
    --output="$AVI_EM_ROOT/logs/cycle-${CYCLE}-%j.out" \
    --error="$AVI_EM_ROOT/logs/cycle-${CYCLE}-%j.err" \
    "$AVI_EM_CODE/scripts/feniks_avi_em.slurm")
  JOB="${RAW%%;*}"
  CYCLE_JOBS+=("$JOB")
  DEPENDENCY="$JOB"
done

export AVI_EM_MODE=prepare-inference AVI_EM_PARTICLES=4096
RAW=$(sbatch --parsable --time=00:30:00 --gres=gpu:1 --cpus-per-task=12 \
  --dependency="afterok:$DEPENDENCY" --export=ALL \
  --output="$AVI_EM_ROOT/logs/prepare-inference-%j.out" \
  --error="$AVI_EM_ROOT/logs/prepare-inference-%j.err" \
  "$AVI_EM_CODE/scripts/feniks_avi_em.slurm")
PREPARE_JOB="${RAW%%;*}"

export AVI_EM_MODE=infer
RAW=$(sbatch --parsable --time=06:00:00 \
  --array="0-${CYCLES}%${INFERENCE_CONCURRENCY}" \
  --dependency="afterok:$PREPARE_JOB" --export=ALL \
  --output="$AVI_EM_ROOT/logs/infer-%A_%a.out" \
  --error="$AVI_EM_ROOT/logs/infer-%A_%a.err" \
  "$AVI_EM_CODE/scripts/feniks_avi_em.slurm")
INFERENCE_JOB="${RAW%%;*}"

export AVI_EM_MODE=report
RAW=$(sbatch --parsable --time=03:00:00 --gres=gpu:1 --cpus-per-task=24 \
  --dependency="afterok:$INFERENCE_JOB" --export=ALL \
  --output="$AVI_EM_ROOT/logs/report-%j.out" \
  --error="$AVI_EM_ROOT/logs/report-%j.err" \
  "$AVI_EM_CODE/scripts/feniks_avi_em.slurm")
REPORT_JOB="${RAW%%;*}"

CYCLE_JOB_CSV=$(IFS=,; echo "${CYCLE_JOBS[*]}")
ALL_JOBS="$CYCLE_JOB_CSV,$PREPARE_JOB,$INFERENCE_JOB,$REPORT_JOB"
JOBS="$AVI_EM_ROOT/JOBS.env"
printf 'export AVI_EM_ROOT=%q\nexport AVI_EM_INFERENCE_ROOT=%q\nexport AVI_EM_CODE=%q\nexport CYCLES=%q\nexport CYCLE_JOBS=%q\nexport PREPARE_JOB=%q\nexport INFERENCE_JOB=%q\nexport REPORT_JOB=%q\nexport ALL_JOBS=%q\n' \
  "$AVI_EM_ROOT" "$AVI_EM_INFERENCE_ROOT" "$AVI_EM_CODE" "$CYCLES" \
  "$CYCLE_JOB_CSV" "$PREPARE_JOB" "$INFERENCE_JOB" "$REPORT_JOB" \
  "$ALL_JOBS" > "$JOBS"

printf 'cycles=%s\ncycle_jobs=%s\nprepare_inference=%s\ninference=%s\nreport=%s\nem_root=%s\ninference_root=%s\npeak_h100s=%s\nwatch=bash scripts/watch_feniks_avi_em.sh %q\n' \
  "$CYCLES" "$CYCLE_JOB_CSV" "$PREPARE_JOB" "$INFERENCE_JOB" \
  "$REPORT_JOB" "$AVI_EM_ROOT" "$AVI_EM_INFERENCE_ROOT" \
  "$((4 * INFERENCE_CONCURRENCY))" "$AVI_EM_ROOT"
