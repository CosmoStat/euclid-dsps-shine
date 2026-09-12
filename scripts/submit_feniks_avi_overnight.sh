#!/bin/bash
# Usage: submit_feniks_avi_overnight.sh SOURCE_TRAINING PRIOR_FOLLOWUP TRAIN_ROOT INFERENCE_ROOT [inference_concurrency]
set -Eeuo pipefail
trap 'echo "AVI overnight submission stopped at line $LINENO: $BASH_COMMAND" >&2' ERR

export AVI_SOURCE_ROOT="$(realpath "${1:?completed avi_encoder_experiments root}")"
export AVI_PRIOR_ROOT="$(realpath "${2:?completed avi_expert_prior_followup root}")"
export AVI_TRAIN_ROOT="$(realpath -m "${3:?new q-refresh root}")"
export AVI_INFERENCE_ROOT="$(realpath -m "${4:?new inference root}")"
INFERENCE_CONCURRENCY="${5:-5}"
[[ "$INFERENCE_CONCURRENCY" =~ ^[1-5]$ ]] || {
  echo "inference concurrency must be 1..5" >&2
  exit 2
}
test ! -e "$AVI_TRAIN_ROOT"
test ! -e "$AVI_INFERENCE_ROOT"

REPO="$(pwd -P)"
test -d "$REPO/filters"
ARCHIVE="$AVI_TRAIN_ROOT.code.tar"
test ! -e "$ARCHIVE"
mkdir -p "$(dirname "$AVI_TRAIN_ROOT")" "$(dirname "$AVI_INFERENCE_ROOT")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export AVI_OVERNIGHT_CODE="$SCRATCH/feniks_sc_drws_runtime/code/avi-overnight-$DIGEST"
mkdir -p "$AVI_OVERNIGHT_CODE"
tar -xf "$ARCHIVE" -C "$AVI_OVERNIGHT_CODE"
if [[ ! -e "$AVI_OVERNIGHT_CODE/Data" && ! -L "$AVI_OVERNIGHT_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$AVI_OVERNIGHT_CODE/Data"
fi
cd "$AVI_OVERNIGHT_CODE"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m scripts.feniks_avi_overnight \
  prepare-training --source "$AVI_SOURCE_ROOT" \
  --prior-followup "$AVI_PRIOR_ROOT" --root "$AVI_TRAIN_ROOT"
printf '%s\n' "$AVI_OVERNIGHT_CODE" > "$AVI_TRAIN_ROOT/CODE_DIR"

export AVI_OVERNIGHT_MODE=preflight
PREFLIGHT=$(sbatch --parsable --time=02:00:00 --array=0-1%2 --export=ALL \
  --output="$AVI_TRAIN_ROOT/logs/preflight-%A_%a.out" \
  --error="$AVI_TRAIN_ROOT/logs/preflight-%A_%a.err" \
  "$AVI_OVERNIGHT_CODE/scripts/feniks_avi_overnight.slurm")
PREFLIGHT=${PREFLIGHT%%;*}

export AVI_OVERNIGHT_MODE=train
TRAIN=$(sbatch --parsable --time=10:00:00 --array=0-1%2 \
  --dependency="afterok:$PREFLIGHT" --export=ALL \
  --output="$AVI_TRAIN_ROOT/logs/train-%A_%a.out" \
  --error="$AVI_TRAIN_ROOT/logs/train-%A_%a.err" \
  "$AVI_OVERNIGHT_CODE/scripts/feniks_avi_overnight.slurm")
TRAIN=${TRAIN%%;*}

export AVI_OVERNIGHT_MODE=prepare-inference
PREPARE=$(sbatch --parsable --time=00:30:00 --gres=gpu:1 --cpus-per-task=12 \
  --dependency="afterok:$TRAIN" --export=ALL \
  --output="$AVI_TRAIN_ROOT/logs/prepare-inference-%j.out" \
  --error="$AVI_TRAIN_ROOT/logs/prepare-inference-%j.err" \
  "$AVI_OVERNIGHT_CODE/scripts/feniks_avi_overnight.slurm")
PREPARE=${PREPARE%%;*}

export AVI_OVERNIGHT_MODE=infer
INFERENCE=$(sbatch --parsable --time=06:00:00 \
  --array="0-4%$INFERENCE_CONCURRENCY" --dependency="afterok:$PREPARE" --export=ALL \
  --output="$AVI_TRAIN_ROOT/logs/infer-%A_%a.out" \
  --error="$AVI_TRAIN_ROOT/logs/infer-%A_%a.err" \
  "$AVI_OVERNIGHT_CODE/scripts/feniks_avi_overnight.slurm")
INFERENCE=${INFERENCE%%;*}

export AVI_OVERNIGHT_MODE=report
REPORT=$(sbatch --parsable --time=06:00:00 --gres=gpu:1 --cpus-per-task=12 \
  --dependency="afterok:$INFERENCE" --export=ALL \
  --output="$AVI_TRAIN_ROOT/logs/report-%j.out" \
  --error="$AVI_TRAIN_ROOT/logs/report-%j.err" \
  "$AVI_OVERNIGHT_CODE/scripts/feniks_avi_overnight.slurm")
REPORT=${REPORT%%;*}

JOBS="$AVI_TRAIN_ROOT/JOBS.env"
printf 'export AVI_SOURCE_ROOT=%q\nexport AVI_PRIOR_ROOT=%q\nexport AVI_TRAIN_ROOT=%q\nexport AVI_INFERENCE_ROOT=%q\nexport AVI_OVERNIGHT_CODE=%q\nexport PREFLIGHT_JOB=%q\nexport TRAIN_JOB=%q\nexport PREPARE_JOB=%q\nexport INFERENCE_JOB=%q\nexport REPORT_JOB=%q\n' \
  "$AVI_SOURCE_ROOT" "$AVI_PRIOR_ROOT" "$AVI_TRAIN_ROOT" \
  "$AVI_INFERENCE_ROOT" "$AVI_OVERNIGHT_CODE" "$PREFLIGHT" "$TRAIN" \
  "$PREPARE" "$INFERENCE" "$REPORT" > "$JOBS"

printf 'preflight=%s\ntraining=%s\nprepare_inference=%s\ninference=%s\nreport=%s\ntrain_root=%s\ninference_root=%s\npeak_h100s=%s\nwatch=bash scripts/watch_feniks_avi_overnight.sh %q\n' \
  "$PREFLIGHT" "$TRAIN" "$PREPARE" "$INFERENCE" "$REPORT" \
  "$AVI_TRAIN_ROOT" "$AVI_INFERENCE_ROOT" "$((4 * INFERENCE_CONCURRENCY))" \
  "$AVI_TRAIN_ROOT"
