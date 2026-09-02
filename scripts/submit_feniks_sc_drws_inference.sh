#!/bin/bash
set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
RECOVERY_ROOT="${RECOVERY_ROOT:?Set RECOVERY_ROOT}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$RECOVERY_ROOT/manifests}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
INFERENCE_ROOT="${INFERENCE_ROOT:-$RECOVERY_ROOT/final_inference}"
LOG_ROOT="${LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$RECOVERY_ROOT")-inference}"
AFTER_JOB="${AFTER_JOB:-}"
ALLOW_DIAGNOSTIC_FULL="${ALLOW_DIAGNOSTIC_FULL:-0}"
cd "$REPO_DIR"
if [[ -z "$AFTER_JOB" ]]; then
  if [[ "$ALLOW_DIAGNOSTIC_FULL" == "1" ]]; then
    test -s "$RECOVERY_ROOT/FULL_TRAIN_PASS.json" \
      || test -s "$RECOVERY_ROOT/FULL_TRAIN_FAIL.json" \
      || { echo "missing frozen full receipt" >&2; exit 2; }
  else
    test -s "$RECOVERY_ROOT/FULL_TRAIN_PASS.json" \
      || { echo "missing FULL_TRAIN_PASS.json" >&2; exit 2; }
  fi
fi
test ! -e "$INFERENCE_ROOT" || { echo "immutable output exists: $INFERENCE_ROOT"; exit 2; }
mkdir -p "$INFERENCE_ROOT/manifests" "$LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs
python - "$MANIFEST_ROOT/full_train_indices.npy" "$INFERENCE_ROOT/manifests" <<'PY'
import numpy as np,sys
values=np.load(sys.argv[1])
for i, shard in enumerate(np.array_split(values, 4)):
    np.save(f"{sys.argv[2]}/shard_{i}.npy", shard, allow_pickle=False)
PY
EXPORTS="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,CATALOG_DIR=$CATALOG_DIR,RECOVERY_ROOT=$RECOVERY_ROOT,INFERENCE_ROOT=$INFERENCE_ROOT,CACHE_ROOT=$CACHE_ROOT,ALLOW_DIAGNOSTIC_FULL=$ALLOW_DIAGNOSTIC_FULL"
DEPENDENCY_ARGS=()
if [[ -n "$AFTER_JOB" ]]; then
  DEPENDENCY_ARGS+=(--dependency="afterok:$AFTER_JOB")
fi
RAW=$(sbatch --parsable "${DEPENDENCY_ARGS[@]}" --array=0-3%4 --output="$LOG_ROOT/infer-%A_%a.out" \
  --error="$LOG_ROOT/infer-%A_%a.err" --export="$EXPORTS" scripts/feniks_sc_drws_inference_h100.slurm)
INFERENCE_JOB="${RAW%%;*}"
GATE_RAW=$(sbatch --parsable --dependency="afterok:$INFERENCE_JOB" \
  --output="$LOG_ROOT/infer-gate-%j.out" --error="$LOG_ROOT/infer-gate-%j.err" \
  --export="$EXPORTS" scripts/feniks_sc_drws_inference_finalize.slurm)
INFERENCE_GATE_JOB="${GATE_RAW%%;*}"
LATEST=outputs/logs/feniks_sc_drws_inference_latest.env
printf 'export INFERENCE_JOB=%q\nexport INFERENCE_GATE_JOB=%q\nexport RECOVERY_ROOT=%q\nexport INFERENCE_ROOT=%q\nexport LOG_ROOT=%q\n' \
  "$INFERENCE_JOB" "$INFERENCE_GATE_JOB" "$RECOVERY_ROOT" "$INFERENCE_ROOT" "$LOG_ROOT" > "$LATEST"
echo "inference_job=$INFERENCE_JOB"
echo "inference_gate_job=$INFERENCE_GATE_JOB"
echo "upstream_dependency=${AFTER_JOB:-none}"
echo "diagnostic_fallback=$ALLOW_DIAGNOSTIC_FULL"
echo "latest_env=$LATEST"
