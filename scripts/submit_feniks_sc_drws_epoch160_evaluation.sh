#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
RECOVERY_ROOT="${RECOVERY_ROOT:?Set RECOVERY_ROOT}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$RECOVERY_ROOT/manifests}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
EVAL_ROOT="${EVAL_ROOT:-$RECOVERY_ROOT/epoch_0160_evaluation}"
EPOCH160_LOG_ROOT="${EPOCH160_LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$RECOVERY_ROOT")-epoch160}"
LOG_ROOT="$EPOCH160_LOG_ROOT"
SELECTED_CANDIDATE="${SELECTED_CANDIDATE:-current_residual_6x256}"
SEED="${SEED:-260826}"

cd "$REPO_DIR"
test "$SELECTED_CANDIDATE" = "current_residual_6x256" || {
  echo "epoch-160 evaluator currently requires current_residual_6x256" >&2
  exit 2
}
test "$SEED" = "260826" || { echo "epoch-160 evaluator currently requires seed 260826" >&2; exit 2; }
for path in "$MANIFEST_ROOT/manifest.json" \
  "$MANIFEST_ROOT/final_validation_indices.npy" \
  "$MANIFEST_ROOT/full_test_indices.npy" "$CATALOG_DIR/train.parquet" \
  "$CATALOG_DIR/test.parquet"; do
  test -s "$path" || { echo "missing input: $path" >&2; exit 2; }
done
test ! -e "$EVAL_ROOT" || { echo "immutable evaluation root exists: $EVAL_ROOT" >&2; exit 2; }
mkdir -p "$EVAL_ROOT/manifests" "$LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs

python - "$MANIFEST_ROOT/final_validation_indices.npy" \
  "$MANIFEST_ROOT/full_test_indices.npy" "$EVAL_ROOT/manifests" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


heldout_path = Path(sys.argv[1])
catalogue_path = Path(sys.argv[2])
out = Path(sys.argv[3])
heldout = np.load(heldout_path, allow_pickle=False).astype(np.int64)
catalogue = np.load(catalogue_path, allow_pickle=False).astype(np.int64)
if len(heldout) != 512:
    raise SystemExit(f"expected 512 held-out objects, found {len(heldout)}")
records = {"heldout": [], "catalogue": []}
for label, values, shards in (("heldout", heldout, 4), ("catalogue", catalogue, 8)):
    for index, shard in enumerate(np.array_split(values, shards)):
        path = out / f"{label}_shard_{index}.npy"
        np.save(path, shard, allow_pickle=False)
        records[label].append(
            {"path": str(path.resolve()), "rows": len(shard), "sha256": sha256(path)}
        )
payload = {
    "status": "frozen",
    "epoch": 160,
    "heldout_objects": len(heldout),
    "heldout_shards": 4,
    "heldout_draws_per_object": 1024,
    "catalogue_objects": len(catalogue),
    "catalogue_shards": 8,
    "catalogue_draws_per_object": 256,
    "variants": ["raw", "ema"],
    "truth_used": False,
    "records": records,
}
(out / "evaluation_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

EXPORTS="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,CATALOG_DIR=$CATALOG_DIR,RECOVERY_ROOT=$RECOVERY_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT,CACHE_ROOT=$CACHE_ROOT,EVAL_ROOT=$EVAL_ROOT,SELECTED_CANDIDATE=$SELECTED_CANDIDATE,SEED=$SEED"
WAIT_RAW=$(sbatch --parsable --output="$LOG_ROOT/epoch160-wait-%j.out" \
  --error="$LOG_ROOT/epoch160-wait-%j.err" --export="$EXPORTS" \
  scripts/feniks_sc_drws_epoch160_wait.slurm)
WAIT_JOB="${WAIT_RAW%%;*}"
HELDOUT_RAW=$(sbatch --parsable --dependency="afterok:$WAIT_JOB" --array=0-7%8 \
  --output="$LOG_ROOT/epoch160-heldout-%A_%a.out" \
  --error="$LOG_ROOT/epoch160-heldout-%A_%a.err" --export="$EXPORTS" \
  scripts/feniks_sc_drws_epoch160_heldout_h100.slurm)
HELDOUT_JOB="${HELDOUT_RAW%%;*}"
CATALOGUE_RAW=$(sbatch --parsable --dependency="afterok:$WAIT_JOB" --array=0-15%16 \
  --output="$LOG_ROOT/epoch160-catalogue-%A_%a.out" \
  --error="$LOG_ROOT/epoch160-catalogue-%A_%a.err" --export="$EXPORTS" \
  scripts/feniks_sc_drws_epoch160_catalogue_h100.slurm)
CATALOGUE_JOB="${CATALOGUE_RAW%%;*}"
FINAL_RAW=$(sbatch --parsable \
  --dependency="afterok:$HELDOUT_JOB:$CATALOGUE_JOB" \
  --output="$LOG_ROOT/epoch160-finalize-%j.out" \
  --error="$LOG_ROOT/epoch160-finalize-%j.err" --export="$EXPORTS" \
  scripts/feniks_sc_drws_epoch160_finalize_h100.slurm)
FINAL_JOB="${FINAL_RAW%%;*}"
ALL_JOBS="$WAIT_JOB,$HELDOUT_JOB,$CATALOGUE_JOB,$FINAL_JOB"
LATEST="outputs/logs/feniks_sc_drws_epoch160_evaluation_latest.env"
printf 'export WAIT_JOB=%q\nexport HELDOUT_JOB=%q\nexport CATALOGUE_JOB=%q\nexport FINAL_JOB=%q\nexport ALL_JOBS=%q\nexport RECOVERY_ROOT=%q\nexport MANIFEST_ROOT=%q\nexport EVAL_ROOT=%q\nexport LOG_ROOT=%q\n' \
  "$WAIT_JOB" "$HELDOUT_JOB" "$CATALOGUE_JOB" "$FINAL_JOB" "$ALL_JOBS" \
  "$RECOVERY_ROOT" "$MANIFEST_ROOT" "$EVAL_ROOT" "$LOG_ROOT" > "$LATEST"

echo "wait_job=$WAIT_JOB"
echo "heldout_job=$HELDOUT_JOB (8 tasks: raw/EMA x 4 shards, K=1024)"
echo "catalogue_job=$CATALOGUE_JOB (16 tasks: raw/EMA x 8 shards, K=256)"
echo "final_job=$FINAL_JOB"
echo "all_jobs=$ALL_JOBS"
echo "evaluation_root=$EVAL_ROOT"
echo "latest_env=$LATEST"
