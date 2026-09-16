#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
RECOVERY_ROOT="${RECOVERY_ROOT:?Set RECOVERY_ROOT}"
SOURCE_EVAL_ROOT="${EVAL_ROOT:-$RECOVERY_ROOT/epoch_0160_evaluation}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$RECOVERY_ROOT/epoch_0160_catalogue_calibration_v1}"
CALIBRATION_LOG_ROOT="${CALIBRATION_LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$RECOVERY_ROOT")-epoch160-catalogue-calibration}"
NUM_REGIONS="${NUM_REGIONS:-100}"
NUM_BOOTSTRAP="${NUM_BOOTSTRAP:-1000}"

cd "$REPO_DIR"
SOURCE_RECEIPT="$SOURCE_EVAL_ROOT/EPOCH160_EVALUATION_COMPLETE.json"
for path in "$SOURCE_RECEIPT" \
  "$SOURCE_EVAL_ROOT/population/catalogue_selected_truth.parquet" \
  "$SOURCE_EVAL_ROOT/population/posterior_aggregate/raw_q.parquet" \
  "$SOURCE_EVAL_ROOT/population/posterior_aggregate/raw_iw.parquet" \
  "$SOURCE_EVAL_ROOT/population/posterior_aggregate/ema_q.parquet" \
  "$SOURCE_EVAL_ROOT/population/posterior_aggregate/ema_iw.parquet"; do
  test -s "$path" || { echo "missing epoch-160 input: $path" >&2; exit 2; }
done
test ! -e "$CALIBRATION_ROOT" || {
  echo "immutable calibration output exists: $CALIBRATION_ROOT" >&2
  exit 2
}
mkdir -p "$CALIBRATION_ROOT/banks/raw_q256" \
  "$CALIBRATION_ROOT/banks/ema_q256" "$CALIBRATION_LOG_ROOT" \
  "$CACHE_ROOT/jax" outputs/logs

python - "$SOURCE_EVAL_ROOT" "$CALIBRATION_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
receipt = json.loads((source / "EPOCH160_EVALUATION_COMPLETE.json").read_text())
if receipt.get("status") != "DIAGNOSTIC_COMPLETE" or int(receipt.get("epoch", -1)) != 160:
    raise SystemExit("invalid epoch-160 source receipt")
if receipt.get("truth_used_for_training_or_checkpoint_selection") is not False:
    raise SystemExit("source receipt violates the truth-free training contract")

records = {}
for variant in ("raw", "ema"):
    destination = root / "banks" / f"{variant}_q256"
    sources = []
    for shard in range(8):
        inference = source / "catalogue" / variant / f"shard_{shard}" / "exact_gaussian_k256"
        if not (inference.parent / "DONE").is_file():
            raise SystemExit(f"missing completed shard marker: {inference.parent / 'DONE'}")
        files = sorted((inference / "posterior_samples").glob("batch_*.parquet"))
        if not files:
            raise SystemExit(f"missing posterior samples: {inference}")
        sources.extend(files)
    for index, path in enumerate(sources):
        target = destination / f"part_{index:05d}.parquet"
        os.symlink(path.resolve(), target)
    records[f"{variant}_q256"] = {
        "directory": str(destination),
        "files": len(sources),
        "source_draws_per_object": 256,
    }

manifest = {
    "status": "prepared",
    "epoch": 160,
    "source_evaluation_root": str(source),
    "source_receipt": str((source / "EPOCH160_EVALUATION_COMPLETE.json").resolve()),
    "catalogue_objects": int(receipt["catalogue_objects"]),
    "cohort": receipt["catalogue_cohort"],
    "truth_role": "post-freeze synthetic closure diagnostics only",
    "truth_used_for_training_or_checkpoint_selection": False,
    "banks": records,
}
(root / "CATALOGUE_CALIBRATION_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY

EXPORTS="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,RECOVERY_ROOT=$RECOVERY_ROOT,SOURCE_EVAL_ROOT=$SOURCE_EVAL_ROOT,CALIBRATION_ROOT=$CALIBRATION_ROOT,CACHE_ROOT=$CACHE_ROOT,NUM_REGIONS=$NUM_REGIONS,NUM_BOOTSTRAP=$NUM_BOOTSTRAP"
CALIBRATION_RAW=$(sbatch --parsable --array=0-3%4 \
  --output="$CALIBRATION_LOG_ROOT/calibration-%A_%a.out" \
  --error="$CALIBRATION_LOG_ROOT/calibration-%A_%a.err" \
  --export="$EXPORTS" \
  scripts/feniks_sc_drws_epoch160_catalogue_calibration_h100.slurm)
CALIBRATION_JOB="${CALIBRATION_RAW%%;*}"
FINAL_RAW=$(sbatch --parsable --dependency="afterok:$CALIBRATION_JOB" \
  --output="$CALIBRATION_LOG_ROOT/finalize-%j.out" \
  --error="$CALIBRATION_LOG_ROOT/finalize-%j.err" \
  --export="$EXPORTS" \
  scripts/feniks_sc_drws_epoch160_catalogue_calibration_finalize.slurm)
FINAL_JOB="${FINAL_RAW%%;*}"
ALL_JOBS="$CALIBRATION_JOB,$FINAL_JOB"
LATEST="outputs/logs/feniks_sc_drws_epoch160_catalogue_calibration_latest.env"
TEMPORARY="${LATEST}.tmp.$$"
printf 'export CALIBRATION_JOB=%q\nexport FINAL_JOB=%q\nexport ALL_JOBS=%q\nexport RECOVERY_ROOT=%q\nexport SOURCE_EVAL_ROOT=%q\nexport CALIBRATION_ROOT=%q\nexport LOG_ROOT=%q\n' \
  "$CALIBRATION_JOB" "$FINAL_JOB" "$ALL_JOBS" "$RECOVERY_ROOT" \
  "$SOURCE_EVAL_ROOT" "$CALIBRATION_ROOT" "$CALIBRATION_LOG_ROOT" > "$TEMPORARY"
mv "$TEMPORARY" "$LATEST"

echo "calibration_job=$CALIBRATION_JOB (4 tasks: common32 MIRA/TARP, q256 MIRA/TARP)"
echo "final_job=$FINAL_JOB"
echo "all_jobs=$ALL_JOBS"
echo "calibration_root=$CALIBRATION_ROOT"
echo "latest_env=$LATEST"
