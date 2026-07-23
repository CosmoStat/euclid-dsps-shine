#!/bin/bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_selfsup_learned_t2_jaxcosmo_v1}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LABEL="selfsup_smcwake_mix2_k4_t2"
TASK_DIR="$ROOT_DIR/$LABEL"
FAILED_TRAIN="$TASK_DIR/train"
ARCHIVE_TRAIN="$TASK_DIR/train_failed_epoch40"
INITIAL_SIDECAR="$FAILED_TRAIN/checkpoints/last.eqx.json"
RWS_LABELS=(selfsup_rws_k8_t2 selfsup_rws_mix2_k8_t2)

for label in "${RWS_LABELS[@]}"; do
  test -f "$ROOT_DIR/$label/DONE" || {
    echo "[recovery-submit][error] completed task missing: $label"; exit 2;
  }
done
test -s "$INITIAL_SIDECAR" || {
  echo "[recovery-submit][error] missing: $INITIAL_SIDECAR"; exit 2;
}
test ! -e "$ARCHIVE_TRAIN" || {
  echo "[recovery-submit][error] archive already exists: $ARCHIVE_TRAIN"; exit 2;
}
test ! -e "$TASK_DIR/inference" || {
  echo "[recovery-submit][error] inference already exists: $TASK_DIR/inference"; exit 2;
}
test ! -f "$TASK_DIR/DONE" || {
  echo "[recovery-submit][error] SMC task is already complete"; exit 2;
}

source_epoch=$(CHECKPOINT_JSON="$INITIAL_SIDECAR" python - <<'PY'
import json
import os

path = os.environ["CHECKPOINT_JSON"]
epoch = int(json.load(open(path, encoding="utf-8"))["epoch"])
if not 1 <= epoch < 40:
    raise SystemExit(
        f"[recovery-submit][error] {path}: checkpoint epoch {epoch}, "
        "expected an epoch in [1, 39]"
    )
print(epoch)
PY
)
start_epoch=$((source_epoch + 1))
printf '[recovery-submit] source checkpoint epoch=%s; resume starts at %s\n' \
  "$source_epoch" "$start_epoch"

mkdir -p "$LOG_DIR"
mv "$FAILED_TRAIN" "$ARCHIVE_TRAIN"
submitted=0
restore_archive() {
  if [[ "$submitted" == "0" && ! -e "$FAILED_TRAIN" && -d "$ARCHIVE_TRAIN" ]]; then
    mv "$ARCHIVE_TRAIN" "$FAILED_TRAIN"
  fi
}
trap restore_archive EXIT

resume_raw=$(sbatch --parsable \
  --export="ALL,ROOT_DIR=$ROOT_DIR,SOURCE_TRAIN=$ARCHIVE_TRAIN,START_EPOCH=$start_epoch,END_EPOCH=40" \
  scripts/feniks_selfsup_production_smc_resume_h100.slurm)
resume_job="${resume_raw%%;*}"
submitted=1

lens_raw=$(sbatch --parsable \
  --dependency="afterok:${resume_job}" \
  --export="ALL,SMOKE=0,ROOT_DIR=$ROOT_DIR" \
  scripts/feniks_selfsup_production_jlens_h100.slurm)
lens_job="${lens_raw%%;*}"

finalize_raw=$(sbatch --parsable \
  --dependency="afterok:${lens_job}" \
  --export="ALL,ROOT_DIR=$ROOT_DIR" \
  scripts/feniks_selfsup_production_finalize_h100.slurm)
finalize_job="${finalize_raw%%;*}"

stamp=$(date +%Y%m%d_%H%M%S)
submission_log="$LOG_DIR/submit_selfsup_production_recovery_${stamp}.log"
{
  printf 'root=%s\n' "$ROOT_DIR"
  printf 'source_train=%s\n' "$ARCHIVE_TRAIN"
  printf 'source_epoch=%s start_epoch=%s end_epoch=40\n' \
    "$source_epoch" "$start_epoch"
  printf 'smc_resume=%s lens=%s finalize=%s\n' \
    "$resume_job" "$lens_job" "$finalize_job"
} | tee "$submission_log"
printf 'monitor: squeue -j %s,%s,%s\n' \
  "$resume_job" "$lens_job" "$finalize_job"
printf 'submission_log=%s\n' "$submission_log"
