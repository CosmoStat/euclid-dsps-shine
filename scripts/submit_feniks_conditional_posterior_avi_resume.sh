#!/bin/bash
set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-outputs/runs/feniks_conditional_posterior_jaxcosmo_v1}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_conditional_posterior_jaxcosmo_v1_completed_v2}"
LABELS=(avi_gaussian_x avi_gaussian_u avi_realnvp avi_rqspline)
SOURCE_EPOCHS=(91 91 91 90)
NPE_LABELS=(npe_gaussian_x npe_gaussian_u npe_realnvp npe_rqspline)

test ! -e "$ROOT_DIR" || {
  echo "[submit][error] refusing to reuse existing output: $ROOT_DIR"; exit 2;
}
for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  checkpoint="$SOURCE_ROOT/$label/train/checkpoints/last.eqx"
  test -s "$checkpoint" || { echo "[submit][error] missing: $checkpoint"; exit 2; }
  test -s "$checkpoint.json" || { echo "[submit][error] missing: $checkpoint.json"; exit 2; }
  CHECKPOINT_JSON="$checkpoint.json" EXPECTED_EPOCH="${SOURCE_EPOCHS[$index]}" python - <<'PY'
import json
import os

path = os.environ["CHECKPOINT_JSON"]
expected = int(os.environ["EXPECTED_EPOCH"])
epoch = int(json.load(open(path, encoding="utf-8"))["epoch"])
if epoch != expected:
    raise SystemExit(
        f"[submit][error] {path}: checkpoint epoch {epoch}, expected {expected}"
    )
PY
done
for label in "${NPE_LABELS[@]}"; do
  test -f "$SOURCE_ROOT/$label/DONE" || {
    echo "[submit][error] incomplete source NPE task: $label"; exit 2;
  }
done

mkdir -p outputs/logs "$ROOT_DIR"
submitted=0
cleanup_unsubmitted_root() {
  if [[ "$submitted" == "0" ]]; then
    for label in "${NPE_LABELS[@]}"; do
      rm -f "$ROOT_DIR/$label"
    done
    rmdir "$ROOT_DIR" 2>/dev/null || true
  fi
}
trap cleanup_unsubmitted_root EXIT
for label in "${NPE_LABELS[@]}"; do
  target=$(realpath --relative-to="$ROOT_DIR" "$SOURCE_ROOT/$label")
  ln -s "$target" "$ROOT_DIR/$label"
done

job_raw=$(sbatch --parsable \
  --export="ALL,SOURCE_ROOT=$SOURCE_ROOT,ROOT_DIR=$ROOT_DIR" \
  scripts/feniks_conditional_posterior_avi_resume_h100.slurm)
job_id="${job_raw%%;*}"
submitted=1

printf 'avi_resume_array=%s tasks=0-3 gpus_per_task=4 max_concurrent=4\n' "$job_id"
printf 'source_root=%s\ncompleted_root=%s\n' "$SOURCE_ROOT" "$ROOT_DIR"
printf 'monitor: squeue -j %s\n' "$job_id"
printf 'accounting: sacct -j %s --format=JobID%%22,State%%20,ExitCode,Elapsed,AllocTRES%%55\n' "$job_id"
