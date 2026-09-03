#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
RECOVERY_ROOT="${RECOVERY_ROOT:?Set RECOVERY_ROOT by sourcing the epoch-160 environment}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$RECOVERY_ROOT/manifests}"
EPOCH160_ROOT="${EPOCH160_ROOT:-${EVAL_ROOT:-$RECOVERY_ROOT/epoch_0160_evaluation}}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
VEM_ROOT="${VEM_ROOT:-$RECOVERY_ROOT/population_vem_epoch160_v1}"
VEM_LOG_ROOT="${VEM_LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$RECOVERY_ROOT")-population-vem}"
CONFIG="configs/experiments/feniks_sc_drws_r29_current_production.yaml"
TRUTH_CONFIG="configs/experiments/feniks_sc_drws_r29_truth_closure.yaml"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[population-vem][error] tracked source changes are not committed" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
for path in "$CONFIG" "$TRUTH_CONFIG" \
  "$EPOCH160_ROOT/CHECKPOINT_FROZEN.json" \
  "$CATALOG_DIR/train.parquet" "$CATALOG_DIR/test.parquet" \
  "$MANIFEST_ROOT/manifest.json" \
  "$MANIFEST_ROOT/full_train_indices.npy" \
  "$MANIFEST_ROOT/full_test_indices.npy"; do
  test -s "$path" || { echo "[population-vem][error] missing: $path" >&2; exit 2; }
done
if [[ -s "$VEM_ROOT/SUBMISSION.json" ]]; then
  echo "[population-vem][error] this immutable run was already submitted: $VEM_ROOT" >&2
  echo "Use its saved environment and monitor; do not duplicate the chain." >&2
  exit 2
fi
mkdir -p "$VEM_LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs

python scripts/prepare_feniks_sc_drws_population_vem.py \
  --config "$CONFIG" --truth-config "$TRUTH_CONFIG" \
  --freeze-receipt "$EPOCH160_ROOT/CHECKPOINT_FROZEN.json" \
  --train-catalog "$CATALOG_DIR/train.parquet" \
  --test-catalog "$CATALOG_DIR/test.parquet" \
  --train-indices "$MANIFEST_ROOT/full_train_indices.npy" \
  --test-indices "$MANIFEST_ROOT/full_test_indices.npy" \
  --selection-manifest "$MANIFEST_ROOT/manifest.json" \
  --source-variant raw --out "$VEM_ROOT"

JOB_REPO_DIR="${VEM_CODE_ROOT:-$CACHE_ROOT/code/population-vem-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")"
if [[ -e "$JOB_REPO_DIR" ]]; then
  EXISTING_COMMIT="$(git -C "$JOB_REPO_DIR" rev-parse HEAD)"
  if [[ "$EXISTING_COMMIT" != "$CODE_COMMIT" ]]; then
    echo "[population-vem][error] code snapshot has wrong commit: $JOB_REPO_DIR" >&2
    exit 2
  fi
else
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi
test -e "$JOB_REPO_DIR/Data/diffsky"

EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,VEM_ROOT=$VEM_ROOT,CACHE_ROOT=$CACHE_ROOT"
BANK_RAW=$(sbatch --parsable --array=0-35%24 \
  --output="$VEM_LOG_ROOT/bank-%A_%a.out" \
  --error="$VEM_LOG_ROOT/bank-%A_%a.err" \
  --export="$EXPORTS,VEM_STAGE=initial" \
  scripts/feniks_sc_drws_population_vem_bank_h100.slurm)
BANK_JOB="${BANK_RAW%%;*}"
BANK_GATE_RAW=$(sbatch --parsable --dependency="afterok:$BANK_JOB" \
  --output="$VEM_LOG_ROOT/bank-gate-%j.out" \
  --error="$VEM_LOG_ROOT/bank-gate-%j.err" --export="$EXPORTS" \
  scripts/feniks_sc_drws_population_vem_bank_finalize.slurm)
BANK_GATE_JOB="${BANK_GATE_RAW%%;*}"
PRIOR_RAW=$(sbatch --parsable --dependency="afterok:$BANK_GATE_JOB" \
  --output="$VEM_LOG_ROOT/prior-%j.out" \
  --error="$VEM_LOG_ROOT/prior-%j.err" --export="$EXPORTS" \
  scripts/feniks_sc_drws_population_vem_prior_h100.slurm)
PRIOR_JOB="${PRIOR_RAW%%;*}"
REFRESH_RAW=$(sbatch --parsable --dependency="afterok:$PRIOR_JOB" \
  --output="$VEM_LOG_ROOT/refresh-%j.out" \
  --error="$VEM_LOG_ROOT/refresh-%j.err" --export="$EXPORTS" \
  scripts/feniks_sc_drws_population_vem_refresh_h100.slurm)
REFRESH_JOB="${REFRESH_RAW%%;*}"
EVAL_RAW=$(sbatch --parsable --dependency="afterok:$REFRESH_JOB" \
  --array=0-15%16 --output="$VEM_LOG_ROOT/eval-%A_%a.out" \
  --error="$VEM_LOG_ROOT/eval-%A_%a.err" \
  --export="$EXPORTS,VEM_STAGE=final" \
  scripts/feniks_sc_drws_population_vem_bank_h100.slurm)
EVAL_JOB="${EVAL_RAW%%;*}"
FINAL_RAW=$(sbatch --parsable --dependency="afterok:$EVAL_JOB" \
  --output="$VEM_LOG_ROOT/final-%j.out" \
  --error="$VEM_LOG_ROOT/final-%j.err" --export="$EXPORTS" \
  scripts/feniks_sc_drws_population_vem_finalize_h100.slurm)
FINAL_JOB="${FINAL_RAW%%;*}"
ALL_JOBS="$BANK_JOB,$BANK_GATE_JOB,$PRIOR_JOB,$REFRESH_JOB,$EVAL_JOB,$FINAL_JOB"
LATEST="outputs/logs/feniks_sc_drws_population_vem_latest.env"

printf 'export BANK_JOB=%q\nexport BANK_GATE_JOB=%q\nexport PRIOR_JOB=%q\nexport REFRESH_JOB=%q\nexport EVAL_JOB=%q\nexport FINAL_JOB=%q\nexport ALL_JOBS=%q\nexport VEM_ROOT=%q\nexport VEM_LOG_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\n' \
  "$BANK_JOB" "$BANK_GATE_JOB" "$PRIOR_JOB" "$REFRESH_JOB" \
  "$EVAL_JOB" "$FINAL_JOB" "$ALL_JOBS" "$VEM_ROOT" "$VEM_LOG_ROOT" \
  "$RECOVERY_ROOT" "$JOB_REPO_DIR" "$CODE_COMMIT" > "$LATEST"

python - "$VEM_ROOT" "$BANK_JOB" "$BANK_GATE_JOB" "$PRIOR_JOB" \
  "$REFRESH_JOB" "$EVAL_JOB" "$FINAL_JOB" "$ALL_JOBS" \
  "$JOB_REPO_DIR" "$CODE_COMMIT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
payload={
    'status':'SUBMITTED',
    'bank_job':sys.argv[2],
    'bank_gate_job':sys.argv[3],
    'prior_job':sys.argv[4],
    'refresh_job':sys.argv[5],
    'evaluation_job':sys.argv[6],
    'final_job':sys.argv[7],
    'all_jobs':sys.argv[8],
    'code_snapshot':sys.argv[9],
    'code_commit':sys.argv[10],
}
(root/'SUBMISSION.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY

echo "bank_job=$BANK_JOB (36 one-H100 tasks, max 24 concurrent)"
echo "bank_gate_job=$BANK_GATE_JOB"
echo "prior_job=$PRIOR_JOB (4 H100, DSPS-free optimizer)"
echo "refresh_job=$REFRESH_JOB (4 H100, 2 epochs x 2 draws)"
echo "evaluation_job=$EVAL_JOB (16 one-H100 tasks, max 16 concurrent)"
echo "final_job=$FINAL_JOB (one H100 for MIRA/TARP + plots)"
echo "root=$VEM_ROOT"
echo "code_snapshot=$JOB_REPO_DIR ($CODE_COMMIT)"
echo "latest_env=$LATEST"
echo "monitor: bash scripts/monitor_feniks_sc_drws_population_vem.sh"
