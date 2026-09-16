#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_DIR/outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536/bands26/full_cont120}"
SMC_ROOT="${SMC_ROOT:?Set SMC_ROOT to the completed SMC512 root}"
RUN_TAG="${RUN_TAG:-smc_direct_em_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$SMC_ROOT/$RUN_TAG}"
EM_DEPENDENCY="${EM_DEPENDENCY:-}"

cd "$REPO_DIR"
mkdir -p outputs/logs
for path in \
  "$SOURCE_ROOT/train/checkpoints/best.eqx" \
  "$SMC_ROOT/pilot_selection/DONE" \
  "$SMC_ROOT/pilot_selection/selection_summary.json" \
  "$SMC_ROOT/floor_0p05/seed_260817/DONE" \
  "$SMC_ROOT/floor_0p05/seed_260818/DONE"; do
  test -e "$path" || {
    echo "[smc-empirical-bayes][error] missing prerequisite: $path" >&2
    exit 2
  }
done
test ! -e "$OUT" || {
  echo "[smc-empirical-bayes][error] output already exists: $OUT" >&2
  exit 2
}

dependency_args=()
if [[ -n "$EM_DEPENDENCY" ]]; then
  dependency_args=("--dependency=afterok:$EM_DEPENDENCY")
fi
export REPO_DIR MINICONDA_PATH CONDA_ENV SOURCE_ROOT SMC_ROOT OUT
raw=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",SOURCE_ROOT="$SOURCE_ROOT",SMC_ROOT="$SMC_ROOT",OUT="$OUT" \
  scripts/popcosmos_smc_empirical_bayes_h100.slurm)
SMC_EM_JOB="${raw%%;*}"
SMC_EM_OUT="$OUT"

receipt=outputs/logs/popcosmos_smc_empirical_bayes_latest.env
printf 'export SMC_EM_JOB=%q\nexport SMC_EM_OUT=%q\nexport SMC_ROOT=%q\nexport SOURCE_ROOT=%q\n' \
  "$SMC_EM_JOB" "$SMC_EM_OUT" "$SMC_ROOT" "$SOURCE_ROOT" > "$receipt"

echo "smc_em_job=$SMC_EM_JOB"
echo "smc_em_out=$SMC_EM_OUT"
echo "population_contract=selected_catalog"
echo "receipt=$receipt"
echo "monitor: squeue -j $SMC_EM_JOB"
echo "report: python scripts/report_popcosmos_smc_empirical_bayes.py --root $SMC_EM_OUT"
