#!/bin/bash
set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${WORK:?Set WORK or REPO_DIR}/dsps-popcosmos}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the completed full_cont120 run}"
SMC_ROOT="${SMC_ROOT:?Set SMC_ROOT to a completed adaptive-SMC pilot}"
BASE_COMPONENTS="${BASE_COMPONENTS:-1}"
MIXTURE_MEAN_OFFSET="${MIXTURE_MEAN_OFFSET:-0.05}"
if [[ "$BASE_COMPONENTS" == 1 ]]; then
  OUT="${OUT:-$SMC_ROOT/proposal_refresh}"
else
  OUT="${OUT:-$SMC_ROOT/proposal_refresh_mix${BASE_COMPONENTS}_k${PROBE_SAMPLES:-2048}}"
fi
BASELINE_REFRESH_OUT="${BASELINE_REFRESH_OUT:-}"
REFRESH_DEPENDENCY="${REFRESH_DEPENDENCY:-}"
cd "$REPO_DIR"
mkdir -p outputs/logs
sbatch_args=(--parsable)
if [[ -n "$REFRESH_DEPENDENCY" ]]; then
  sbatch_args+=(--dependency="afterok:${REFRESH_DEPENDENCY}")
fi
raw=$(sbatch "${sbatch_args[@]}" \
  --export=ALL,REPO_DIR="$REPO_DIR",MINICONDA_PATH="$MINICONDA_PATH",CONDA_ENV="$CONDA_ENV",SOURCE_ROOT="$SOURCE_ROOT",SMC_ROOT="$SMC_ROOT",OUT="$OUT",REFRESH_EPOCHS="${REFRESH_EPOCHS:-20}",PROBE_SAMPLES="${PROBE_SAMPLES:-2048}",MIN_MEDIAN_ESS_FRACTION="${MIN_MEDIAN_ESS_FRACTION:-0.05}",MAX_FRACTION_PARETO_K_GT_0P7="${MAX_FRACTION_PARETO_K_GT_0P7:-0.2}",BASE_COMPONENTS="$BASE_COMPONENTS",MIXTURE_MEAN_OFFSET="$MIXTURE_MEAN_OFFSET",BASELINE_REFRESH_OUT="$BASELINE_REFRESH_OUT" \
  scripts/popcosmos_posthoc_smc_refresh_h100.slurm)
job="${raw%%;*}"
env_file=outputs/logs/popcosmos_posthoc_smc_refresh_latest.env
printf 'export SMC_REFRESH_JOB=%q\nexport SMC_REFRESH_OUT=%q\nexport SMC_ROOT=%q\nexport SOURCE_ROOT=%q\nexport BASE_COMPONENTS=%q\nexport BASELINE_REFRESH_OUT=%q\nexport REFRESH_DEPENDENCY=%q\n' \
  "$job" "$OUT" "$SMC_ROOT" "$SOURCE_ROOT" "$BASE_COMPONENTS" "$BASELINE_REFRESH_OUT" "$REFRESH_DEPENDENCY" > "$env_file"
echo "smc_refresh_job=$job"
echo "smc_refresh_out=$OUT"
echo "base_components=$BASE_COMPONENTS"
echo "dependency=${REFRESH_DEPENDENCY:-none}"
echo "monitor: squeue -j $job"
echo "log: outputs/logs/cosmos_qfix-${job}.out"
echo "latest_env=$env_file"
