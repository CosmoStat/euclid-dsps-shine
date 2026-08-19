#!/bin/bash
#SBATCH --job-name=cosmos_qfix
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT}"
SMC_ROOT="${SMC_ROOT:?Set SMC_ROOT}"
OUT="${OUT:?Set OUT}"
REFRESH_EPOCHS="${REFRESH_EPOCHS:-20}"
PROBE_SAMPLES="${PROBE_SAMPLES:-2048}"
MIN_MEDIAN_ESS_FRACTION="${MIN_MEDIAN_ESS_FRACTION:-0.05}"
MAX_FRACTION_PARETO_K_GT_0P7="${MAX_FRACTION_PARETO_K_GT_0P7:-0.2}"
BASE_COMPONENTS="${BASE_COMPONENTS:-1}"
MIXTURE_MEAN_OFFSET="${MIXTURE_MEAN_OFFSET:-0.05}"
BASELINE_REFRESH_OUT="${BASELINE_REFRESH_OUT:-}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export MPLBACKEND=Agg

cd "$REPO_DIR"
mkdir -p outputs/logs
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

SELECTION="$SMC_ROOT/pilot_selection/selection_summary.json"
test -s "$SELECTION"
variant=$(python - "$SELECTION" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
if payload.get("selection_status") != "PASS" or not payload.get("selected_variant"):
    raise SystemExit("SMC pilot did not select an adequate likelihood")
print(payload["selected_variant"])
PY
)
case "$variant" in
  floor_0p00) SOURCE_CONFIG=configs/experiments/popcosmos_native15d_rws.yaml ;;
  floor_0p02) SOURCE_CONFIG=configs/experiments/popcosmos_native15d_rws_floor02.yaml ;;
  floor_0p05) SOURCE_CONFIG=configs/experiments/popcosmos_native15d_rws_floor05.yaml ;;
  *) echo "[proposal-refresh][error] unsupported selected variant: $variant" >&2; exit 2 ;;
esac
case "$BASE_COMPONENTS" in
  1) CONFIG="$SOURCE_CONFIG" ;;
  2)
    if [[ "$variant" != floor_0p05 ]]; then
      echo "[proposal-refresh][error] mix2 is frozen to selected floor_0p05" >&2
      exit 2
    fi
    CONFIG=configs/experiments/popcosmos_native15d_rws_floor05_mix2.yaml
    ;;
  *) echo "[proposal-refresh][error] BASE_COMPONENTS must be 1 or 2" >&2; exit 2 ;;
esac

DATASET=Data/cosmos2020/prepared/farmer_a24_full.parquet
SOURCE_CHECKPOINT="$SOURCE_ROOT/train/checkpoints/best.eqx"
SOURCE_STATS="$SOURCE_ROOT/train/feature_stats.json"
BANK_A="$SMC_ROOT/$variant/seed_260817"
BANK_B="$SMC_ROOT/$variant/seed_260818"
REFRESH="$OUT/encoder_refresh"
PROPOSAL="$OUT/moderate_k${PROBE_SAMPLES}_proposal"
CORRECTION="$OUT/moderate_k${PROBE_SAMPLES}_importance"
ROW_INDICES="$SMC_ROOT/cohorts/proposal_probe_indices.npy"
test ! -e "$OUT"
mkdir -p "$OUT"
test -s "$ROW_INDICES"
PROBE_LIMIT=$(python - "$ROW_INDICES" <<'PY'
import numpy as np
import sys
print(len(np.load(sys.argv[1])))
PY
)
if [[ "$BASE_COMPONENTS" -gt 1 ]]; then
  test -n "$BASELINE_REFRESH_OUT"
  BASELINE_IMPORTANCE="$BASELINE_REFRESH_OUT/moderate_k${PROBE_SAMPLES}_importance/importance_summary.json"
  BASELINE_PROBE_INDICES="$BASELINE_REFRESH_OUT/moderate_k${PROBE_SAMPLES}_proposal/inference_indices.npy"
  test -s "$BASELINE_IMPORTANCE"
  test -s "$BASELINE_PROBE_INDICES"
  python - "$ROW_INDICES" "$BASELINE_PROBE_INDICES" <<'PY'
import numpy as np
import sys

expected, baseline = (np.load(path) for path in sys.argv[1:])
if not np.array_equal(expected, baseline):
    raise SystemExit("mixture candidate and unimodal baseline probe cohorts differ")
PY
else
  BASELINE_IMPORTANCE=""
fi
python - "$BANK_A/row_indices.npy" "$ROW_INDICES" "$SOURCE_ROOT/inference/inference_indices.npy" <<'PY'
import numpy as np
import sys
smc, probe, evaluation = (np.load(path) for path in sys.argv[1:])
if np.intersect1d(smc, probe).size:
    raise SystemExit("SMC refresh and ordinary-IS probe cohorts overlap")
if np.intersect1d(probe, evaluation).size:
    raise SystemExit("Ordinary-IS probe overlaps the spectroscopic evaluation cohort")
PY

echo "[proposal-refresh] selected_variant=$variant base_components=$BASE_COMPONENTS banks=$BANK_A,$BANK_B"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/refresh_popcosmos_encoder_from_smc.py \
  --config "$CONFIG" \
  --source-config "$SOURCE_CONFIG" \
  --dataset "$DATASET" \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --feature-stats "$SOURCE_STATS" \
  --bank "$BANK_A" \
  --bank "$BANK_B" \
  --out "$REFRESH" \
  --epochs "$REFRESH_EPOCHS" \
  --object-batch-size 32 \
  --learning-rate 2e-5 \
  --seed 260817 \
  --mixture-mean-offset "$MIXTURE_MEAN_OFFSET" \
  --require-gpu

python - "$REFRESH/refresh_summary.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
if payload.get("refresh_gate") != "PASS":
    raise SystemExit("Encoder refresh held-out weighted NLL did not improve")
PY

python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-cosmos \
  --runtime gpu \
  --dataset "$DATASET" \
  --checkpoint "$REFRESH/checkpoints/best.eqx" \
  --feature-stats "$REFRESH/feature_stats.json" \
  --row-indices-file "$ROW_INDICES" \
  --selection-mode sequential \
  --limit "$PROBE_LIMIT" \
  --batch-size 128 \
  --jax-batch-size 64 \
  --posterior-samples "$PROBE_SAMPLES" \
  --prior-samples 256 \
  --decoder-sample-chunk-size 1 \
  --prior-predictive-batch-size 64 \
  --out "$PROPOSAL" \
  --shard-outputs \
  --no-posterior-predictive \
  --no-residual-samples
python -m euclid_dsps.cli --config "$CONFIG" \
  amortized-finalize-inference --out "$PROPOSAL" --limit "$PROBE_LIMIT"
python scripts/importance_correct_posterior.py \
  --posterior "$PROPOSAL" \
  --out "$CORRECTION" \
  --config "$CONFIG" \
  --target-checkpoint "$REFRESH/checkpoints/best.eqx" \
  --resample-count 256 \
  --seed 260817 \
  --min-median-ess-fraction "$MIN_MEDIAN_ESS_FRACTION" \
  --max-fraction-pareto-k-gt-0p7 "$MAX_FRACTION_PARETO_K_GT_0P7"

python - "$variant" "$REFRESH/refresh_summary.json" "$CORRECTION/importance_summary.json" "$OUT" "$MIN_MEDIAN_ESS_FRACTION" "$MAX_FRACTION_PARETO_K_GT_0P7" "$BASELINE_IMPORTANCE" <<'PY'
import json
import sys
from pathlib import Path
variant, refresh_path, importance_path, out, min_ess, max_bad_k, baseline_path = sys.argv[1:]
refresh = json.load(open(refresh_path))
importance = json.load(open(importance_path))
support = importance["support_gate"]
metrics = {
    key: importance[key]
    for key in (
        "median_raw_ess_fraction",
        "median_psis_ess_fraction",
        "fraction_pareto_k_gt_0p7",
        "fraction_pareto_k_gt_1",
    )
}
payload = {
    "status": "complete",
    "selected_likelihood_variant": variant,
    "encoder_refresh_gate": refresh["refresh_gate"],
    "ordinary_importance_support_gate": support["status"],
    "proposal_base_components": refresh["target_base_components"],
    "proposal_probe_cohort": "disjoint frozen validation rows",
    "cohort_role": "proposal-capacity tuning; spectroscopy unused",
    "ordinary_importance_thresholds": {
        "min_median_ess_fraction": float(min_ess),
        "max_fraction_pareto_k_gt_0p7": float(max_bad_k),
    },
    "candidate_metrics": metrics,
    "confirmation_required": (
        refresh["refresh_gate"] == "PASS" and support["status"] == "PASS"
    ),
    "ready_for_empirical_bayes": False,
}
if baseline_path:
    baseline = json.load(open(baseline_path))
    baseline_metrics = {key: baseline[key] for key in metrics}
    payload["unimodal_baseline_metrics"] = baseline_metrics
    payload["candidate_minus_baseline"] = {
        key: metrics[key] - baseline_metrics[key] for key in metrics
    }
Path(out, "refresh_validation_summary.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
Path(out, "DONE").touch()
print(json.dumps(payload, indent=2))
PY
