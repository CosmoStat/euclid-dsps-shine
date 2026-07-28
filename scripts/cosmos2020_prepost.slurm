#!/bin/bash
#SBATCH --job-name=cosmos2020_dl
#SBATCH --account=jrx@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
ASSET_DIR="${ASSET_DIR:-Data/cosmos2020/assets}"
PREPARED_DIR="${PREPARED_DIR:-Data/cosmos2020/prepared}"
EXPECTED_SELECTED="${EXPECTED_SELECTED:-140938}"
INSTALL="${INSTALL:-1}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"
mkdir -p outputs/logs "$ASSET_DIR" "$PREPARED_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip_install_with_retry() {
  local attempt
  for attempt in 1 2 3; do
    if python -m pip install \
      --no-build-isolation --retries 10 --timeout 120 "$@"; then
      return 0
    fi
    echo "[cosmos2020-prepost] pip attempt $attempt/3 failed" >&2
    sleep $((attempt * 15))
  done
  return 1
}

if [[ "$INSTALL" == "1" ]]; then
  pip_install_with_retry -e '.[amortized,samplers,gpu]'
fi
python scripts/download_cosmos2020_assets.py --out "$ASSET_DIR"
if [[ "$INSTALL" == "1" ]]; then
  pip_install_with_retry --no-deps -e "$ASSET_DIR/external/pop-cosmos"
  python - <<'PY'
from pop_cosmos.constants import COSMOS_FILTERS
if len(COSMOS_FILTERS) != 26:
    raise SystemExit("Pop-COSMOS install does not expose 26 COSMOS filters")
print("[cosmos2020-prepost] Pop-COSMOS import: OK")
PY
fi
python scripts/prepare_cosmos2020_farmer.py \
  --input "$ASSET_DIR/cosmos2020_farmer_v21.fits" \
  --out "$PREPARED_DIR" \
  --expected-selected "$EXPECTED_SELECTED"
python scripts/validate_cosmos2020_reproduction.py \
  --data-dir "$PREPARED_DIR" --asset-dir "$ASSET_DIR" \
  --expected-full "$EXPECTED_SELECTED"
python - <<PY
import json
from pathlib import Path
path = Path("$PREPARED_DIR/PREPOST_COMPLETE.json")
path.write_text(json.dumps({
    "status": "complete",
    "slurm_job_id": "${SLURM_JOB_ID:-local}",
    "expected_selected": int("$EXPECTED_SELECTED"),
}, indent=2) + "\\n")
print(f"[cosmos2020-prepost] completion marker -> {path}")
PY
