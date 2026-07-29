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
EXPECTED_PUBLIC="${EXPECTED_PUBLIC:-139489}"
EXPECTED_SELECTED="${EXPECTED_SELECTED:-}"
MIN_PUBLIC_RETENTION="${MIN_PUBLIC_RETENTION:-0.99}"
INSTALL="${INSTALL:-1}"
TAP_TIMEOUT="${TAP_TIMEOUT:-14400}"
TAP_JOB_URL="${TAP_JOB_URL:-}"

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
DOWNLOAD_ARGS=(--out "$ASSET_DIR" --timeout "$TAP_TIMEOUT")
if [[ -n "$TAP_JOB_URL" ]]; then
  DOWNLOAD_ARGS+=(--tap-job-url "$TAP_JOB_URL")
fi
if [[ -s "$ASSET_DIR/DOWNLOAD_COMPLETE.json" ]]; then
  echo "[cosmos2020-prepost] reusing completed asset download: $ASSET_DIR"
else
  python scripts/download_cosmos2020_assets.py "${DOWNLOAD_ARGS[@]}"
fi
if ! python - "$ASSET_DIR/download_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

from euclid_dsps.cosmos2020 import COSMOS_BANDS

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
found = {row["band"]: row["svo_id"] for row in manifest["filters"]}
expected = {band.name: band.svo_id for band in COSMOS_BANDS}
if found != expected:
    raise SystemExit(1)
print("[cosmos2020-prepost] filter provenance: current")
PY
then
  echo "[cosmos2020-prepost] refreshing changed filter provenance"
  python scripts/download_cosmos2020_assets.py \
    --out "$ASSET_DIR" --filters-only
fi
if ! python - "$ASSET_DIR/download_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

from euclid_dsps.cosmos2020 import (
    SPECZ_COMPILATION_SHA256,
    SPECZ_COMPILATION_SIZE,
    sha256_file,
)

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
entry = manifest.get("spectroscopy") or {}
path = Path(entry.get("path", ""))
if (
    not path.is_file()
    or path.stat().st_size != SPECZ_COMPILATION_SIZE
    or sha256_file(path) != SPECZ_COMPILATION_SHA256
):
    raise SystemExit(1)
print("[cosmos2020-prepost] spectroscopy provenance: current")
PY
then
  echo "[cosmos2020-prepost] downloading public COSMOS spectroscopy"
  python scripts/download_cosmos2020_assets.py \
    --out "$ASSET_DIR" --specz-only
fi
if [[ "$INSTALL" == "1" ]]; then
  pip_install_with_retry --no-deps -e "$ASSET_DIR/external/pop-cosmos"
  python - <<'PY'
from pop_cosmos.constants import COSMOS_FILTERS
if len(COSMOS_FILTERS) != 26:
    raise SystemExit("Pop-COSMOS install does not expose 26 COSMOS filters")
print("[cosmos2020-prepost] Pop-COSMOS import: OK")
PY
fi
PREPARE_ARGS=(
  --input "$ASSET_DIR/cosmos2020_farmer_v21.fits"
  --out "$PREPARED_DIR"
  --public-summary "$ASSET_DIR/zenodo/summaries.txt"
  --specz-catalog \
    "$ASSET_DIR/specz/specz_compilation_COSMOS_DR1.1_unique.fits"
  --expected-public "$EXPECTED_PUBLIC"
  --min-public-retention "$MIN_PUBLIC_RETENTION"
)
VALIDATE_ARGS=(--data-dir "$PREPARED_DIR" --asset-dir "$ASSET_DIR")
if [[ -n "$EXPECTED_SELECTED" ]]; then
  PREPARE_ARGS+=(--expected-selected "$EXPECTED_SELECTED")
  VALIDATE_ARGS+=(--expected-full "$EXPECTED_SELECTED")
fi
python scripts/prepare_cosmos2020_farmer.py "${PREPARE_ARGS[@]}"
python scripts/validate_cosmos2020_reproduction.py "${VALIDATE_ARGS[@]}"
python - <<PY
import json
from pathlib import Path
manifest = json.loads(
    Path("$PREPARED_DIR/preparation_manifest.json").read_text(encoding="utf-8")
)
path = Path("$PREPARED_DIR/PREPOST_COMPLETE.json")
path.write_text(json.dumps({
    "status": "complete",
    "slurm_job_id": "${SLURM_JOB_ID:-local}",
    "expected_public": int("$EXPECTED_PUBLIC"),
    "selected_rows": int(manifest["selected_rows"]),
}, indent=2) + "\\n")
print(f"[cosmos2020-prepost] completion marker -> {path}")
PY
