#!/bin/bash
#SBATCH --job-name=feniks_xselect
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
ACTION="${ACTION:-select}"
SELECTION="${SELECTION:-$ROOT_DIR/pilot/selected_mclmc.json}"

cd "$REPO_DIR"
mkdir -p outputs/logs "$(dirname "$SELECTION")"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu

mapfile -t GALAXIES < <(
  find "$ROOT_DIR/galaxies" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)
(( ${#GALAXIES[@]} >= 2 )) || {
  echo "[exact-pilot-select][error] expected at least two prepared galaxies"
  exit 2
}

extra=()
if [[ "$ACTION" == "validate" ]]; then
  extra+=(--validate-galaxy-dir "${GALAXIES[1]}")
elif [[ "$ACTION" != "select" ]]; then
  echo "[exact-pilot-select][error] unsupported ACTION=$ACTION"
  exit 2
fi

python scripts/select_feniks_mclmc_pilot.py \
  --root "$ROOT_DIR" --galaxy-dir "${GALAXIES[0]}" --out "$SELECTION" \
  "${extra[@]}"

if [[ "$ACTION" == "select" ]]; then
  test -s "$SELECTION"
else
  python - "$SELECTION" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not payload.get("validation_passes_rhat_1_10", False):
    raise SystemExit("MCLMC pilot validation gate failed")
print("[exact-pilot-select] second-galaxy validation passed")
PY
fi
