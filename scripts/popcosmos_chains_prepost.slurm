#!/bin/bash
#SBATCH --job-name=popcosmos_chains
#SBATCH --account=jrx@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
ASSET_ROOT="${ASSET_ROOT:-${SCRATCH:?Set SCRATCH or ASSET_ROOT}/popcosmos_v1p1_chains}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536}"
PAIRED_SPECZ="${PAIRED_SPECZ:-$RUN_ROOT/publication_checks/checks_20260805_143042/redshift_only_comparison/paired_public_specz_objects.parquet}"
POSTERIOR_OUT="${POSTERIOR_OUT:-$ASSET_ROOT/redshift_posterior_1395}"
ZIP_URL="${ZIP_URL:-https://zenodo.org/records/13820043/files/chains.h5.zip?download=1}"
ZIP_MD5="${ZIP_MD5:-657a17446fc8b13d15e60bd54552b015}"
ZIP_PATH="${ZIP_PATH:-$ASSET_ROOT/chains.h5.zip}"
H5_PATH="${H5_PATH:-$ASSET_ROOT/chains.h5}"
EXPECTED_OBJECTS="${EXPECTED_OBJECTS:-1395}"
SAMPLES="${SAMPLES:-128}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$ASSET_ROOT"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

test -s "$PAIRED_SPECZ" || {
  echo "[popcosmos-chains][error] missing paired cohort: $PAIRED_SPECZ"
  exit 2
}
command -v curl >/dev/null || {
  echo "[popcosmos-chains][error] curl is unavailable"
  exit 2
}

echo "[popcosmos-chains] start: $(date --iso-8601=seconds)"
echo "[popcosmos-chains] asset_root=$ASSET_ROOT"
echo "[popcosmos-chains] paired_specz=$PAIRED_SPECZ"
df -h "$ASSET_ROOT"

if [[ -s "$ZIP_PATH" ]] && echo "$ZIP_MD5  $ZIP_PATH" | md5sum --check --status; then
  echo "[popcosmos-chains] reusing checksum-validated archive: $ZIP_PATH"
else
  echo "[popcosmos-chains] downloading with HTTP resume: $ZIP_URL"
  curl \
    --fail --location --show-error \
    --retry 30 --retry-delay 10 --retry-max-time 7200 --retry-all-errors \
    --connect-timeout 60 --speed-time 300 --speed-limit 1024 \
    --continue-at - --output "$ZIP_PATH" "$ZIP_URL"
  echo "$ZIP_MD5  $ZIP_PATH" | md5sum --check
fi

export ZIP_PATH H5_PATH
python - <<'PY'
import os
import shutil
import zipfile
from pathlib import Path

archive = Path(os.environ["ZIP_PATH"])
target = Path(os.environ["H5_PATH"])
with zipfile.ZipFile(archive) as handle:
    candidates = [item for item in handle.infolist() if Path(item.filename).name == "chains.h5"]
    if len(candidates) != 1:
        raise SystemExit(f"Expected one chains.h5 member, found {[item.filename for item in candidates]}")
    member = candidates[0]
    print(
        f"[popcosmos-chains] archive member={member.filename} "
        f"compressed={member.compress_size} uncompressed={member.file_size}",
        flush=True,
    )
    if target.is_file() and target.stat().st_size == member.file_size:
        print(f"[popcosmos-chains] reusing extracted HDF5: {target}", flush=True)
    else:
        free = shutil.disk_usage(target.parent).free
        required = member.file_size + 10 * 1024**3
        if free < required:
            raise SystemExit(
                f"Insufficient free space for extraction: free={free}, required={required}"
            )
        partial = target.with_suffix(target.suffix + ".part")
        partial.unlink(missing_ok=True)
        print(f"[popcosmos-chains] extracting -> {partial}", flush=True)
        with handle.open(member) as source, partial.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        if partial.stat().st_size != member.file_size:
            raise SystemExit("Extracted HDF5 size differs from ZIP metadata")
        partial.replace(target)
        print(f"[popcosmos-chains] extraction complete -> {target}", flush=True)
PY

python - "$H5_PATH" <<'PY'
import sys
import h5py

with h5py.File(sys.argv[1], "r") as handle:
    print(f"[popcosmos-chains] HDF5 top-level keys={list(handle.keys())}")
PY

if [[ -e "$POSTERIOR_OUT/DONE" ]]; then
  echo "[popcosmos-chains] reusing extracted redshift posterior: $POSTERIOR_OUT"
else
  if [[ -e "$POSTERIOR_OUT" ]]; then
    echo "[popcosmos-chains][error] incomplete output exists: $POSTERIOR_OUT"
    exit 2
  fi
  python scripts/extract_popcosmos_redshift_chains.py \
    --chains "$H5_PATH" \
    --cohort "$PAIRED_SPECZ" \
    --out "$POSTERIOR_OUT" \
    --expected-objects "$EXPECTED_OBJECTS" \
    --samples "$SAMPLES"
fi

for path in \
  "$POSTERIOR_OUT/DONE" \
  "$POSTERIOR_OUT/inference_truth.parquet" \
  "$POSTERIOR_OUT/posterior_samples.parquet" \
  "$POSTERIOR_OUT/posterior_samples_all.parquet" \
  "$POSTERIOR_OUT/extraction_manifest.json"; do
  test -s "$path" || test -e "$path" || {
    echo "[popcosmos-chains][error] missing output: $path"
    exit 3
  }
done

cat > "$ASSET_ROOT/PREPOST_COMPLETE.json" <<EOF
{
  "status": "complete",
  "zenodo_record": "13820043",
  "archive": "$ZIP_PATH",
  "chains": "$H5_PATH",
  "posterior": "$POSTERIOR_OUT",
  "objects": $EXPECTED_OBJECTS,
  "primary_samples_per_object": $SAMPLES
}
EOF
echo "[popcosmos-chains] complete: $(date --iso-8601=seconds)"
