#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PREPARED_ROOT="${PREPARED_ROOT:-outputs/runs/feniks_exact_posterior_two_galaxy_nuts_20260728_211755}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/feniks_exact_posterior_two_galaxy_nuts_big_${STAMP}}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
NUTS_WARMUP="${NUTS_WARMUP:-200}"
NUTS_MAX_DOUBLINGS="${NUTS_MAX_DOUBLINGS:-4}"
SAMPLE_CHUNKS="${SAMPLE_CHUNKS:-100:100:100:100:100:100:100:100:100:100}"
NUTS_TIME="${NUTS_TIME:-20:00:00}"

cd "$REPO_DIR"
mkdir -p outputs/logs
for path in "$CONFIG" "$DATASET" "$CHECKPOINT" "${CHECKPOINT}.json" \
  "$FEATURE_STATS" "$PREPARED_ROOT/cohort.csv" \
  "$PREPARED_ROOT/cohort.parquet" "$PREPARED_ROOT/contract.json"; do
  test -s "$path" || {
    echo "[two-galaxy-nuts-big][error] missing $path"
    exit 2
  }
done

probe_summary=$(python - "$PREPARED_ROOT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
row = pd.read_parquet(root / "cohort.parquet").iloc[0]
galaxy = (
    root
    / "galaxies"
    / f"{int(row['order']):02d}_{row['example_key']}_row{int(row['row_index'])}"
)
print(galaxy / "nuts" / "batched_probe_summary.json")
PY
)
test -s "$probe_summary" || {
  echo "[two-galaxy-nuts-big][error] missing batched probe: $probe_summary"
  echo "Run submit_feniks_exact_posterior_nuts_batched_probe.sh and its summarizer first."
  exit 2
}
python - "$probe_summary" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))
speedup = float(summary.get("throughput_speedup", float("nan")))
divergences = summary.get("batched_divergences")
if summary.get("status") != "passed":
    raise SystemExit(f"Batched NUTS probe did not pass: {path}")
if divergences != 0:
    raise SystemExit(f"Batched NUTS probe has {divergences} divergences: {path}")
if not math.isfinite(speedup) or speedup <= 1.0:
    raise SystemExit(f"Batched NUTS is not faster than scalar execution: {speedup}")
print(
    "[two-galaxy-nuts-big] batched probe passed "
    f"speedup={speedup:.2f} divergences={divergences}"
)
PY

if [[ ! -e "$ROOT_DIR" ]]; then
  python - "$PREPARED_ROOT" "$ROOT_DIR" "$CONFIG" "$DATASET" \
    "$CHECKPOINT" "$FEATURE_STATS" "$NUTS_WARMUP" \
    "$NUTS_MAX_DOUBLINGS" "$SAMPLE_CHUNKS" "$NUTS_TIME" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
config = Path(sys.argv[3])
dataset = Path(sys.argv[4])
checkpoint = Path(sys.argv[5])
feature_stats = Path(sys.argv[6])
warmup = int(sys.argv[7])
max_doublings = int(sys.argv[8])
chunks = [int(value) for value in sys.argv[9].replace(":", ",").split(",")]
nuts_time = sys.argv[10]
cohort = pd.read_parquet(source / "cohort.parquet")
if cohort["row_index"].astype(int).tolist() != [1358, 400]:
    raise SystemExit("Prepared cohort is not the planned [1358, 400] pair")
if cohort["example_key"].astype(str).tolist() != ["typical", "nearby"]:
    raise SystemExit("Prepared cohort labels are not [typical, nearby]")

root_artifacts = ("cohort.csv", "cohort.parquet", "COHORT_DONE")
galaxy_artifacts = (
    "PREP_DONE",
    "observation.parquet",
    "encoder_samples.parquet",
    "importance_weighted_samples.parquet",
    "importance_resampled_samples.parquet",
    "importance_diagnostics.json",
    "map_solutions.parquet",
    "map_trace.parquet",
    "initial_positions.npy",
    "truth.parquet",
    "target_audit.parquet",
    "prepare_manifest.json",
)
for relative in root_artifacts:
    if not (source / relative).exists():
        raise SystemExit(f"Missing preparation artifact: {source / relative}")
for row in cohort.itertuples(index=False):
    name = f"{int(row.order):02d}_{row.example_key}_row{int(row.row_index)}"
    galaxy = source / "galaxies" / name
    for relative in galaxy_artifacts:
        if not (galaxy / relative).exists():
            raise SystemExit(f"Missing preparation artifact: {galaxy / relative}")
    manifest = json.loads(
        (galaxy / "prepare_manifest.json").read_text(encoding="utf-8")
    )
    if int(manifest["row_index"]) != int(row.row_index):
        raise SystemExit(f"row_index mismatch in {galaxy}")
    if str(manifest["object_id"]) != str(row.object_id):
        raise SystemExit(f"object_id mismatch in {galaxy}")
    observed_ids = (
        pd.read_parquet(galaxy / "observation.parquet")["object_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    if observed_ids != [str(row.object_id)]:
        raise SystemExit(f"Observation object_id mismatch in {galaxy}")

target.mkdir(parents=True)
inventory = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link(relative: Path) -> None:
    src = source / relative
    dst = target / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)
    inventory.append(
        {
            "path": str(relative),
            "source": str(src),
            "bytes": src.stat().st_size,
            "sha256": sha256(src),
        }
    )


for relative in root_artifacts:
    link(Path(relative))
for row in cohort.itertuples(index=False):
    name = f"{int(row.order):02d}_{row.example_key}_row{int(row.row_index)}"
    for relative in galaxy_artifacts:
        link(Path("galaxies") / name / relative)

source_contract_path = source / "contract.json"
source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
derived_contract = dict(source_contract)
derived_contract.update(
    {
        "status": "prepared",
        "mode": "pilot",
        "derived_from_prepared_root": str(source),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
)
(target / "contract.json").write_text(
    json.dumps(derived_contract, indent=2, sort_keys=True, allow_nan=False),
    encoding="utf-8",
)
objects = [
    {
        "order": int(row.order),
        "example_key": str(row.example_key),
        "row_index": int(row.row_index),
        "object_id": str(row.object_id),
    }
    for row in cohort.itertuples(index=False)
]
expected_per_galaxy = [
    "nuts/samples.parquet",
    "nuts/diagnostics.parquet",
    "nuts/diagnostics.json",
    "corner_key5.png",
    "corner_key5.pdf",
    "corner_full15.png",
    "corner_full15.pdf",
    "photometric_predictions.parquet",
    "sed_draws.npz",
    "sed_photometry_comparison.png",
    "sed_photometry_comparison.pdf",
    "nuts_trace.png",
    "nuts_convergence.png",
    "DONE",
]
big_contract = {
    "status": "prepared",
    "method": "four-chain vmapped NUTS",
    "posterior_target": (
        "DSPS Student-t2 photometric likelihood plus the jointly learned "
        "RWS RealNVP prior density in normalized latent x-space"
    ),
    "prepared_root": str(source),
    "output_root": str(target),
    "config": str(config),
    "dataset": str(dataset),
    "checkpoint": str(checkpoint),
    "feature_stats": str(feature_stats),
    "checkpoint_sha256": sha256(checkpoint),
    "feature_stats_sha256": sha256(feature_stats),
    "source_contract_sha256": sha256(source_contract_path),
    "code_commit": derived_contract["code_commit"],
    "objects": objects,
    "nuts": {
        "chains_per_galaxy": 4,
        "execution": "vmap_batched_chains",
        "warmup_steps": warmup,
        "target_accept": 0.65,
        "max_num_doublings": max_doublings,
        "sample_chunks": chunks,
        "draws_per_chain": sum(chunks),
        "draws_per_galaxy": 4 * sum(chunks),
        "time_limit_per_galaxy": nuts_time,
        "resume_granularity": "completed sample chunk",
    },
    "linked_preparation_artifacts": inventory,
    "expected_per_galaxy_artifacts": expected_per_galaxy,
    "expected_run_artifacts": [
        "scoreboard.csv",
        "scoreboard.parquet",
        "posterior_agreement.csv",
        "posterior_agreement.parquet",
        "photometric_fit_metrics.csv",
        "photometric_fit_metrics.parquet",
        "posterior_method_agreement.png",
        "posterior_method_agreement.pdf",
        "photometric_fit_comparison.png",
        "photometric_fit_comparison.pdf",
        "benchmark_summary.json",
        "DONE",
    ],
}
(target / "big_run_contract.json").write_text(
    json.dumps(big_contract, indent=2, sort_keys=True, allow_nan=False),
    encoding="utf-8",
)
print(
    f"[two-galaxy-nuts-big] materialized {target} with "
    f"{len(inventory)} hard-linked preparation artifacts"
)
PY
else
  python - "$ROOT_DIR" "$PREPARED_ROOT" "$NUTS_WARMUP" \
    "$NUTS_MAX_DOUBLINGS" "$SAMPLE_CHUNKS" "$NUTS_TIME" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
prepared = Path(sys.argv[2]).resolve()
contract = json.loads((root / "big_run_contract.json").read_text(encoding="utf-8"))
expected = {
    "prepared_root": str(prepared),
    "warmup_steps": int(sys.argv[3]),
    "max_num_doublings": int(sys.argv[4]),
    "sample_chunks": [
        int(value) for value in sys.argv[5].replace(":", ",").split(",")
    ],
    "time_limit_per_galaxy": sys.argv[6],
}
actual = {
    "prepared_root": contract.get("prepared_root"),
    "warmup_steps": contract.get("nuts", {}).get("warmup_steps"),
    "max_num_doublings": contract.get("nuts", {}).get("max_num_doublings"),
    "sample_chunks": contract.get("nuts", {}).get("sample_chunks"),
    "time_limit_per_galaxy": contract.get("nuts", {}).get(
        "time_limit_per_galaxy"
    ),
}
if actual != expected:
    raise SystemExit(
        f"Incompatible big-run recovery contract: actual={actual} expected={expected}"
    )
print(f"[two-galaxy-nuts-big] resuming existing root {root}")
PY
fi

test ! -e "$ROOT_DIR/DONE" || {
  echo "[two-galaxy-nuts-big] run already complete: $ROOT_DIR"
  exit 0
}

missing_galaxies=()
for galaxy_index in 0 1; do
  galaxy_dir=$(python - "$ROOT_DIR" "$galaxy_index" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
row = pd.read_parquet(root / "cohort.parquet").iloc[int(sys.argv[2])]
print(
    root
    / "galaxies"
    / f"{int(row['order']):02d}_{row['example_key']}_row{int(row['row_index'])}"
)
PY
)
  test -f "$galaxy_dir/PREP_DONE" || {
    echo "[two-galaxy-nuts-big][error] incomplete preparation: $galaxy_dir"
    exit 2
  }
  complete=0
  for chain_index in 0 1 2 3; do
    chain_dir=$(printf '%s/nuts/chain_%02d' "$galaxy_dir" "$chain_index")
    if [[ -f "$chain_dir/DONE" ]]; then
      python - "$chain_dir/chain_manifest.json" "$NUTS_WARMUP" \
        "$NUTS_MAX_DOUBLINGS" "$SAMPLE_CHUNKS" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "execution": "vmap_batched_chains",
    "warmup_steps": int(sys.argv[2]),
    "max_num_doublings": int(sys.argv[3]),
    "sample_chunks": [
        int(value) for value in sys.argv[4].replace(":", ",").split(",")
    ],
}
actual = {key: manifest.get(key) for key in expected}
if actual != expected:
    raise SystemExit(
        f"Incompatible completed NUTS chain {sys.argv[1]}: "
        f"actual={actual} expected={expected}"
    )
PY
      complete=$(( complete + 1 ))
    fi
  done
  if (( complete < 4 )); then
    missing_galaxies+=("$galaxy_index")
  fi
done

common_export="ALL,ROOT_DIR=$ROOT_DIR,CONFIG=$CONFIG,DATASET=$DATASET,MODEL_ROOT=$MODEL_ROOT,CHECKPOINT=$CHECKPOINT,FEATURE_STATS=$FEATURE_STATS"
dependency=()
nuts=""
if (( ${#missing_galaxies[@]} > 0 )); then
  array_spec=$(IFS=,; echo "${missing_galaxies[*]}")
  concurrency=${#missing_galaxies[@]}
  nuts=$(sbatch --parsable --array="${array_spec}%${concurrency}" \
    --time="$NUTS_TIME" \
    --export="$common_export,MODE=pilot,NUTS_WARMUP=$NUTS_WARMUP,NUTS_MAX_DOUBLINGS=$NUTS_MAX_DOUBLINGS,SAMPLE_CHUNKS=$SAMPLE_CHUNKS" \
    scripts/feniks_exact_nuts_batched_h100.slurm)
  nuts="${nuts%%;*}"
  dependency=(--dependency="afterok:$nuts")
fi
finalize=$(sbatch --parsable --array=0-1%2 --time=02:00:00 \
  "${dependency[@]}" \
  --export="$common_export,MODE=pilot,FINAL_SAMPLERS=nuts" \
  scripts/feniks_exact_finalize_h100.slurm)
finalize="${finalize%%;*}"
aggregate=$(sbatch --parsable --time=00:20:00 \
  --dependency="afterok:$finalize" \
  --export="$common_export,FINAL_SAMPLERS=nuts" \
  scripts/feniks_exact_aggregate_h100.slurm)
aggregate="${aggregate%%;*}"

requested_hours=$(python - "${#missing_galaxies[@]}" "$NUTS_TIME" <<'PY'
import sys

parts = [int(value) for value in sys.argv[2].split(":")]
if len(parts) == 3:
    hours = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
elif len(parts) == 2:
    hours = parts[0] / 60.0 + parts[1] / 3600.0
else:
    raise SystemExit(f"Unsupported Slurm time format: {sys.argv[2]}")
print(f"{int(sys.argv[1]) * hours + 4.0 + 1.0 / 3.0:.2f}")
PY
)
log="outputs/logs/submit_feniks_exact_two_galaxy_nuts_big_${STAMP}.log"
{
  printf 'prepared_root=%q\nroot=%q\n' "$PREPARED_ROOT" "$ROOT_DIR"
  printf 'nuts_warmup=%q nuts_max_doublings=%q sample_chunks=%q nuts_time=%q\n' \
    "$NUTS_WARMUP" "$NUTS_MAX_DOUBLINGS" "$SAMPLE_CHUNKS" "$NUTS_TIME"
  printf 'nuts=%q finalize=%q aggregate=%q\n' \
    "$nuts" "$finalize" "$aggregate"
} | tee "$log"
echo "monitor: squeue -j $nuts,$finalize,$aggregate"
echo "missing_galaxies=${#missing_galaxies[@]} peak_h100=2"
echo "requested_upper_bound_h100_hours=$requested_hours"
echo "submission_log=$log"
