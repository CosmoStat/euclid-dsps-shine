#!/bin/bash
#SBATCH --job-name=feniks_sc_e160_wait
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
RECOVERY_ROOT="${RECOVERY_ROOT:?RECOVERY_ROOT is required}"
EVAL_ROOT="${EVAL_ROOT:?EVAL_ROOT is required}"
SELECTED_CANDIDATE="${SELECTED_CANDIDATE:-current_residual_6x256}"
SEED="${SEED:-260826}"
EPOCH="${EPOCH:-160}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TRAIN="$RECOVERY_ROOT/full/$SELECTED_CANDIDATE/seed_$SEED/train"
CHECKPOINT="$TRAIN/checkpoints/epoch_$(printf '%04d' "$EPOCH")"
RECEIPT="$EVAL_ROOT/CHECKPOINT_FROZEN.json"

cd "$REPO_DIR"
mkdir -p "$EVAL_ROOT"

while true; do
  if [[ -s "$CHECKPOINT/raw_model.eqx" \
        && -s "$CHECKPOINT/raw_model.eqx.json" \
        && -s "$CHECKPOINT/ema_model.eqx" \
        && -s "$CHECKPOINT/ema_model.eqx.json" \
        && -s "$CHECKPOINT/prior.eqx" \
        && -s "$CHECKPOINT/prior.eqx.json" \
        && -s "$TRAIN/feature_stats.json" ]]; then
    break
  fi
  echo "[epoch160-wait] waiting for $CHECKPOINT"
  sleep "$POLL_SECONDS"
done

python - "$CHECKPOINT" "$TRAIN/feature_stats.json" "$RECEIPT" "$EPOCH" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1])
feature_stats = Path(sys.argv[2])
destination = Path(sys.argv[3])
epoch = int(sys.argv[4])
records = {}
shared = None
for component in ("raw_model", "ema_model", "prior"):
    path = root / f"{component}.eqx"
    sidecar_path = path.with_suffix(".eqx.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if int(sidecar.get("epoch", -1)) != epoch:
        raise SystemExit(f"{component} sidecar is not epoch {epoch}")
    if sidecar.get("truth_used") is not False:
        raise SystemExit(f"{component} sidecar does not prove truth-free training")
    actual = sha256(path)
    if sidecar.get("sha256") != actual:
        raise SystemExit(f"{component} checkpoint hash mismatch")
    identity = (
        sidecar.get("config_hash"),
        sidecar.get("latent_transform_hash"),
        sidecar.get("feature_stats_hash"),
    )
    if shared is None:
        shared = identity
    elif identity != shared:
        raise SystemExit("epoch-160 component provenance differs")
    records[component] = {
        "path": str(path.resolve()),
        "sidecar": str(sidecar_path.resolve()),
        "sha256": actual,
    }

payload = {
    "status": "FROZEN",
    "epoch": epoch,
    "truth_used_for_training_or_checkpoint_selection": False,
    "checkpoint_directory": str(root.resolve()),
    "feature_stats": {
        "path": str(feature_stats.resolve()),
        "sha256": sha256(feature_stats),
    },
    "components": records,
    "config_hash": shared[0],
    "latent_transform_hash": shared[1],
    "feature_stats_hash": shared[2],
}
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
print(json.dumps(payload, indent=2), flush=True)
PY

echo "[epoch160-wait] frozen -> $RECEIPT"
