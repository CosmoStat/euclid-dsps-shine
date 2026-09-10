#!/bin/bash
set -Eeuo pipefail
export MANIFEST_ROOT="$(realpath "${1:?existing manifest directory}")"
export GUARDED_ROOT="${2:?new output directory}"
export CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?}/feniks_sc_drws_runtime}"
export CATALOG_DIR="$(realpath Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized)"
REPO="$(pwd -P)"
git diff --quiet
git diff --cached --quiet
for FILE in manifest.json full_train_indices.npy confirmation_indices.npy; do
  test -s "$MANIFEST_ROOT/$FILE"
done
test ! -e "$GUARDED_ROOT"
COMMIT="$(git rev-parse HEAD)"
export REPO_DIR="$CACHE_ROOT/code/global-guarded-$COMMIT"
mkdir -p "$CACHE_ROOT/code" "$GUARDED_ROOT/logs"
python - "$MANIFEST_ROOT" "$GUARDED_ROOT" <<'PY'
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
source, root = map(Path, sys.argv[1:])
manifest = json.loads((source / 'manifest.json').read_text())
manifest['manifests'] = {}
for name in ('full_train_indices.npy', 'confirmation_indices.npy'):
    rows = np.load(source / name, allow_pickle=False)
    if not len(rows):
        raise SystemExit(f'empty cohort: {name}')
    target = root / ('smoke_' + name)
    np.save(target, rows[:64], allow_pickle=False)
    label = 'validation' if name.startswith('confirmation') else 'smoke_train'
    manifest['manifests'][label] = dict(path=str(target), count=len(rows[:64]),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest())
(root / 'smoke_manifest.json').write_text(json.dumps(manifest, indent=2))
PY
if [[ ! -e "$REPO_DIR" ]]; then
  git worktree add --detach "$REPO_DIR" "$COMMIT"
  mkdir -p "$REPO_DIR/Data"
  ln -s "$REPO/Data/diffsky" "$REPO_DIR/Data/diffsky"
fi
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$COMMIT"
printf '%s\n' "$COMMIT" > "$GUARDED_ROOT/CODE_COMMIT"
export GUARDED_SMOKE=1
SMOKE=$(sbatch --parsable --time=01:00:00 --export=ALL \
  --output="$GUARDED_ROOT/logs/smoke-%j.out" \
  --error="$GUARDED_ROOT/logs/smoke-%j.err" \
  "$REPO_DIR/scripts/feniks_global_guarded.slurm")
export GUARDED_SMOKE=0
FULL=$(sbatch --parsable --array=0-1%1 --dependency="afterok:${SMOKE%%;*}" \
  --export=ALL --output="$GUARDED_ROOT/logs/train-%A_%a.out" \
  --error="$GUARDED_ROOT/logs/train-%A_%a.err" \
  "$REPO_DIR/scripts/feniks_global_guarded.slurm")
printf 'smoke=%s\ntraining_array=%s\nroot=%s\n' "$SMOKE" "$FULL" "$GUARDED_ROOT"
