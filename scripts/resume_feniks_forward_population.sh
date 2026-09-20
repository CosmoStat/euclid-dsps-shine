#!/bin/bash
# Recover a failed population fit, retaining the completed reference/classifier.
# Usage: bash scripts/resume_feniks_forward_population.sh EXISTING_ROOT
set -Eeuo pipefail
export FORWARD_ROOT=$(realpath "${1:?existing forward campaign root}")
REPO=$(pwd -P)
test -s "$FORWARD_ROOT/JOBS.env"
test -d "$REPO/filters"
exec 9>"$FORWARD_ROOT/.recovery.lock"
flock -n 9 || { echo 'Another recovery submission is active' >&2; exit 2; }
source "$FORWARD_ROOT/JOBS.env"
if [[ -f "$FORWARD_ROOT/RECOVERY_PENDING" ]]; then
  echo 'An incomplete recovery submission exists; inspect RECOVERY_PENDING before retrying' >&2
  exit 2
fi
CHAIN="$POPULATION_JOB,$POSTERIOR_BANK_JOB,$TRAIN_JOB,$REPORT_JOB"
STATES=$(squeue -h -j "$CHAIN" -o '%i %T')
declare -A CANCEL=()
while read -r id state; do
  [[ -n "$id" ]] || continue
  id=${id%%_*}
  [[ "$id" =~ ^[0-9]+$ && "$state" == PENDING ]] || {
    echo "Refusing recovery while old job $id is $state" >&2; exit 2;
  }
  CANCEL["$id"]=1
done <<< "$STATES"

# Read-only input, bank and optimizer verification, before cancelling anything.
mapfile -t SETTINGS < <(JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu EUCLID_DSPS_REQUIRE_GPU=0 JAX_ENABLE_X64=true python - "$FORWARD_ROOT" <<'PY'
import json, sys
from pathlib import Path
from scripts.feniks_forward_population import contract
from scripts.feniks_avi_experiments import read, sha
root=Path(sys.argv[1]); manifest,cfg=contract(root)
if (root/'population/FINAL.json').exists():
    raise SystemExit('Population already finalized; do not refit its frozen parent')
if list((root/'posterior_bank').glob('shard_*/FINAL.json')) or (root/'posterior/FINAL.json').exists():
    raise SystemExit('Downstream banks/posterior already exist; refuse to change their parent')
assert read(root/'preflight/FINAL.json')['status']=='FORWARD_PREFLIGHT_PASS'
for task in range(cfg['reference_shards']):
    folder=root/'reference'/f'shard_{task:03d}'
    receipt=read(folder/'FINAL.json')
    assert receipt['status']=='FORWARD_BANK_COMPLETE'
    assert receipt['bank_sha256']==sha(folder/'bank.npz')
resume=read(root/'population/RESUME.json')
assert resume['epoch']==cfg['classifier']['epochs'], 'Classifier training is not complete'
assert sha(root/'population'/resume['state_file'])==resume['state_sha256']
assert (root/'population/best.eqx').is_file()
r=cfg['resources']
assert r['gpus_per_task']==1
for value in (cfg['posterior_shards'],r['posterior_bank_concurrency'],
              r['bank_hours'],r['classifier_hours'],r['posterior_hours'],r['report_hours']):
    assert isinstance(value,int) and value>0
    print(value)
PY
)
[[ ${#SETTINGS[@]} == 6 ]] || { echo 'Recovery validation failed' >&2; exit 2; }
read -r POST PC BH CH QH RH <<< "${SETTINGS[*]}"

TAG=$(date +%Y%m%d_%H%M%S)
RECOVERY="$FORWARD_ROOT/recovery/$TAG"
mkdir -p "$RECOVERY"
cp "$FORWARD_ROOT/JOBS.env" "$RECOVERY/JOBS.previous.env"
cp "$FORWARD_ROOT/CODE_DIR" "$RECOVERY/CODE_DIR.previous"
cp "$FORWARD_ROOT/CODE_SHA256" "$RECOVERY/CODE_SHA256.previous"
sha256sum "$FORWARD_ROOT/population/best.eqx" > "$RECOVERY/classifier.sha256"
ARCHIVE="$RECOVERY/code.tar"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export FORWARD_CODE="${SCRATCH:?}/feniks_sc_drws_runtime/code/forward-$DIGEST"
mkdir -p "$FORWARD_CODE"
tar -xf "$ARCHIVE" -C "$FORWARD_CODE"
if [[ ! -e "$FORWARD_CODE/Data" && ! -L "$FORWARD_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$FORWARD_CODE/Data"
fi
# Only the old pending successors of THIS campaign are cancelled.
printf '%s\n' "$RECOVERY" > "$FORWARD_ROOT/RECOVERY_PENDING"
if [[ ${#CANCEL[@]} -gt 0 ]]; then
  scancel "${!CANCEL[@]}"
fi
printf '%s\n' "$FORWARD_CODE" > "$FORWARD_ROOT/CODE_DIR"
printf '%s\n' "$DIGEST" > "$FORWARD_ROOT/CODE_SHA256"
printf 'export PREFLIGHT_JOB=%q\nexport REFERENCE_JOB=%q\n' \
  "$PREFLIGHT_JOB" "$REFERENCE_JOB" > "$RECOVERY/JOBS.env"
ALL=("$PREFLIGHT_JOB" "$REFERENCE_JOB")
submit() {
  local mode=$1 hours=$2 dependency=$3 array=${4:-}
  export FORWARD_MODE=$mode
  local opts=() raw
  [[ -z "$dependency" ]] || opts+=(--dependency="afterok:$dependency")
  [[ -z "$array" ]] || opts+=(--array="$array")
  raw=$(sbatch --parsable --time="$hours:00:00" --export=ALL "${opts[@]}" \
    --output="$FORWARD_ROOT/logs/$mode-%A_%a.out" \
    --error="$FORWARD_ROOT/logs/$mode-%A_%a.err" "$FORWARD_CODE/scripts/feniks_forward_population.slurm")
  JOB=${raw%%;*}; ALL+=("$JOB")
  printf 'export %s_JOB=%q\n' "${mode^^}" "$JOB" | tr '-' '_' >> "$RECOVERY/JOBS.env"
  printf 'export ALL_JOBS=%q\n' "$(IFS=,; echo "${ALL[*]}")" >> "$RECOVERY/JOBS.env"
  cp "$RECOVERY/JOBS.env" "$FORWARD_ROOT/JOBS.next.env"
  mv "$FORWARD_ROOT/JOBS.next.env" "$FORWARD_ROOT/JOBS.env"
  printf '%s=%s\n' "$mode" "$JOB"
}
submit population "$CH" ''
submit posterior-bank "$BH" "$JOB" "0-$POST%$PC"
submit train "$QH" "$JOB"
submit report "$RH" "$JOB"
rm "$FORWARD_ROOT/RECOVERY_PENDING"
printf 'Reused reference banks and completed classifier. KKT tolerance remains 2e-6.\n'
printf 'watch=bash scripts/watch_feniks_forward_population.sh %q\n' "$FORWARD_ROOT"
