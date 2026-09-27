#!/bin/bash
# SOURCE_PARENT [NIGHT_ROOT], or --resume NIGHT_ROOT. Never duplicate active work.
set -Eeuo pipefail
REPO=$(pwd -P)
command -v sbatch >/dev/null
if [[ ${1:-} == --resume ]]; then
  NIGHT=$(realpath "${2:?night root}")
  source "$NIGHT/NIGHT.env"
else
  SOURCE=$(realpath "${1:?completed blind parent root}")
  BASE=$(dirname "$SOURCE")
  exec 7>"$BASE/.feniks_overnight.lock"
  flock -n 7 || { echo 'Another overnight launch is active' >&2; exit 1; }
  TAG=$(date +%Y%m%d_%H%M%S)
  NIGHT=$(realpath -m "${2:-$BASE/avi_overnight_$TAG}")
  test ! -e "$NIGHT"
  POSTERIOR_RUN="$BASE/avi_parent_to_posterior_$TAG"
  CONDITIONAL="$BASE/avi_conditional_parent_$TAG"
  # Reuse a previous posterior branch only when its source parent matches.
  if [[ -f "$BASE/avi_parent_to_posterior_latest.env" ]]; then
    PREVIOUS=$(bash -c 'source "$1"; printf "%s" "${POSTERIOR_RUN:?}"' _ "$BASE/avi_parent_to_posterior_latest.env")
    MATCH=$(python - "$PREVIOUS" "$SOURCE" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])/'MANIFEST.json'
print(int(p.is_file() and Path(json.loads(p.read_text())['parent_run']).resolve() == Path(sys.argv[2]).resolve()))
PY
)
    [[ $MATCH != 1 ]] || POSTERIOR_RUN=$PREVIOUS
  fi
  if [[ -f "$BASE/avi_conditional_parent_latest.env" ]]; then
    PREVIOUS=$(bash -c 'source "$1"; printf "%s" "${CONDITIONAL:?}"' _ "$BASE/avi_conditional_parent_latest.env")
    MATCH=$(python - "$PREVIOUS" "$SOURCE" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])/'MANIFEST.json'
print(int(p.is_file() and Path(json.loads(p.read_text())['baseline_parent_run']).resolve() == Path(sys.argv[2]).resolve()))
PY
)
    [[ $MATCH != 1 ]] || CONDITIONAL=$PREVIOUS
  fi
  mkdir -p "$NIGHT"
  printf 'export SOURCE=%q\nexport POSTERIOR_RUN=%q\nexport CONDITIONAL=%q\n' \
    "$SOURCE" "$POSTERIOR_RUN" "$CONDITIONAL" > "$NIGHT/NIGHT.env"
fi
exec 8>"$NIGHT/.submit.lock"
flock -n 8 || { echo 'Another overnight submission is active' >&2; exit 1; }
BASE=$(dirname "$SOURCE")
printf 'export NIGHT=%q\n' "$NIGHT" > "$BASE/avi_overnight_latest.env"
active() {
  local root=$1 live ids
  [[ -f "$root/JOBS.env" ]] || return 1
  ids=$(bash -c 'source "$1"; printf "%s" "${ALL_JOBS:?}"' _ "$root/JOBS.env") || return 2
  # A failed queue query is fatal, not evidence that a run is inactive.
  live=$(squeue -h -u "${USER:?}" -o '%i') || return 2
  printf '%s\n' "$live" | awk -v ids="$ids" 'BEGIN {n=split(ids,a,","); for(i=1;i<=n;i++) wanted[a[i]]=1} {base=$1; sub(/_.*/,"",base); if(base in wanted) found=1} END {exit !found}'
}
for branch in conditional posterior; do
  if [[ $branch == conditional ]]; then
    ROOT=$CONDITIONAL
    SCRIPT=scripts/submit_feniks_conditional_parent.sh
    ARGS=("$SOURCE" "$ROOT")
  else
    ROOT=$POSTERIOR_RUN
    SCRIPT=scripts/submit_feniks_parent_to_posterior.sh
    ARGS=(--exploratory "$SOURCE" "$ROOT")
  fi
  if active "$ROOT"; then
    echo "$branch: existing active jobs retained ($ROOT)"
    continue
  else
    status=$?
    [[ $status == 1 ]] || { echo 'SLURM query failed; no duplicate submission' >&2; exit "$status"; }
  fi
  if [[ -d "$ROOT" ]]; then ARGS=(--resume "$ROOT"); fi
  bash "$REPO/$SCRIPT" "${ARGS[@]}" | tee -a "$NIGHT/$branch-submission.txt"
done
echo 'Two independent branches. Default peak <=5 H100s; total ceiling <=14 H100-hours, not expected runtime.'
printf 'watch=bash scripts/watch_feniks_overnight.sh %q\n' "$NIGHT"
