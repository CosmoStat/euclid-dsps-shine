#!/bin/bash
set -uo pipefail
NIGHT=${1:?night root}
source "$NIGHT/NIGHT.env"
SCRIPTS=$(cd "$(dirname "$0")" && pwd -P)
while true; do
  printf '\n===== PARENT: BLIND SFH-CONDITIONAL REFINEMENT =====\n'
  if [[ -f "$CONDITIONAL/MANIFEST.json" ]]; then
    bash "$SCRIPTS/watch_feniks_conditional_parent.sh" "$CONDITIONAL" --once
  else
    echo 'Not submitted'
  fi
  printf '\n===== POSTERIOR: FROZEN BASELINE PARENT =====\n'
  if [[ -f "$POSTERIOR_RUN/MANIFEST.json" ]]; then
    bash "$SCRIPTS/watch_feniks_parent_to_posterior.sh" "$POSTERIOR_RUN" --once
  else
    echo 'Not submitted'
  fi
  [[ ${2:-} != --once ]] || break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
