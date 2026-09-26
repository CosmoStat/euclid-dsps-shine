#!/bin/bash
# Run locally. Only reports/configs/logs <=25 MiB, never banks or checkpoints.
set -Eeuo pipefail
KIND=${1:-refinement}
case "$KIND" in
  refinement) PREFIX=avi_coherent_refinement; VARIABLE=REFINE ;;
  inference) PREFIX=avi_coherent_inference; VARIABLE=INFERENCE ;;
  *) echo 'Usage: bash scripts/rsync_feniks_coherent_results.sh refinement|inference' >&2; exit 2 ;;
esac
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
REMOTE=${FENIKS_REMOTE:-urx63nr@jean-zay.idris.fr}
JUMP=${FENIKS_JUMP:-mr287471@hubble.extra.cea.fr}
SOURCE=$(ssh -J "$JUMP" "$REMOTE" \
  "source '$BASE/${PREFIX}_latest.env' && printf '%s\n' \"\$$VARIABLE\"" | tail -n 1)
RUN=${SOURCE##*/}
[[ "$RUN" =~ ^${PREFIX}_[0-9]{8}_[0-9]{6}$ && "$SOURCE" == "$BASE/$RUN" ]] || {
  printf 'Refusing unexpected remote source: %q\n' "$SOURCE" >&2; exit 1;
}
LOCAL="${FENIKS_LOCAL_RESULTS:-/home/maxime/src/DSPS/outputs/forward_population_results}/$RUN"
mkdir -p "$LOCAL"
printf 'Source: %s\nLocal: %s\n' "$SOURCE" "$LOCAL"
rsync -avhm --partial --info=progress2 --max-size=25m \
  -e "ssh -J $JUMP" \
  --exclude='banks/' --exclude='cache/' \
  --include='*/' --include='*.png' --include='*.pdf' \
  --include='*.csv' --include='*.json' --include='*.jsonl' \
  --include='*.md' --include='*.yaml' --include='*.env' \
  --include='*.out' --include='*.err' --include='/CODE_SHA256' \
  --exclude='*' "$REMOTE:$SOURCE/" "$LOCAL/"
