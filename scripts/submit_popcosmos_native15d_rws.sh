#!/bin/bash
set -Eeuo pipefail

STAGE="${1:-}"
case "$STAGE" in
  n5k) WALLTIME=04:00:00 ;;
  n20k) WALLTIME=08:00:00 ;;
  n40k|full) WALLTIME=15:00:00 ;;
  *)
    echo "Usage: $0 {n5k|n20k|n40k|full}"
    exit 2
    ;;
esac

REPO_DIR="${REPO_DIR:-${PWD}}"
CONFIG="${CONFIG:-configs/experiments/popcosmos_native15d_rws.yaml}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LATEST_ENV="$LOG_DIR/popcosmos_native15d_rws_latest.env"

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"

if [[ "$STAGE" != "n5k" && -z "${ROOT_DIR:-}" && -s "$LATEST_ENV" ]]; then
  source "$LATEST_ENV"
fi

if [[ "$STAGE" == "n5k" ]]; then
  ROOT_DIR="${ROOT_DIR:-outputs/runs/popcosmos_native15d_rws_$(date +%Y%m%d_%H%M%S)}"
else
  ROOT_DIR="${ROOT_DIR:?Set ROOT_DIR or keep $LATEST_ENV from the n5k submission}"
fi

case "$STAGE" in
  n20k) test -e "$ROOT_DIR/n5k/DONE" ;;
  n40k|full) test -e "$ROOT_DIR/n20k/DONE" ;;
esac
test ! -e "$ROOT_DIR/$STAGE" || {
  echo "[cosmos-rws15-submit][error] stage already exists: $ROOT_DIR/$STAGE"
  exit 2
}

test -s Data/cosmos2020/prepared/PREPOST_COMPLETE.json
test -s configs/experiments/popcosmos_native15d_rws.yaml

job_raw=$(sbatch --parsable \
  --time="$WALLTIME" \
  --export=ALL,STAGE="$STAGE",ROOT_DIR="$ROOT_DIR",CONFIG="$CONFIG",REPO_DIR="$REPO_DIR" \
  scripts/popcosmos_native15d_rws_h100.slurm)
job="${job_raw%%;*}"

previous_jobs=""
if [[ "$STAGE" != "n5k" ]]; then
  previous_jobs="${JOB_IDS:-}"
fi
if [[ -n "$previous_jobs" ]]; then
  job_ids="$previous_jobs,$job"
else
  job_ids="$job"
fi

printf 'export ROOT_DIR=%q\nexport CONFIG=%q\nexport LAST_STAGE=%q\nexport LAST_JOB=%q\nexport JOB_IDS=%q\n' \
  "$ROOT_DIR" "$CONFIG" "$STAGE" "$job" "$job_ids" \
  > "$LATEST_ENV"

submission_log="$LOG_DIR/submit_popcosmos_native15d_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
printf '%s=%s\nroot_dir=%s\nobjective=reweighted_wake_sleep\nwake_particles=8\n' \
  "$STAGE" "$job" "$ROOT_DIR" | tee "$submission_log"
echo "monitor: squeue -j $job"
echo "after completion: sacct -X -j $job --format=JobID,State,Elapsed,AllocTRES,ExitCode"
echo "latest_env=$LATEST_ENV"
echo "submission_log=$submission_log"
