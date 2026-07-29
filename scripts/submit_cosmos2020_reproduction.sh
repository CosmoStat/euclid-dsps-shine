#!/bin/bash
set -Eeuo pipefail

THROUGH="${1:-smoke}"
ROOT_DIR="${ROOT_DIR:-outputs/runs/popcosmos_a24_rws_v1}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
STAGES=(smoke n5k n20k n40k full)
TIMES=(00:30:00 01:00:00 04:00:00 08:00:00 20:00:00)
GRES=(gpu:1 gpu:1 gpu:1 gpu:4 gpu:4)

mkdir -p "$LOG_DIR"
test -s Data/cosmos2020/prepared/PREPOST_COMPLETE.json || {
  echo "[cosmos-submit][error] run and verify the prepost job first"; exit 2;
}

dependency=()
jobs=()
found=0
start_index=0
if [[ -e "$ROOT_DIR" ]]; then
  for index in "${!STAGES[@]}"; do
    if [[ "${STAGES[$index]}" == "$THROUGH" ]]; then start_index="$index"; found=1; fi
  done
  if [[ "$found" != "1" ]]; then
    echo "[cosmos-submit][error] argument must be smoke,n5k,n20k,n40k,full"; exit 2
  fi
  if (( start_index == 0 )); then
    echo "[cosmos-submit][error] smoke root already exists: $ROOT_DIR"; exit 2
  fi
  previous="${STAGES[$((start_index - 1))]}"
  test -e "$ROOT_DIR/$previous/DONE" || {
    echo "[cosmos-submit][error] previous stage is incomplete: $previous"; exit 2;
  }
  test ! -e "$ROOT_DIR/$THROUGH" || {
    echo "[cosmos-submit][error] target stage already exists: $THROUGH"; exit 2;
  }
fi

for ((index=start_index; index<${#STAGES[@]}; index++)); do
  stage="${STAGES[$index]}"
  job_raw=$(sbatch --parsable --time="${TIMES[$index]}" \
    --gres="${GRES[$index]}" \
    "${dependency[@]}" --export=ALL,STAGE="$stage",ROOT_DIR="$ROOT_DIR" \
    scripts/cosmos2020_rws_h100.slurm)
  job="${job_raw%%;*}"
  jobs+=("$stage=$job")
  dependency=(--dependency="afterok:$job")
  if [[ -e "$ROOT_DIR" || "$stage" == "$THROUGH" ]]; then found=1; break; fi
done
if [[ "$found" != "1" ]]; then
  echo "[cosmos-submit][error] argument must be smoke,n5k,n20k,n40k,full"; exit 2
fi

stamp=$(date +%Y%m%d_%H%M%S)
log="$LOG_DIR/submit_cosmos2020_${stamp}.log"
printf '%s\n' "${jobs[@]}" | tee "$log"
printf 'monitor: squeue -j %s\n' "$(printf '%s,' "${jobs[@]#*=}" | sed 's/,$//')"
printf 'after completion: sacct -j %s --format=JobID,State,Elapsed,AllocTRES,ExitCode\n' \
  "$(printf '%s,' "${jobs[@]#*=}" | sed 's/,$//')"
printf 'submission_log=%s\n' "$log"
