#!/bin/bash
set -u

REPO_DIR="${REPO_DIR:-$PWD}"
ENV_FILE="${1:-outputs/logs/feniks_smc_teacher_audit_latest.env}"
cd "$REPO_DIR" || exit 1
source "$ENV_FILE"

echo "===== SLURM ====="
squeue -r -j "$AUDIT_JOB,$AUDIT_FINALIZER_JOB" 2>/dev/null || true
sacct -X -j "$AUDIT_JOB,$AUDIT_FINALIZER_JOB" \
  --format=JobID,State,Elapsed,Timelimit,ExitCode,NodeList 2>/dev/null || true

echo
echo "===== ARTEFACTS ====="
bootstrap_prepared=$(find "$BOOTSTRAP_ROOT/galaxies" -name PREP_DONE 2>/dev/null | wc -l)
nuts_chains=$(find "$BOOTSTRAP_ROOT/galaxies" -path '*/nuts/chain_*/DONE' 2>/dev/null | wc -l)
nuts_galaxies=$(find "$BOOTSTRAP_ROOT/galaxies" -mindepth 2 -maxdepth 2 \
  -name DONE 2>/dev/null | wc -l)
distilled_prepared=$(find "$DISTILLED_ROOT/galaxies" -name PREP_DONE 2>/dev/null | wc -l)
printf 'teacher SMC prepared : %s/8\n' "$bootstrap_prepared"
printf 'NUTS chains complete : %s/32\n' "$nuts_chains"
printf 'NUTS galaxies done   : %s/8\n' "$nuts_galaxies"
printf 'distilled q prepared : %s/8\n' "$distilled_prepared"
test -e "$AUDIT_SUMMARY/DONE" && echo "summary             : DONE" \
  || echo "summary             : pending"

echo
echo "===== PAR TACHE ====="
for index in $(seq 0 7); do
  output="outputs/logs/feniks_teacher-${AUDIT_JOB}_${index}.out"
  error="outputs/logs/feniks_teacher-${AUDIT_JOB}_${index}.err"
  state=$(sacct -X -n -j "${AUDIT_JOB}_${index}" --format=State 2>/dev/null \
    | awk 'NF {print $1; exit}')
  phase=$(grep -E '\[teacher-audit\]' "$output" 2>/dev/null | tail -n 1)
  sampler=$(grep -E '\[exact-sampler' "$output" 2>/dev/null | tail -n 1)
  errors=$(grep -Ec 'Traceback|RuntimeError|ValueError|OutOfMemory|FAILED' \
    "$error" 2>/dev/null || true)
  printf '[%s] %-11s errors=%s %s\n' "$index" "${state:-UNKNOWN}" \
    "${errors:-0}" "${phase:-not started}"
  test -n "$sampler" && printf '    %s\n' "$sampler"
done

echo
echo "===== DERNIERES ERREURS ====="
grep -hE 'Traceback|RuntimeError|ValueError|OutOfMemory|FAILED' \
  outputs/logs/feniks_teacher-${AUDIT_JOB}_*.err 2>/dev/null | tail -n 20 || true
