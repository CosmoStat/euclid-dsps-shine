#!/bin/bash
# Usage: submit_feniks_sbeb_continuation.sh PARENT_ROOT CONTINUATION_ROOT [dependency_job]
set -Eeuo pipefail
trap 'echo "SBEB continuation submission stopped at line $LINENO: $BASH_COMMAND" >&2' ERR

PARENT_ROOT="$(realpath "${1:?completed SBEB benchmark root}")"
export SBEB_CONTINUATION_ROOT="$(realpath -m "${2:?new continuation root}")"
PARENT_DEPENDENCY="${3:-}"
TRACKS=(warm_iw_r27 warm_iw_r29)
CYCLES=3

test -s "$PARENT_ROOT/MANIFEST.json"
test -s "$PARENT_ROOT/JOBS.env"
test ! -e "$SBEB_CONTINUATION_ROOT"

if [[ -z "$PARENT_DEPENDENCY" ]]; then
  PARENT_DEPENDENCY=$(
    bash -c 'source "$1/JOBS.env"; printf "%s" "$FINAL_REPORT_JOB"' \
      _ "$PARENT_ROOT"
  )
fi
[[ "$PARENT_DEPENDENCY" =~ ^[0-9]+$ ]] || {
  echo "dependency job must be a numeric Slurm job ID" >&2
  exit 2
}

REPO="$(pwd -P)"
test -d "$REPO/filters"
ARCHIVE="$SBEB_CONTINUATION_ROOT.code.tar"
test ! -e "$ARCHIVE"
mkdir -p "$(dirname "$SBEB_CONTINUATION_ROOT")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$ARCHIVE" \
  euclid_dsps scripts configs pyproject.toml
tar --dereference -rf "$ARCHIVE" filters
DIGEST=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
export AVI_EM_CODE="$SCRATCH/feniks_sc_drws_runtime/code/sbeb-continuation-$DIGEST"
mkdir -p "$AVI_EM_CODE"
tar -xf "$ARCHIVE" -C "$AVI_EM_CODE"
if [[ ! -e "$AVI_EM_CODE/Data" && ! -L "$AVI_EM_CODE/Data" ]]; then
  ln -s "$REPO/Data" "$AVI_EM_CODE/Data"
fi
mkdir -p "$SBEB_CONTINUATION_ROOT"
printf '%s\n' "$AVI_EM_CODE" > "$SBEB_CONTINUATION_ROOT/CODE_DIR"

cd "$AVI_EM_CODE"

TRACK_ROOTS=()
INFERENCE_ROOTS=()
for TRACK in "${TRACKS[@]}"; do
  PARENT_EM="$PARENT_ROOT/trajectories/$TRACK/em"
  test -s "$PARENT_EM/MANIFEST.json"
  test -s "$PARENT_EM/cycles/cycle_04/FINAL.json"

  mapfile -t COMPONENTS < <(python - "$PARENT_EM" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
final = json.loads((root / "cycles/cycle_04/FINAL.json").read_text())
if final.get("status") != "EM_CYCLE_COMPLETE":
    raise SystemExit("parent cycle 4 is incomplete")
if manifest.get("e_step_mode") != "ordinary_iw":
    raise SystemExit("continuation requires ordinary_iw parent")
if not manifest.get("selection_objective_enabled"):
    raise SystemExit("continuation requires the corrected selection objective")

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

encoder = Path(final["encoder"])
prior = Path(final["prior"])
if sha(encoder) != final["encoder_sha256"]:
    raise SystemExit("cycle-4 encoder hash mismatch")
if sha(prior) != final["prior_sha256"]:
    raise SystemExit("cycle-4 prior hash mismatch")

print(manifest["source_training"])
print(encoder)
print(prior)
print(manifest["selection_root"])
PY
  )
  [[ "${#COMPONENTS[@]}" -eq 4 ]] || {
    echo "failed to resolve cycle-4 components for $TRACK" >&2
    exit 2
  }

  EM_ROOT="$SBEB_CONTINUATION_ROOT/$TRACK/em"
  INFERENCE_ROOT="$SBEB_CONTINUATION_ROOT/$TRACK/inference"
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
    python -m scripts.feniks_avi_em prepare-components \
      --runtime-root "${COMPONENTS[0]}" \
      --initial-encoder "${COMPONENTS[1]}" \
      --initial-prior "${COMPONENTS[2]}" \
      --selection "${COMPONENTS[3]}" \
      --root "$EM_ROOT" --inference-root "$INFERENCE_ROOT" \
      --cycles "$CYCLES" --q-epochs 24 --prior-sweeps 5 \
      --cycle-offset 4 \
      --e-step-mode ordinary_iw --selection-objective corrected \
      --track "${TRACK}_continuation_c5_c7"
  TRACK_ROOTS+=("$EM_ROOT")
  INFERENCE_ROOTS+=("$INFERENCE_ROOT")
done

python - "$PARENT_ROOT" "$SBEB_CONTINUATION_ROOT" "$PARENT_DEPENDENCY" \
  "$DIGEST" <<'PY'
import json
import sys
from pathlib import Path

parent, root, dependency, digest = sys.argv[1:]
payload = {
    "version": 1,
    "suite": "feniks_sbeb_cycle4_continuation_v1",
    "parent_root": str(Path(parent).resolve()),
    "parent_global_cycle": 4,
    "continuation_cycles": 3,
    "global_cycle_mapping": {"0": 4, "1": 5, "2": 6, "3": 7},
    "tracks": ["warm_iw_r27", "warm_iw_r29"],
    "q_epochs_per_cycle": 24,
    "prior_sweeps_per_cycle": 5,
    "e_step_mode": "ordinary_iw",
    "selection_objective": "corrected",
    "parent_dependency_job": dependency,
    "code_sha256": digest,
    "truth_used_for_training_or_checkpoint_selection": False,
}
(Path(root) / "MANIFEST.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

ALL_JOB_IDS=()
REPORT_JOBS=()
TRACK_JOB_LINES=()

for INDEX in "${!TRACKS[@]}"; do
  TRACK="${TRACKS[$INDEX]}"
  export AVI_EM_ROOT="${TRACK_ROOTS[$INDEX]}"
  export AVI_EM_INFERENCE_ROOT="${INFERENCE_ROOTS[$INDEX]}"
  DEPENDENCY="$PARENT_DEPENDENCY"
  CYCLE_JOBS=()

  for CYCLE in $(seq 1 "$CYCLES"); do
    export AVI_EM_MODE=cycle AVI_EM_CYCLE="$CYCLE"
    RAW=$(sbatch --parsable --time=10:00:00 \
      --dependency="afterok:$DEPENDENCY" --export=ALL \
      --output="$AVI_EM_ROOT/logs/cycle-${CYCLE}-%j.out" \
      --error="$AVI_EM_ROOT/logs/cycle-${CYCLE}-%j.err" \
      "$AVI_EM_CODE/scripts/feniks_avi_em.slurm")
    JOB="${RAW%%;*}"
    CYCLE_JOBS+=("$JOB")
    ALL_JOB_IDS+=("$JOB")
    DEPENDENCY="$JOB"
  done

  export AVI_EM_MODE=prepare-inference AVI_EM_PARTICLES=4096
  RAW=$(sbatch --parsable --time=00:30:00 --gres=gpu:1 --cpus-per-task=12 \
    --dependency="afterok:$DEPENDENCY" --export=ALL \
    --output="$AVI_EM_ROOT/logs/prepare-inference-%j.out" \
    --error="$AVI_EM_ROOT/logs/prepare-inference-%j.err" \
    "$AVI_EM_CODE/scripts/feniks_avi_em.slurm")
  PREPARE_JOB="${RAW%%;*}"
  ALL_JOB_IDS+=("$PREPARE_JOB")

  export AVI_EM_MODE=infer
  RAW=$(sbatch --parsable --time=06:00:00 --array="0-${CYCLES}%1" \
    --dependency="afterok:$PREPARE_JOB" --export=ALL \
    --output="$AVI_EM_ROOT/logs/infer-%A_%a.out" \
    --error="$AVI_EM_ROOT/logs/infer-%A_%a.err" \
    "$AVI_EM_CODE/scripts/feniks_avi_em.slurm")
  INFERENCE_JOB="${RAW%%;*}"
  ALL_JOB_IDS+=("$INFERENCE_JOB")

  export AVI_EM_MODE=report
  RAW=$(sbatch --parsable --time=03:00:00 --gres=gpu:1 --cpus-per-task=24 \
    --dependency="afterok:$INFERENCE_JOB" --export=ALL \
    --output="$AVI_EM_ROOT/logs/report-%j.out" \
    --error="$AVI_EM_ROOT/logs/report-%j.err" \
    "$AVI_EM_CODE/scripts/feniks_avi_em.slurm")
  REPORT_JOB="${RAW%%;*}"
  ALL_JOB_IDS+=("$REPORT_JOB")
  REPORT_JOBS+=("$REPORT_JOB")

  CYCLE_CSV=$(IFS=,; echo "${CYCLE_JOBS[*]}")
  TRACK_JOB_LINES+=(
    "export ${TRACK^^}_CYCLE_JOBS=$CYCLE_CSV"
    "export ${TRACK^^}_PREPARE_JOB=$PREPARE_JOB"
    "export ${TRACK^^}_INFERENCE_JOB=$INFERENCE_JOB"
    "export ${TRACK^^}_REPORT_JOB=$REPORT_JOB"
  )
done

ALL_JOBS=$(IFS=,; echo "${ALL_JOB_IDS[*]}")
REPORT_JOB_CSV=$(IFS=,; echo "${REPORT_JOBS[*]}")
JOBS="$SBEB_CONTINUATION_ROOT/JOBS.env"
{
  printf 'export SBEB_CONTINUATION_ROOT=%q\n' "$SBEB_CONTINUATION_ROOT"
  printf 'export AVI_EM_CODE=%q\n' "$AVI_EM_CODE"
  printf 'export PARENT_DEPENDENCY=%q\n' "$PARENT_DEPENDENCY"
  printf 'export TRACKS=%q\n' "${TRACKS[*]}"
  printf 'export CYCLES=%q\n' "$CYCLES"
  printf 'export ALL_JOBS=%q\n' "$ALL_JOBS"
  printf 'export REPORT_JOBS=%q\n' "$REPORT_JOB_CSV"
  printf '%s\n' "${TRACK_JOB_LINES[@]}"
} > "$JOBS"

printf 'continuation_root=%s\nparent_dependency=%s\ntracks=%s\ncycles=%s\nreports=%s\npeak_h100s=8\nwatch=bash scripts/watch_feniks_sbeb_continuation.sh %q\n' \
  "$SBEB_CONTINUATION_ROOT" "$PARENT_DEPENDENCY" "${TRACKS[*]}" \
  "$CYCLES" "$REPORT_JOB_CSV" "$SBEB_CONTINUATION_ROOT"
