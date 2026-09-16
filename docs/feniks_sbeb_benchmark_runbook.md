# FENIKS SBEB benchmark on Jean-Zay

This suite compares the current warm-start model with a genuinely random
four-expert encoder trained for 180 selected-sleep epochs. The scratch q uses
the embedded source prior during sleep, then its first SBEB M-step starts from
an identity prior; the warm branch starts from the current learned prior. It
freezes one object-ID split and evaluates observed `lsst_r` cuts 25, 27 and 29.

## Scientific matrix

The frozen-q diagnosis has 24 cells:

- warm q plus learned prior versus scratch q plus identity prior starting point;
- unweighted dense joint `raw_q` versus ordinary full-15D importance weights;
- selected-density ablation versus the parent objective with `+log(alpha)`;
- observed cuts 25, 27 and 29.

Only eight valid population trajectories are continued for four EM cycles:

- raw-q SBEB at `r < 29`, warm and scratch;
- ordinary-IW, selection-corrected EM at `r < 25, 27, 29`, warm and scratch.

Each cycle runs five prior sweeps followed by 24 q-refresh epochs. Every cycle
keeps its prior, encoder, optimizer resume states, dense inference bank,
population corners, individual corners, PIT and truth-versus-posterior plots.
Every final trajectory also runs the controlled `Q0/P0`, `Q4/P0`, `Q0/P4`,
`Q4/P4` endpoint factorial, with its own individual and population corners,
MIRA, PIT, coverage and causal-effect tables.
The final state of every trajectory is inferred on the exact eight historical
NUTS observations.

## Launch

From the repository checkout on Jean-Zay:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine

BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
SOURCE="$BASE/avi_encoder_experiments_v6"
Q_TRAIN="$BASE/avi_b4_prior_coadaptation_20260913_004803"
PRIOR="$BASE/avi_expert_prior_followup_v2"
SELECTION="$BASE/avi_selection_nuts_sfh_validation_20260913_170521"
NUTS="$BASE/frozen_geometry_nuts_observed8_dense_depth6_v1"
TAG=$(date +%Y%m%d_%H%M%S)
ROOT="$BASE/avi_sbeb_benchmark_$TAG"

bash scripts/submit_feniks_sbeb_benchmark.sh \
  "$SOURCE" "$Q_TRAIN" "$PRIOR" "$SELECTION" "$NUTS" "$ROOT" 8 \
  | tee "$BASE/avi_sbeb_submission_$TAG.txt"

printf 'export ROOT=%q\n' "$ROOT" > "$BASE/avi_sbeb_latest.env"
```

The last argument is array concurrency. Every task uses four H100s, so `8`
permits a 32-H100 peak. Use `4` for a 16-H100 peak.

Preparation is synchronous and must print the exact counts for all cuts before
the first `sbatch`. It fails unless every cut has at least 4,096 selected train
objects, 512 selected validation objects and 1,000 selected blind objects.
On the local 50k catalogue, the deterministic split gives respectively:

| r cut | selected total | selected train | selected validation | selected blind |
|---:|---:|---:|---:|---:|
| 25 | 7,667 | 5,455 | 1,088 | 1,124 |
| 27 | 24,688 | 17,380 | 3,646 | 3,662 |
| 29 | 47,076 | 33,047 | 6,903 | 7,126 |

The preparation recomputes these counts from the archived remote inputs and
aborts before submission if they differ in a way that violates the gates.

## Monitor

```bash
source "$BASE/avi_sbeb_latest.env"
bash scripts/watch_feniks_sbeb_benchmark.sh "$ROOT"
```

After reconnecting to another login node:

```bash
source "$BASE/avi_sbeb_latest.env"
source "$ROOT/JOBS.env"
squeue -r -j "$ALL_JOBS" -o "%.24i %.18T %.12M %R"
sacct -X -j "$ALL_JOBS" --format=JobID%24,State%18,Elapsed,ExitCode
```

The watcher distinguishes bootstrap, factor fits, factor inference, every EM
cycle, final inference, NUTS inference and report completion. A missing queue
entry is not success; require `0:0` plus the stage `FINAL.json` receipt.

## Primary outputs

```bash
column -s, -t < "$ROOT/cohort_counts.csv"
column -s, -t < "$ROOT/factor/report/factor_scorecard.csv"
column -s, -t < "$ROOT/report/trajectory_scorecard.csv"
cat "$ROOT/report/REPORT.md"
cat "$ROOT/report/FINAL.json"
```

Plots are under:

```text
factor/report/population/<cell>/
factor/report/individual/<cell>/
factor/report/calibration/<cell>/
trajectories/<track>/inference/report/cycle_XX/
trajectories/<track>/endpoint_factorial/report/
report/nuts/<track>/
```

## Selective rsync

The dense posterior banks are intentionally excluded from this lightweight
readback. Run from the local workstation:

```bash
REMOTE_ROOT=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111/avi_sbeb_benchmark_YYYYMMDD_HHMMSS
LOCAL=/home/maxime/src/DSPS/outputs/$(basename "$REMOTE_ROOT")
mkdir -p "$LOCAL"

rsync -avh --progress \
  -e "ssh -J mr287471@hubble.extra.cea.fr" \
  --include='*/' \
  --include='FINAL.json' \
  --include='MANIFEST.json' \
  --include='REPORT.md' \
  --include='*.csv' \
  --include='*.png' \
  --include='JOBS.env' \
  --exclude='*' \
  urx63nr@jean-zay.idris.fr:"$REMOTE_ROOT/" "$LOCAL/"
```

Rerun a failed stage only after reading its `.err` log. The stage directories
are resumable and checksum-locked; do not delete successful receipts or reuse a
root with changed code or inputs.
