# Frozen support probe after controlled local VI

## State and rationale

Operator job 1948458 completed in 21m33s with numerical contract PASS.
All 48 observed final local distributions have bad Pareto-k. Smaller steps
reduce extreme excursions; 16 gradient draws do not systematically recover
support. This is not evidence for population training or a globally adequate
posterior. The illustrated page is `docs/source/feniks_current_status.rst`.

## CPU audit first

Run `scripts/audit_feniks_saved_draws.py CONTROLLED --out NEW_AUDIT` to read
all saved banks, check their hashes against local SUMMARY receipts, and write
concentration.csv, coordinates.csv and standardized weighted covariance
matrices. No decoder evaluations or truth columns are needed. An incomplete
audit leaves its directory for inspection; use a new directory for a retry.
The report keeps nonfinite inputs explicitly. Weighted moments from a bank
with tiny ESS are not reliable posterior moments. Coordinate indices follow
the latent_x order of the source config; they are not physical parameters.

## Fixed GPU experiment

- Same eight observed and eight simulated development contexts. The latter
  must reproduce the saved source inputs exactly or the job stops.
- Use both final slow_mc16 checkpoints, without choosing a favorable step or
  start. This choice tests the previous hypothesis; it is not a best-fit claim.
- Four proposals per start: original local base, base std multiplied by 1.5,
  base std multiplied by 2, and a 50/50 mixture of local-x1.5 and the unchanged
  amortized proposal. If broadening hits a base-scale bound, stop instead of
  silently clipping. Local coupling layers, target and all physics stay fixed.
- The mixture samples a Bernoulli component independently for each draw and
  computes logaddexp(logq_local, logq_anchor) - log(2) for every sample. Never
  assign component-only density or merge normalized weights from separate banks.
- New evaluation keys, independent of previous training/evaluation. Two K128
  replicates per proposal (pooled K256), including a freshly evaluated amortized
  baseline. All candidates reported; no winner selection or adaptive stopping.
- No optimizer is constructed. One H100 / one node, sequential, 3h allocation
  ceiling, 9900s internal time and 50000 charged decoder evaluations. Source
  numerical contract audit still runs, including a few gradient evaluations.
- Checkpoints verified against case receipts, simulated inputs against FINAL;
  selected source hashes pinned at preparation and rechecked at execution.
  Historical source lacks a complete original artifact inventory; this is
  explicitly recorded rather than described as a retrospective certification.

## Commands

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
CONTROLLED="$BASE/frozen_parent_controlled_local_vi_v1"
JAX_PLATFORMS=cpu python scripts/audit_feniks_saved_draws.py "$CONTROLLED" \
  --out "$BASE/frozen_parent_concentration_audit_v1" &&
bash scripts/submit_feniks_sc_drws_support_probe.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_precision_night_v1" "$CONTROLLED" \
  "$BASE/frozen_parent_support_probe_v1"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

Readout uses the existing table reader; `step=0` means no new optimization.
The source checkpoints are the fixed step-64 checkpoints:

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_controlled_local_vi.py "$DIAGNOSTIC_ROOT"
```

A positive result is a development lead requiring independent-cohort
confirmation, not promotion. A negative result does not prove that no
dispersion, mixture or flow architecture can work. Preserve all candidates,
bad-k flags, draws and density conventions. No NPE teacher or population update.
