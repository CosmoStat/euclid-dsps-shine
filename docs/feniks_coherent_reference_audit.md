# Bounded coherent-reference audit

Purpose: identify the blocking error in `avi_coherent_inference_20260926_094557`,
not launch another model sweep. One CPU job, 4 threads, 30-minute allocation
ceiling. No GPUs, DSPS calls, new noisy photometry, neural training or q draws.
Runtime is not guaranteed by the ceiling. Existing simulation banks stay remote.

## Tests and interpretation

1. **Fixed-family capacity:** exact component CDFs in 5 normalized physical
   coordinates, plus 16 fixed random projections, at 25 TRAIN quantiles each.
   A simplex linear program finds diagnostic weights minimizing maximum CDF
   discrepancy from the true parent TRAIN sample. Evaluation on validation and
   test never chooses weights, directions or thresholds. All generated draws
   remain 15D, using the original kernels/conditional SFH associations.
2. **Observable tails:** exact weighted selected role-4 bank flux distributions
   with u_j/r_j weights, not a fresh importance resample. Report full raw and
   asinh-flux W1, tail probability above target q99.9, excess-flux expectation,
   and per-component tail contributions. Upper-capped W1 is explicitly a
   diagnostic comparison; neither training data nor inference are clipped.

Equations for the normalized kernel CDF and minimax LP are in
`euclid_dsps/amortized/reference_capacity.py`. This LP diagnoses a finite set of
distribution features, not the minimum possible SW or unrestricted joint15D
capacity. A large optimum establishes a restriction of those features. A small
optimum does not exclude other misspecification. No truth-fitted diagnostic
weight is saved as `parent.json` or read by a production pipeline.

Source manifests, component receipts and every bank block consumed are verified.
New outputs have their own manifest and frozen code snapshot. Resuming skips
completed audit sections. Original run, simulator, classifier and checkpoints
are never changed. Joint/conditional posterior calibration remains separate.

## Jean-Zay launch

Run in a subshell so a failed check never closes the SSH shell. No abbreviated
SHA comparison with full `git rev-parse HEAD`.

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
INFERENCE="$BASE/avi_coherent_inference_20260926_094557"
AUDIT="$BASE/avi_coherent_reference_audit_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_feniks_coherent_reference_audit.sh "$INFERENCE" "$AUDIT"
)
```

Reconnect-safe watcher, no conda needed beyond a working `python`:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_coherent_reference_audit_latest.env"
bash scripts/watch_feniks_coherent_reference_audit.sh "$AUDIT"
```

After a genuine failed/timeout job (not just a disconnected terminal), reuse the
same frozen snapshot with:

```bash
bash scripts/submit_feniks_coherent_reference_audit.sh --resume "$AUDIT"
```

Locally, retrieve only small reports/configs/logs, no arrays or models:

```bash
bash scripts/rsync_feniks_coherent_results.sh reference_audit
```

Inspect `capacity/cdf_errors.csv`, `capacity/capacity_physical.png`,
`capacity/joint.csv`, `tails/observable_tails.csv`,
`tails/observable_cdfs.png`, `tails/component_tail_contributions.csv`.
`report/FINAL.json` means execution finished, never production approval.
