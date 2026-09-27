# Bounded reference redesign qualification

## Why this run

The completed reference audit (`avi_coherent_reference_audit_20260927_151331`)
certifies a 0.1231 optimum residual on its TRAIN projected-CDF features. Training
the same classifier longer cannot make its fixed component family reproduce
those features. Rare bright predictions dominate the raw r-flux distance, but
the aggregate diagnostic did not identify their physical cause.

This run changes reference geometry, not the target catalogue, posterior flow,
selection, or observation model. It does not train a network or repeat bootstrap
sweeps. All old artifacts stay immutable. New code is a separate frozen snapshot.

## Four controlled capacity cells

| Cell | Coordinates / components / kernels |
|---|---|
| `legacy` | Exact old 64-component reference, scalar Gaussian width 0.15 |
| `affine_mass` | Same anchor memberships and width; affine normalized log10 mass |
| `local_128` | Affine log-mass; 127 local components and one separate broad component |
| `local_256` | Affine log-mass; 255 local components and one separate broad component |

All use the existing independent native 32768 joint 15D reference anchors. No
target labels, fitted population weights or q samples construct the new bases.
The ten SFH coordinates remain joint anchor coordinates plus stochastic kernels.
This retains reference physical/SFH associations, not a proof of the target's
conditional SFH distribution. SFH mismatch remains a possible source of bias.

Mass is already log10 mass: an additional asinh transform followed by Gaussian
kernels gives an ideal mass distribution with divergent E[Mstar]. Affine log10
mass instead gives a finite Gaussian-logmass moment in each finite kernel.
This fixes an analytic tail risk; it does not establish the cause of the saved
finite-sample bright outliers. No clipping, row removal or post-hoc truncation.

For local cells, k-means centers cover joint physical coordinates. Each anchor
has soft memberships among its four nearest centers, proportional to
`exp(-2*(distance/fourth-nearest-distance)^2)`. Each component's memberships are
normalized over anchors. One independent component covers all anchors uniformly;
there is no compulsory broad floor mixed into every local component.

Physical kernel widths are 0.15 times coordinate-wise RMS distances to the
32 nearest reference anchors, bounded to [0.005, 0.15] in normalized space.
All ten SFH widths stay 0.15. Both local resolutions use identical kernels.
The components are exactly normalized 15D densities:

`g_j(x) = sum_l A_lj Normal(x; anchor_l, diag(h_l^2)), sum_l A_lj = 1`.

The inverse coordinate Jacobian defines the corresponding physical density.
Old scalar-bandwidth sampling streams and coordinate specifications remain valid.
Affine vs local changes both gates and physical widths; this is explicitly not
a full factorial separation of those two changes. Local 128 vs 256 isolates
component resolution with the same anchors/kernels.

## Capacity objective and decision

The analytic component CDF is fitted to parent TRAIN truth with a simplex
minimax LP: minimize the largest absolute residual on 25 thresholds in each
of five axes plus 16 fixed random physical projections. Only diagnostic weights
use truth. They are explicitly marked `production_prior=false`.

Independent validation supplies CDF residuals, 1D Wasserstein/IQR, physical and
SFH sliced-Wasserstein, and plots. Test truth is not read by this pipeline.
The previously inspected test results are not a pristine future paper holdout;
use a new independent evaluation sample for final scientific claims.

Engineering gates fixed before launch:
- finite analytic mean linear mass;
- maximum validation CDF residual <= max(0.03, 2 x empirical comparator);
- physical SW <= max(0.035, 2 x empirical comparator);
- maximum physical 1D W1/IQR <= 0.1.

The comparator is an independent TRAIN subsample versus validation. These are
engineering tolerances, not calibrated hypothesis-test confidence intervals.
Legacy mixed projections have a different mass coordinate. Physical SW and
physical marginal plots share units across all candidates; do not compare the
legacy joint-CDF objective as though it used identical projected features.
The smallest passing non-legacy basis is preferred, then lower validation CDF.
No sharp SFH posterior requirement is introduced; SFH errors are still reported.

## Paired bright-object replay in parallel

Use only reserved selected bank rows (role 4): 32 influential upper-r-flux rows
under saved parent weights, plus 32 uniform reserved controls. The brightness
threshold comes from selected **validation** q99.9, not test. Reconstruct exact
original anchor indices, Gaussian perturbations and per-band observation noise
from saved block seeds. Mismatched replay RNG aborts before DSPS.

Replay 64 objects in four cases: original native anchor, saved old perturbation,
affine-mass perturbation, and local-kernel perturbation. Thus **256 DSPS rows**,
using the existing decoder, filters and noise law. Identical pairs preserve all
15 coordinates and noise draws; the unperturbed anchor is a diagnostic control,
not a deterministic nuisance treatment for production. The two local bases
share this replay because only their mixture gates differ.

Require replay of old observed flux within 1e-5 times its reported error.
`paired_theta.csv`, `paired_photometry.csv`, `replay_summary.csv` and the scatter
plot separate anchor behavior from kernel perturbation. This is a targeted
causal comparison, **not** a population predictive or alpha-efficiency estimate.
A capacity pass plus replay identity permits a new-bank benchmark. Inspect the
paired-tail results before that launch; no automatic production approval.

## Launch and reconnect

In the updated checkout and activated `shine` environment:

```bash
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
SOURCE="$BASE/avi_coherent_inference_20260926_094557"
REDESIGN="$BASE/avi_reference_redesign_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_feniks_reference_redesign.sh "$SOURCE" "$REDESIGN"
```

Preparation: 4 CPU threads, 20-minute ceiling. Then **four CPU cells** (each
4 threads, 25-minute ceiling) run in parallel with **one H100** (4 CPU threads,
15-minute ceiling). CPU report: 5 minutes, `afterany` so missing cells are shown
as BLOCKED. At most 0.25 H100-hours and 8.34 CPU-partition core-hours allocated
(plus one CPU core-hour associated with the GPU replay); these are
ceilings, not runtime estimates. Queue delays are additional. No `--mem` flags.

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_reference_redesign_latest.env"
bash scripts/watch_feniks_reference_redesign.sh "$REDESIGN"
```

For an interrupted job, not merely a disconnected terminal:
`bash scripts/submit_feniks_reference_redesign.sh --resume "$REDESIGN"`.
Completed receipts and source fingerprints are checked; only missing cells are
resubmitted, using the same frozen code. Active jobs prevent duplicate submission.
The completed source inference is never restarted.

Locally, small results only:
`bash scripts/rsync_feniks_coherent_results.sh reference_redesign`.

Look at `report/reference_capacity_comparison.png`, each
`{legacy,affine_mass,local_128,local_256}/capacity/physical.png`,
`replay/paired_bright_replay.png`, then `report/DECISION.json` and
`ROADMAP_STATUS.md`. No bank, basis NPZ or model checkpoint is downloaded.

## Next gate, not the big run yet

A passing run writes `report/REFERENCE_CANDIDATE.json` with basis/coordinate
paths and hashes, **never the truth-fitted weights**. This is a qualification
handoff, not an implemented replacement of the downstream production launcher.
A compatible next reference bank must be generated, its classifier and alpha_j
trained/estimated anew, and the parent fitted from photometry alone. Do not reuse
old component labels, classifiers or selection efficiencies. Preserve the coherent
target and the successful supervised posterior control. Only improved blind
parent and observable closure justify new final 15D posterior training and joint /
conditional calibration. No final NPE continuation on the old overfitted bank.

## Local verification

49 focused tests passed, including the eight new numerical/pipeline/submission
tests, old reference/selection/NPE smoke tests, Jacobians, and source isolation.
Three historical representation tests failed on a pre-existing `celuimport`
syntax error in `tests/test_audit_feniks_coherent_parent.py`; that unrelated user
edit is preserved. Ruff, compileall and shell syntax checks pass. Plots from the
tiny synthetic smoke were inspected. The paired end-to-end smoke uses a simple
forward test double and the real observation noise implementation; actual DSPS
GPU replay is pending the Jean-Zay job. The obsolete AGENTS fit-config paths are
absent locally, so legacy CLI fit checks were not run. No remote job submitted.
