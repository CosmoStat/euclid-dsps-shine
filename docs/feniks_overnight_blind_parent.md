# Overnight: blind parent refinement and independent posterior training

## Two independent branches, not a return to supervised parent fitting

1. **Conditional parent**: learn parent weights from catalogue flux/error/mask
   only, with a nested SFH refinement of the existing component family.
2. **Parent to posterior**: generate fresh simulations from the old learned,
   frozen parent and train a supervised 15D conditional posterior. Evaluate
   independent simulations and the coherent target separately.

The parent branch never calls q and never reads target latent columns in its fit.
The posterior branch never updates the parent. No NUTS, SMC, per-object MCMC,
or truth-trained replacement prior. Both remain exploratory, not paper approval.
The target catalogue, decoder, noise, selection and coordinate map are unchanged.

## Exactly what changes in the parent

The baseline has 256 physical-gated components over 32768 independent native
15D anchors with fixed Gaussian kernels. Its SFH law within each component
cannot be freely adjusted. The new experiment splits each component into two
overlapping SFH subcomponents: **512 weights**, retaining the original kernels.

Fit one leading PCA direction of the ten normalized SFH contrasts under the
unweighted-component reference mixture. Within each physical component, center
and scale its projection using that component's anchor weights. A logistic gate
with probability floor 0.05 defines two soft groups. The sign is just a PC score,
NOT high/low absolute star-formation rate. No catalogue or anchor zero is edited,
and all ten SFH coordinates remain active. The unchanged continuous kernels
still smooth discrete features; this does not implement a point mass at zero.
No target truth determines the gates.

For original `g_j = sum_l A_lj K_l`, construct

```text
sum_s h_ljs = 1
Z_js = sum_l A_lj h_ljs
g_js = sum_l A_lj h_ljs K_l / Z_js
g_j = sum_s Z_js g_js
```

Thus all densities are normalized in 15D, and every old population remains
exactly representable using `u_js = u_j Z_js`. Learning relative subcomponent
weights changes the SFH/physical associations; it is not an unrestricted flow
or full recovery of the SFH conditional. One PC split may be insufficient.
It cannot repair missing kernel support, extreme kernel tails, or arbitrary
within-subcomponent correlations. This is one falsifiable family change, not
a claim that every previous failure came from SFH.

The source reference geometry was developed using earlier truth-assisted
diagnostics. The **new population fit** is blind, but this is not a completely
blind method-development or final independent evaluation dataset.

## Reuse 2097152 simulations, zero new DSPS calls for the parent

Replay the original seed and sampling order to recover the generating anchor.
Require exact equality to each saved 15D latent draw. Then assign a subcomponent
label using its anchor gate and an independent RNG. The latent draw, flux,
noise, selection flag and row role remain byte-identical. If replay fails,
stop; never guess labels, approximate the anchor, or silently resimulate.

The relabeled bank is a Monte Carlo sample of the split components. Its parent
class frequencies are `Z_js / 256`, NOT uniform across the 512 classes.
Classifier frequencies use actual selected calibration counts. Selection
efficiencies use selected/total counts, including all rejected parents.
Reserved-bank predictive weights use `u_js / n_parent,js`, not just `u_js`.

Train one new 54-input, width-256/depth-3 classifier with the baseline training
and stopping settings: batch 2048, LR 3e-4, at most 600 epochs, fixed validation,
20-epoch stopping blocks, minimum 80 epochs and three gains below 0.001.
Reuse the convex/KKT-certified KL-regularized selected-weight solver and its
four baseline penalty candidates. Choose penalty using held-out observed flux
likelihood, never true redshift/mass/SFH. Correct selected `v` into parent `u`.
Class support must pass; no new alpha floor or unconstrained inverse selection.

## Matched control and evaluations

The same new classifier also fits a **tied** population with original SFH
proportions. Its selected component ratio combines subcomponents using
`Z_js alpha_js / sum_s Z_js alpha_js`; summing classifier probabilities alone
would be incorrect when finite calibration frequencies differ.

This distinguishes new family flexibility from changing the classifier.
Expanded and tied fits select their own penalty by the same rule; KL penalties
live in their respective selected simplexes, not an identical numerical prior.
Weak subcomponent support blocks the comparison rather than pretending the
feasible model families are unchanged.

After fitting, report target truths for diagnostic evaluation only:

- Baseline, tied and expanded parent and selected marginals; five-dimensional,
  SFH and full joint-15D sliced Wasserstein metrics.
- Physical/SFH cross-correlations and all 15 marginal errors.
- Reserved-bank flux CDFs, joint asinh-flux SW, raw bright-tail amplitudes and
  tail probabilities, with effective row counts and parent selection fraction.
- Paired observed-validation log-likelihood gain for expanded versus tied.
  This validation selected the penalties; its SE is descriptive, not a fresh
  confirmatory significance test.

All old parent/CDF/ratio failures remain in the manifest and decision. No
truth-optimized capacity fit or threshold relaxation is added. The two branches
do not automatically promote the new parent or start another posterior from it.

## Resources and artifacts

Parent: CPU prepare (30 min), four CPU relabel tasks (40 min each, concurrency 4),
one H100 classifier plus parent/tied solvers (180 min), CPU report (30 min).
No new DSPS. Relabeling materializes a local bank copy on scratch; it is not
another forward simulation campaign. Resume retains complete blocks/checkpoints.

Posterior: [existing fresh-parent pipeline](feniks_parent_to_posterior.md),
1048576 new simulations, four H100 bank tasks, one H100 15D flow training job,
one H100 two-cohort evaluation and a CPU report. At most 200 epochs, two experts,
12 spline layers/expert, 256 hidden width, 16 bins, full 15D output.

Combined defaults: **at most five H100s concurrently; 14 H100-hours allocated-time
ceiling**, not predicted runtime or a guarantee of completion before morning.
Both branches save immutable code/config/source hashes, receipts and checkpoints.

Local verification covers actual tiny classifier/flow training, numerical
selection/nesting tests, interrupted relabel/resume, source immutability and
mocked Slurm dependencies/duplicate guards. Actual remote anchor replay remains
a runtime contract. Legacy one-row/batch CLI smoke commands cannot start here:
their AGENTS-listed configuration files are absent. No new local DSPS result
or remote submission is claimed.

## Launch everything once on Jean-Zay

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
bash scripts/submit_feniks_overnight.sh \
  "$BASE/avi_reference_to_parent_20260927_184817"
)
```

The wrapper reuses the latest matching-source posterior and conditional-parent
roots instead of creating duplicate work. It leaves active jobs alone, resumes
inactive incomplete runs, and fails closed if SLURM cannot be queried.

One reconnect-safe watcher:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_overnight_latest.env"
bash scripts/watch_feniks_overnight.sh "$NIGHT"
```

After a failed submission or time limit, inspect logs/accounting, then activate
shine and run `bash scripts/submit_feniks_overnight.sh --resume "$NIGHT"`.
This reuses frozen code and source artifacts; no active jobs are canceled.
Never modify frozen experiment configuration in place.

Local small-file retrieval:

```bash
bash scripts/rsync_feniks_coherent_results.sh conditional_parent
bash scripts/rsync_feniks_coherent_results.sh parent_to_posterior
```

Parent reading order: `report/parent_physical.png`, `physical_sfh_correlations.png`,
`comparison.png`, `classifier_and_sfh_weights.png`, then `joint.csv`,
`observable_joint.csv`, `observable_predictive.csv`, `tied/comparison.json`
and `report/DECISION.json`. Posterior: `report/calibration_comparison.png`,
`training.png`, PIT plots and selected-aggregate closure. Each run has a roadmap.
