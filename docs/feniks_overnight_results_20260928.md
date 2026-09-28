# Overnight parent and posterior review, 2026-09-28

## Evidence and scope

Local synchronized runs in `outputs/forward_population_results/`:

- `avi_conditional_parent_20260927_234702`
- `avi_parent_to_posterior_20260927_234702`

Verified 8 and 33 available artifact hash references respectively, with zero
mismatches. Three and four referenced large artifacts are intentionally absent;
simulation banks are also absent locally. In particular neither posterior
`evaluation/*/draws.npz` nor trained EQX files are local. The user supplied
completed posterior SLURM states; this review does not query live SLURM.
The conditional population stderr records a genuine finalization exception.

This is a coherent synthetic benchmark, NOT calibration on real sky galaxies.
Its target test split has already been inspected during development. A final
paper evaluation needs a genuinely untouched set.

## Bottom line

The fresh supervised posterior has learned substantial conditional structure,
but is neither converged nor calibrated across both cohorts. Its aggregate
physical marginals look substantially better than its individual calibration.
Good aggregate agreement does not certify correct per-galaxy uncertainty.

The expanded parent was fitted blindly from photometry, with no target theta
or q feedback. Its scientific comparison is still missing because the tied
control crashed after optimization. This is a reproducible arithmetic bug,
not evidence that the expanded population model failed scientifically.

There is no justification for regenerating either simulation bank now.

## Conditional parent: complete fit, incomplete comparison

The expanded 512-subcomponent parent has sum(u)=1 and sum(v)=1 to floating-point
precision. Its selection efficiency is 0.596694. All four expanded and four
tied regularized solves have saved KKT gaps <=2e-6. Both paths select penalty
0.003. This certifies the numerical objectives, not population recovery.

The classifier reaches the 600-epoch cap, not its plateau rule. Best validation
NLL is 2.275745 at epoch 566; independent audit NLL is 2.283322 versus null
6.050745. Recent block gains are 0.002186, 0.008527 and zero. Independent ratio
moment error is 0.032400, so ratio accuracy is not newly validated. Raw NLL must
not be compared directly with the old 256-class classifier's NLL.

### Exact failure

`conditional_reference.tied_selected_ratios` computes grouped efficiencies as
`sum_s Z_js * alpha_js`. Saved `Z` row sums differ from one by up to 2.89e-15.
For 21 components both sub-efficiencies are exactly one, and summation produces
a grouped efficiency just above one, at most **1.0000000000000016**.
`parent_from_selected` correctly rejects probabilities above one. The failure
is at `feniks_conditional_parent.py:375`, after writing `tied/regularization.csv`.
Consequently the tied parent and report are not finalized.

Fix the grouped probability arithmetic while preserving the nested distribution
and rejecting genuinely invalid inputs; do not globally weaken probability
validation or clip arbitrary efficiencies. Add an all-selected regression test.
Then recover only the tied control and report under a new recorded code
snapshot. The completed expanded fit, classifier and replayed banks are reusable.
The tied candidate vectors were not saved, so that small optimization path needs
recomputation; classifier and DSPS work does not.

An ordinary `--resume` uses the old frozen code and would repeat the bug.
The gain from SFH flexibility cannot be claimed until expanded/tied/baseline
parent, selected and observable comparisons are produced.

## Posterior: training and calibration

The bank contains 1,048,576 direct-parent simulations, including rejected rows.
Selected train/validation/evaluation role counts are 537,106 / 63,472 / 31,485.
Training uses unit-weight simulator targets, no importance resampling, no RWS,
no target-catalogue truth and no posterior-to-parent update. All 15 dimensions
remain active. Measured selection fraction 0.602782 agrees closely with the
frozen parent's 0.602352.

The two-expert spline posterior stops at 200 epochs with `plateau=false` and
`reason=maximum_epoch`. Best validation checkpoint is epoch 181, NLL -16.949721;
the best at epoch 140 was -14.441490. Recent best-NLL gains are 1.6511, 0.3920,
0.4651. Last validation NLL is -16.033746: evaluation uses the best, not the last
checkpoint. Validation is visibly noisy; reducing LR at 50-epoch intervals
produces substantial gains. More optimization is motivated, but more epochs
alone are not a calibration guarantee.

The trained transport audit passes on its checked contexts, worst discrepancy
about 4.94e-13 against tolerance 1e-5. This is not an exhaustive tail audit.

### Two cohorts, different questions

Each evaluation uses 2048 unique galaxies and 512 joint draws per galaxy.

- **In-model:** same frozen parent and simulator as training, independent rows.
  Failure here implicates posterior approximation/training/evaluation, without
  needing target-parent mismatch as an explanation.
- **Coherent target:** the benchmark target has a different parent. Failures
  combine remaining posterior error and population mismatch; they cannot all
  be assigned to one cause yet.

| Coordinate | In-model 68 / 95 coverage (%) | Target 68 / 95 coverage (%) |
|---|---:|---:|
| Redshift | 81.64 / 98.29 | 72.85 / 95.85 |
| Log stellar mass | 73.58 / 96.44 | 60.16 / 88.23 |
| Log metallicity | 69.78 / 95.41 | 74.02 / 96.19 |
| Dust Av | 72.17 / 95.61 | 54.20 / 81.69 |
| Dust slope | 69.19 / 95.26 | 60.64 / 89.01 |

The expected columns are 68 and 95 percent. At N=2048 the nominal binomial
standard error is about 1.03 and 0.48 percentage points respectively, before
finite-draw and multiple-comparison qualifications. Major deviations are not
plausibly explained simply by having too few evaluated galaxies.

This is **not uniform posterior narrowing**. In-model redshift overcovers, while
target mass and dust undercover. PIT also fails for in-model z/mass/metallicity;
target PIT fails throughout. In-model metallicity's acceptable interval coverage
does not imply a uniform PIT or correct joint distribution.

### Tail problem masked by central diagnostics

In-model SFH 68/95 coverage and PIT pass the configured marginal tolerances.
However `selected_aggregate_joint.csv` reports SFH SW **4.4398e13**. Marginal
W1/IQR is **1.7329e14** for `sfh_dlog_sfr_05` and 53.24 for contrast 07.
Target SFH SW is 0.5876, with contrast 05 W1/IQR 1.4472.

The metrics include the full support of 32768 sampled joint draws, while the
histogram displays only truth quantiles 0.5-99.5%. It does not renormalize away
excluded tail probability, but extreme values remain outside the visible axes.
Central coverage and PIT can miss a very small mass at enormous distances.
Therefore the watcher SFH PASS must not be read as full SFH validation.

The saved coordinate transform applies `sinh` to SFH latent coordinates. Large
latent excursions can thus amplify dramatically in physical coordinates.
This is a plausible mechanism, not a demonstrated attribution: inspect the
saved draws, their inverse coordinates, IQR denominator, object IDs and expert
contributions before changing the flow or claiming a sampler bug. The available
CSV cannot determine outlier frequency or which expert is responsible.
Do not clip draws or remove dimensions to make metrics pass. These are SFH
contrast coordinates, not direct star-formation-rate bins; this anomaly does
not dispute that a physical SFH can contain zero star formation.

### What the aggregate does establish

Physical selected-aggregate SW is 0.02478 in-model and 0.05723 on target.
Target marginal W1/IQR: z 0.03654, mass 0.01121, metallicity 0.02764,
Av 0.10257, dust slope 0.01664. This demonstrates reasonable bulk marginal
reconstruction, not a normalized-parent recovery or individual calibration.
In particular, excellent mass aggregate agreement coexists with only 60.16%
target coverage at nominal 68%.

## Figures to inspect

All links below refer to the synchronized posterior run.

1. [Calibration comparison](../outputs/forward_population_results/avi_parent_to_posterior_20260927_234702/report/calibration_comparison.png):
   compare blue in-model with red target; dashed lines are nominal coverage.
2. [Training](../outputs/forward_population_results/avi_parent_to_posterior_20260927_234702/report/training.png):
   continuing best-NLL improvement and LR-driven changes; no convergence claim.
3. [Selected aggregate](../outputs/forward_population_results/avi_parent_to_posterior_20260927_234702/report/selected_aggregate_15d.png):
   useful bulk comparison, with a restricted display range. Read the
   [full-support joint metrics](../outputs/forward_population_results/avi_parent_to_posterior_20260927_234702/report/selected_aggregate_joint.csv)
   alongside it, especially SFH.
4. [In-model physical PIT](../outputs/forward_population_results/avi_parent_to_posterior_20260927_234702/report/in_model_physical_pit_truth.png)
   and [target physical PIT](../outputs/forward_population_results/avi_parent_to_posterior_20260927_234702/report/coherent_target_physical_pit_truth.png):
   the full rank shapes reveal calibration discrepancies hidden by one interval.

## Next work, ordered to avoid wasted compute

1. **Recover the conditional-parent comparison**, no new DSPS or classifier
   training. Fix roundoff, reuse completed artifacts and finish tied/report.
2. **In parallel, audit existing posterior draws**, no training. Measure extrema,
   tail probabilities, latent excursions and which objects/experts contribute;
   distinguish an implementation fault from an undertrained distribution tail.
   Reuse remote NPZ/checkpoint artifacts rather than downloading banks.
3. **After that diagnosis, one bounded posterior continuation** on the same
   fresh bank, retaining the best baseline checkpoint and reducing LR if
   justified. Evaluate fixed validation cohorts at milestones, including tails,
   coverage and PIT; lower NLL alone is not a promotion criterion. A completed
   run's ordinary resume does not extend its frozen epoch cap.
4. **Freeze an improved blind parent only on comparison evidence**, then train
   or adapt its posterior using simulator truth, never catalogue latent truth
   or q-generated training targets. The current posterior belongs to the old
   frozen parent, not the incomplete new-parent comparison.

No new large simulation campaign is justified at this point. This review changes
neither source code, thresholds, checkpoints nor scientific promotion status.
