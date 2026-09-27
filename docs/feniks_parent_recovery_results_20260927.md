# Frozen-parent recovery result and overnight decision

Source: `avi_parent_recovery_20260927_205052`, downloaded 2026-09-27.
User-provided Slurm output: 253496, 253497_0/1 and 253498 COMPLETED 0:0.
All 22 artifact references in the downloaded FINAL receipts match their hashes.
The large cache and original simulation/model assets were excluded from rsync;
this review does not re-execute those parts or assert a fresh independent rerun.

## What has actually succeeded

The frozen classifier and regularized selection-corrected inversion recover
two known parent mixtures from weighted reserved-bank photometric observations.
Different bank rows evaluate the recovered densities. No neural training or
new DSPS simulation occurred in this diagnostic; numerical mixture weights
were fitted. The earlier classifier did learn useful component information:
its independent NLL was 1.60638 versus the no-feature reference 5.35584.

| Metric | Learned-parent mixture | Capacity-control mixture |
|---|---:|---:|
| Parent physical SW | 0.018641 | 0.034812 |
| Selected physical SW | 0.012922 | 0.015652 |
| Largest physical parent W1/IQR | 0.037292 (Av) | 0.054970 (z) |
| Parent SFH SW, descriptive | 0.010753 | 0.025313 |
| Parent physical SW with known labels + same penalty | 0.004040 | 0.002415 |
| Effective selected fit rows | 10996 | 10231 |
| Classifier inversion KKT gap | 1.981e-6 | 1.932e-6 |

Both physical parent/selected SW gates (0.05), physical marginal gates (0.1),
known-label controls and effective-row gates pass. Parent vectors normalize to
one and all four penalized solves meet the unchanged 2e-6 KKT tolerance.
This is evidence for practical density recovery in the tested model family,
not unique identification of all component weights or arbitrary populations.

The classifier remains less accurate than the label-aware control; its error
has not disappeared. Independent ratio moment error remains 0.03170, versus
the old 0.030 gate. The conditional block-bootstrap interval for that error is
[0.03056, 0.04512], with centered sampling-noise q95 0.02506. These different
statistics must not be equated or used to call the ratio exact. This audit
conditions on saved calibration/selection efficiencies and bank geometry.

## What has NOT improved automatically

Do not compare 0.01864 to the blind target-fit 0.09751 as a before/after gain.
The former is recovery of a known in-family mixture; the latter compares a
different fit with the actual coherent target parent. The target has not changed,
and neither has its learned production-candidate parent.

Target parent recovery still has Av W1/IQR 0.18677 and physical SW 0.09751.
Four physical marginal gates pass, but the five-dimensional joint gate fails.
The target SFH conditional remains unvalidated. Success on in-family SFH
distribution comparisons does not imply individual SFH inference is identifiable.
Wide calibrated individual SFH posteriors are acceptable.

The new observable checks support a reference/target mismatch hypothesis:

| Compared with coherent target validation fluxes | Learned known mixture | Capacity known mixture |
|---|---:|---:|
| Standardized asinh-flux joint SW, descriptive | 0.02928 | 0.14876 |
| Selected r-flux tail probability above target q99.9 | 0.1608% | 1.4016% |
| Target probability at the same threshold | 0.1066% | 0.1066% |
| Raw r-flux W1 / saved scale | 5.032 | 33091.96 |

The flux scale in this diagnostic has a 1e-30 floor; these are not fractional
errors, and no universal PASS threshold exists for the asinh joint statistic.
The capacity mixture's tail-weighted mean redshift is 0.01736; components 148
and 66 contribute 99.9987% of its upper-excess statistic in this finite bank.
This is descriptive attribution, not proof of a simulator bug or justification
to delete objects/components. The learned mixture assigns those problematic
components negligible weight; its residual bright-tail excess is much smaller.

Thus a physically attractive capacity fit is not automatically a good full
generative parent. The evidence prioritizes joint reference/target mismatch
(including z/SFH/dust correlations and low-z tails), without proving it is the
only cause. No new cutoff, tail clipping or truth-trained replacement is applied.

## Overnight decision

Proceed with the already implemented **fresh-parent supervised 15D posterior**
in a new exploratory root, or let it continue if already submitted. No need to
repeat this diagnostic, blindly extend the classifier, or enlarge the same
population fit before learning from the final inference block.

The [existing launch runbook](feniks_parent_to_posterior.md) specifies 1,048,576
new parent simulations, four H100 bank tasks, then one two-expert 15D flow trained
for up to 200 epochs. No q feedback, no RWS, no catalogue truth as training targets.
Evaluate 2048 unique galaxies with 512 joint draws in each of two cohorts:

1. Independent simulations from the learned parent: calibration under its own
   training distribution. Failure here prioritizes posterior optimization/capacity.
2. Coherent target catalogue: transfer under the imperfect learned parent.
   If only this fails, prioritize parent/reference mismatch, not automatic extra
   posterior epochs. Conditional failures can be inspected from saved draws.

Inspect validation history and trained transport, 68/95 coverage, PIT, median
bias and widths for core z/mass/metallicity, dust and SFH separately. Check
redshift/brightness/SNR strata in addition to global averages before acceptance.
The current report includes global metrics and dense selected-aggregate closure;
conditional-stratum diagnostics remain analysis of the saved draws/observations,
not an additional already implemented automatic acceptance gate.

For a final paper claim, population bias, parameter degeneracies and population
uncertainty remain relevant; obtain an untouched evaluation cohort after methods
are frozen. Repeatedly inspected coherent test data are development evidence.
An overnight posterior run is justified progress, not a guarantee of full parent
recovery or production approval. No new remote submission is claimed here.

## Reading order

- `learned_parent/parent_physical.png`: blue known population versus orange
  photometric recovery; green label-aware control.
- `capacity_control/parent_physical.png`: the more demanding second mixture.
- Each `joint.csv` and `marginals.csv`: full-support density errors.
- Each `observable_joint.csv`, `observable_predictive.csv`, `tail_components.csv`
  and `tail_physical_sfh.csv`: remaining comparison with the coherent target.
- `report/DECISION.json`: all gates and conditional ratio uncertainty.
