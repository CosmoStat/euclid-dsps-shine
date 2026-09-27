# Reference-to-parent qualification results, 2026-09-27

Run: `avi_reference_to_parent_20260927_173822`.
Contract: `73f8159b43d5cf0100bce27cbabc405419487d36b6e36040079b76856c85454a`.
Seven present receipt artifacts and the frozen configuration verify. Two excluded
NPZs (`reference/basis.npz`, `qualification/physical_cdf.npz`) are not local.
Downloaded stderr/stdout files are empty. The qualification deliberately returns
exit 2 after saving a scientifically failed result; this is not an optimizer
exception. SLURM output supplied by the user shows downstream GPU jobs cancelled.

## Exact result

| Check | Old minimax objective | Physical integrated-CDF objective | Limit |
|---|---:|---:|---:|
| Physical sliced Wasserstein | 0.0458766 | 0.0276976, PASS | 0.035 |
| Maximum physical marginal W1/IQR | 0.0525071 | 0.0470190, PASS | 0.1 |
| Maximum validation CDF error | 0.0319998, PASS | 0.0338331, FAIL | 0.0324000 |
| Finite mean linear mass | PASS | PASS | Required |

The physical SW improves by 39.63% within this paired comparison. The empirical
comparison is 0.0143031, so the new result is about 1.94 times that distance,
not indistinguishable from it. No uncertainty interval was produced for this
difference. All five physical marginals improve:

| Physical parameter | Old W1/IQR | New W1/IQR |
|---|---:|---:|
| Redshift | 0.03102 | 0.02583 |
| Log stellar mass | 0.05251 | 0.04702 |
| Log metallicity | 0.03386 | 0.01437 |
| Dust Av | 0.05114 | 0.02381 |
| Dust delta | 0.03563 | 0.02800 |

This is evidence that the fitting objective mattered, and that the current
family has useful physical capacity. It is not photometry-only recovery:
these diagnostic weights were fitted using TRAIN truth. No new reference bank,
classifier, blind population or posterior was completed in this run.

## What the remaining failure means

The effective CDF limit is **0.0324**, not the nominal 0.030 quoted provisionally
from configuration. The implemented rule takes max(0.030, 2 x empirical CDF
distance), and the saved comparator is 0.0162. The excess is therefore
**0.001433**, or 0.1433 percentage point of cumulative probability. This is not
relative parameter error and is not proof that it is sampling noise.

The old solution's CDF stayed exactly at 0.03199984, but its gate changed from
FAIL in the preceding experiment to PASS here because the random empirical
comparator changed the effective threshold. Do not attribute that gate change
to a better model. Similarly, old-minimax SW 0.0506 in the preceding run versus
0.0459 here uses different evaluation draws; use the within-run comparison above.

The new solver reached HiGHS Optimal after 16056 iterations. Its original TRAIN
CDF constraint is active: 0.026009839922 against cap 0.026009839922. More identical
iterations are not an identified remedy. Qualification has not saved the new
worst CDF direction/threshold, so the downloaded files cannot establish which
parameter or joint projection causes the remaining maximum. Do not reuse the
previous run's stellar-mass attribution for the new weights without measuring it.

Diagnostic u sums to 0.9999999999999992 and is nonnegative; 62/256 components have
positive weight, inverse-squared-weight effective count 19.73, maximum weight
0.1217, broad-component weight zero. Sparse LP weights alone are not posterior
collapse and do not establish density non-identifiability. These are not the
weights that a blind classifier/population fit would learn.

## Remaining population-model risk

SFH discrepancies remain substantial. The last two contrasts change from
0.621/0.898 to 0.659/0.958 W1/IQR. This does not prove the cause of a future
photometric failure, but it does not validate the reference SFH conditional.
Weak individual SFH constraints or genuine zero star formation are not errors.
The population distribution of those variables still affects fluxes and the
selection probability, and cannot be dismissed for parent inference.

## Recommended next step, not implemented or submitted in this review

Do not widen thresholds silently, regenerate the target, rerun the same LP,
increase the number of components blindly, or retrain the final posterior.
Use two bounded checks in parallel, with no neural training:

1. CPU: reconstruct the saved analytic CDF residuals using fixed weights, record
   the worst directions/thresholds and signed residuals, and quantify validation
   sampling uncertainty. Keep the observed FAIL and existing thresholds; do not
   tune weights on validation. Reuse the saved basis, weights and cached features.
2. Short forward predictive control: sample a small declared cohort (e.g. 8192
   parents) from the truth-assisted capacity fit, keep all 15D coordinates,
   use the exact saved decoder/noise/selection, and compare parent alpha and
   selected photometry with the coherent validation catalogue. Include both
   statistical uncertainty and bright-tail checks. These simulations are only
   a diagnostic and may not supply truth-fitted weights to population training.

The second check asks whether a physically adequate parent approximation also
produces adequate observables despite the unresolved SFH distribution. It tests
a scientific obstruction that physical marginal plots alone cannot exclude.
Its small sample cannot certify extremely rare tails or population recovery.

If those checks justify proceeding, the next substantial stage is the existing
new-reference-bank -> classifier -> selection-corrected parent benchmark. If the
CDF gate remains FAIL, proceeding as an exploratory benchmark requires an explicit
documented decision and launcher change, not editing its saved receipt or calling
it qualified. If observables already fail under the truth-assisted fit, address
the joint/SFH reference model before spending on that classifier.

Only after parent/selected/observable closure is convincing: fresh simulations
from the frozen learned parent, supervised 15D posterior training, independent
coverage/PIT/bias/width evaluation. An untouched final evaluation cohort is still
needed because development truth has influenced the reference design.

## Look here

Under `outputs/forward_population_results/avi_reference_to_parent_20260927_173822/`:

- `qualification/physical_objective.png`: blue validation truth, orange old fit,
  green physical objective. Notice metallicity and dust Av improvement, reduced
  redshift upper tail, residual sharp features and off-axis probability labels.
- `qualification/marginals.csv`: all 15 parameters, including unresolved SFH.
- `qualification/FINAL.json`: exact gates, empirical comparator and limits.
- `qualification/solver.json`: successful optimization and active TRAIN constraint.

No production prior was changed. No job was submitted in this local analysis.
