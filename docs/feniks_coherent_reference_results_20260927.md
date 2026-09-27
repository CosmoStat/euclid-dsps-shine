# Reference capacity audit: 2026-09-27

Run: `avi_coherent_reference_audit_20260927_151331`, job 247856 (user-reported
COMPLETED, 0:0). Thirteen present receipt artifacts verify against hashes; only
`capacity/cdf_features.npz` was excluded from download. Source manifest, parent
and reference-receipt hashes match the previously inspected coherent inference
run. The current CDF helper matches the executed helper hash. Stderr is empty.
No new physical simulation or neural training occurred in this audit.

## Decisive result: a restricted parent family

The optimal minimax discrepancy on 525 analytic component-CDF features is
**0.123101** on TRAIN, despite access to true parent parameters. The evaluated
discrepancy is 0.126221 on TEST, versus 0.020150 for validation versus test.
The photometry-fitted weights give 0.139655 on test.

This establishes a representational restriction: within the implemented fixed
64-component family, no choice of simplex weights can match all those TRAIN
features to better than about 12.3 percentage points. This statement is about
the empirical TRAIN target and this finite feature set, not an exact bound
against the infinite true population or an optimal Wasserstein certificate.
It does not depend on classifier errors, q approximation, or the population
regularization penalty. More classifier epochs alone cannot remove it.

Physical SW for the truth-CDF optimum is 0.1540, versus 0.1208 for the original
photometric fit and 0.01407 for validation/test. This does NOT mean the LP failed:
it minimizes the worst CDF feature, not SW or average CDF error. Average CDF
error can worsen while its maximum improves. Do not promote the truth-fit
weights as a better production estimate.

The plot `capacity/capacity_physical.png` makes the restriction visible:
redshift remains too flat, the low-mass shoulder remains broad, and the sharp
dust-slope peak is not reproduced even by the truth-fitted weights. SFH SW
remains 0.618 for that fit; this does not invalidate broad individual SFH
posteriors, but does leave the population SFH law unqualified.

## Observable mismatch: dominated by a bright tail, not only a tail

Using all 70589 reserved selected rows with exact saved-bank weights (importance
effective size 2993), the r-band metrics are:

- Raw scaled W1: 236.59.
- Diagnostic W1 after applying the same upper cap at target q99.9 to both
  distributions: 8.77, a 96.29% reduction.
- Predicted probability above target q99.9: 0.9088%, versus target 0.1072%.
- Asinh-scaled flux W1, retaining all values: 0.3454.

Across the 18 bands, diagnostic upper capping reduces raw scaled W1 by
95.4--96.5%. Thus the enormous raw-flux distance is predominantly sensitive to
the upper tail. The residual and the CDF curves still disagree; neither result
authorizes clipping or discarding bright galaxies. This is not a magnitude error.

Component 19 contributes 54.0% of the predicted r-band upper excess, component
42 contributes 16.1%, then components 2, 26 and 60 contribute about 10.5%, 10.4%
and 9.0%. This locates the excess in the mixture, not its physical cause. The
download lacks individual tail-row theta, noiseless flux and anchor identities;
it cannot distinguish low redshift, stellar mass, SFH and kernel perturbation.

Reporting caveat: the saved column `truth_flux_iqr` actually contains
`max(actual_IQR, 1e-30)`. It is a numerical scale floor for u/g/r/F062 here, not
their literal IQR. That floor reduces, rather than creates, their scaled raw
W1. Preserve old values; future reports should distinguish actual IQR and scale.

## Additional source-level tail risk

`coherent_coordinates.py` applies asinh to log10 stellar mass, and the current
reference adds Gaussian noise in that coordinate. Its inverse is:

```
m = log10(Mstar) = location + width * sinh(center + scale * x)
```

For any Gaussian kernel with nonzero variance in x, the ideal unbounded model's
expectation of `10**m` diverges: the positive tail of the log integrand grows
like a positive exponential in x, faster than the negative Gaussian quadratic.
The density remains normalized; normalization does not guarantee finite mass
or predictive moments. `model.py::normalize_sfh_to_stellar_mass_jax` uses
`10**log10_stellar_mass` without a physical upper cutoff.

This is a mathematical design risk, NOT proof that the measured finite-bank
outliers arise from this asymptotic tail. Moderate anchors, small redshift or
the shared broad gate term can dominate the current finite sample. Numerical
overflow/clamping is not a principled physical-tail model.

A conservative next reference uses an affine coordinate for the already-log
mass, giving Gaussian log-mass kernels with finite linear-mass moments, or a
separately justified tail model. Keep existing coordinates/checkpoints immutable.

## Next-run decision

Do NOT continue the current classifier, final posterior, low-rank/bootstrap
sweeps, or rerun the unchanged pipeline. Do not generate a new target catalogue
or add bands as a remedy for the demonstrated representational restriction.

The next justified experiment is ONE bounded reference-redesign qualification,
before any new large bank:

1. Retain independent native 15D anchors and physical/SFH associations. Use
   affine log-mass, avoiding the additional sinh amplification. Keep all ten
   SFH coordinates stochastic; no fixed nuisance values or injected truth prior.
2. Compare a small controlled set of reference definitions to isolate physical
   kernel smoothing and broad gates. Separate the broad support component from
   the local components instead of forcing it into every component. Use local
   physical resolution rather than assuming 64 broad fixed gates are adequate.
   Define component geometry from the independent reference, not test truth.
3. Evaluate capacity analytically on TRAIN and validation; include per-direction
   residuals and a same-sample empirical comparator. Do not choose a candidate
   on its published test SW or call a lower CDF objective a guaranteed SW gain.
4. In the same qualification, inspect the most influential saved flux-tail rows
   and their reproducible original anchors. A bounded paired forward replay of
   those rows can distinguish already-bright anchors from perturbation-created
   outliers; it is not a million-object resimulation or an MCMC diagnostic.

The ablation/replay launcher is NOT implemented by the current audit. The
old audit only diagnoses the original frozen family; relaunching it unchanged
cannot test a redesign. These are next implementation specifications, not
submitted jobs or a promise of a passing alternative.

Only after the new family passes a declared capacity/tail qualification should
we submit a new reference-bank -> classifier/parent -> supervised posterior
chain. Reuse the coherent target dataset and retain the existing oracle as
control. Changing components/coordinates invalidates their old classifier labels
and selection efficiencies: do not silently relabel old banks. Explicitly
validated importance reuse is optional, not assumed safe. Fresh/diversified
learned-parent pairs address the previously observed finite-bank NPE overfit.

For publication, reference design may use development TRAIN/validation truth,
but blind population fitting must remain photometry-only and the final claim
requires fresh independent test evidence after design choices are frozen.

## Roadmap consequence

The capacity diagnostic is COMPLETE and it rules out the adequacy of this fixed
family on the tested features. Parent recovery is still FAIL; it has not improved
because this run changed no model. The advance is a decisive localization of a
blocking assumption. Numerical normalization/selection tests and the previous
physical-marginal posterior control remain valid in their stated scopes.
