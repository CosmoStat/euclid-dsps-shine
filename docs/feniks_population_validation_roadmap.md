# Population validation roadmap, 2026-09-25

## Scope and evidence

Goal: learn a selection-corrected parent population and calibrated individual
15D posteriors. Keep the ten SFH coordinates active and marginalized through
simulation. Population learning has no input from approximate posterior draws.
Broad SFH posteriors are acceptable; a wrong assumed SFH distribution can still
bias inferred physical population coordinates through its photometric effects.

The current audit is a synthetic population experiment with a known parent in
the component family. Success here would not establish recovery for every
parent shape, for the reference FENIKS catalogue, or for real sky observations.

Verified source artifacts:

- `outputs/forward_population_results/avi_population_low_rank_audit_20260924_190751/`
  contains the complete noiseless result and incomplete noisy bootstrap.
- Its `noiseless_photometry/runtime/feature_stats.json` specifies six LSST bands,
  Euclid VIS plus three NISP bands, and eight Roman bands: 18 bands in total.
- Its manifest uses 4096 metric draws; `population_metrics` uses 64 projected
  directions after division by reference marginal IQRs.
- `_common_draws` reuses component-selection uniforms and Gaussian variates
  across mixture weights. Identical weights therefore produce identical samples
  and zero distance. An independent-sample null would not reproduce this metric.
- The noiseless bootstrap median is 0.01888 against a configured 0.015 gate;
  noisy heldout-admissible candidates have parent SW 0.03298--0.03542 against
  a 0.03056 closure gate. No noisy bootstrap table survived the timeout.
- The earlier ratio-ladder observation audit reports a decoder residual p95 of
  about 1.13 reported sigma in Roman F087. This is a discrepancy between numerical
  photometry and the reference catalogue, not proof of irreducible astrophysical
  model error. Its noise contract passes separately.

## Interpretation correction

The configured gates fail. This does NOT yet prove that the data fundamentally
cannot identify the parent or that more filters are required. The current
metric combines finite-catalogue variation, approximate classifier effects and
finite-draw distance estimation. Its bootstrap interval only measures resampling
of the saved replicates, conditional on the classifier and metric setup.

Likewise, stable or unstable component coefficients are not equivalent to stable
or unstable physical densities: overlapping components can exchange mass with
little observable or physical consequence. Assess the joint physical density,
redshift/mass/metallicity marginals, correlations and their uncertainty.

A parameter-space oracle observes more information than any photometric method.
If it succeeds while photometric inversion fails, either genuine information
loss or classifier error may explain the gap. Oracle success alone does not
identify classifier error. Oracle failure means the chosen estimator has not
demonstrated the desired precision, not a general impossibility theorem.

## Ordered decisions

### 1. Calibrate the evaluation and sampling reference

Reuse the saved component family, target catalogue, selection efficiencies and
exact latent ratios. Use the same selected-object identities and bootstrap
counts as the photometric fit. Freeze model choices before this comparison.

- Measure exact-ratio parent closure and bootstrap stability on matched samples.
- Separate variation from increasing the number of observed galaxies from
  variation due only to increasing metric draws (4096, 16384, 65536).
- Fix projection directions and physical scaling across comparisons; quantify
  seed variation with the existing common-random-number coupling. Include the
  equal-weight zero-distance invariant. Varying metric draws needs no DSPS.
- Report physical marginals and joint density uncertainty, not only one SW.
- Begin with eight paired replicates, saving every replicate. Extend only when
  the uncertainty on the resulting decision requires it. Eight is a diagnostic
  first tranche, not a publication-level bootstrap certification.
- Measure first-refit timing and cap wall time before submission. This proposal
  does not promise that 32 solves fit in a short allocation.

Deliverable: one comparison of numerical metric variation, oracle sampling
variation and photometric variation. Retain the original 0.015 gate in reports;
any revised scientific accuracy target must be justified independently and then
tested on an untouched confirmation sample.

### 2. Resolve the observation-model discrepancy

For identical latent objects, compare catalogue noiseless fluxes and decoded
fluxes with explicit filter curves, units, AB normalization, redshift, SED assets,
integration settings and latent transforms. Plot residuals in flux and reported
sigma by band, redshift and S/N. A small relative error can exceed sigma for a
bright object. Numerical and statistical model checks are different tests.

Existing decoder-quadrature work is documented separately in
`feniks_full_decoder_qualification_runbook.md`; its forward/gradient checks are
not evidence that catalogue compatibility has already passed. Preserve bank
provenance and regenerate only photometry invalidated by a verified correction.

### 3. Identify which physical ambiguities matter

Use the frozen photometric likelihood approximation to find poorly constrained
population directions, and express them as changes in p(z), p(M), p(Z) and their
joint distributions. Check these competing populations with independent forward
predictive data, because a learned likelihood Hessian alone cannot establish
intrinsic identifiability. All fifteen latent coordinates must remain variable.

If differences exceed the oracle sampling reference, distinguish classifier
approximation from indistinguishable photometric distributions before changing
population capacity. Good average classification NLL or normalization of ratio
moments alone is not enough. Broad uncertainty is appropriate where the data
leave genuinely different parent densities compatible.

### 4. Test added bands only against an explicit ambiguity

The current 18 measurements are not 18 independent physical constraints.
Overlapping bands can improve precision if noise is independent but may provide
little new spectral information. Candidate additions should sample spectral
features or wavelength ranges that distinguish the competing populations, with
realistic depth, covariance and missingness. Medium bands for redshift are a
motivated candidate, not a verified solution for this run.

Compare the original 18-band baseline with one justified augmented set, using
the same latent objects, original-band noise realizations, r-band selection,
splits and comparable training/convergence checks. Retrain the classifiers for
their respective inputs. Evaluate gain on independent test data and require
that it exceed metric/sampling variation. Keep r<29 selection fixed unless a
change in the survey selection is explicitly part of the experiment.

Additional bands must be measured for the target catalogue to help its actual
inference. Generating them only in simulations evaluates a future observing
scenario. Broad-band photometry does not guarantee sharp stellar metallicities;
do not interpret a narrow prior-driven result as new data information.

Scientific context: [Ilbert et al. (2009)](https://arxiv.org/abs/0809.2101)
demonstrate a 30-band photo-z pipeline with broad/intermediate/narrow bands;
its gains also depend on template and emission-line modeling. Those results
do not predict accuracy at r<29 for Feniks.
[Conroy (2013)](https://arxiv.org/abs/1301.7095) reviews SED constraints and their
degeneracies across stellar mass, metallicity, dust and SFH.

### 5. One end-to-end confirmation

After the preceding evidence supports a fixed observation model and population
estimator, fit the parent, freeze it, and train a supervised 15D posterior from
simulation truth. Validate parent marginals/joints, observable predictions,
68/95 percent coverage, ranks, bias and widths on independent simulations, with
redshift/magnitude/S/N strata. Test parents outside the exact fitted component
family and plausible SFH conditional changes before broad scientific claims.

Report uncertainty in the recovered parent. Individual intervals conditional on
a single fitted parent do not automatically include population uncertainty;
quantify this effect before claiming fully calibrated population-marginalized
posteriors. SFH recovery need not be sharp to pass, but physical inference must
remain calibrated after nuisance marginalization.

## Immediate action

Steps 1 and 2 now have a bounded parallel implementation. See
[the launch runbook](feniks_population_precision_runbook.md). One GPU job caches
ratios and full fits; CPU metric and paired-bootstrap arrays reuse the cache.
An independent GPU job screens decoder integration against catalogue fluxes.
Every bootstrap cell and decoder batch is persisted for selective recovery.

| Step | Implementation | Scientific status |
|---|---|---|
| 1: metric precision and matched oracle reference | Implemented, local smoke tests | Awaiting cluster artifacts |
| 2: observation model | 128-object integration screen implemented | Full qualification still pending |
| 3: physical ambiguity | Analysis plan | Not established by classifier curvature alone |
| 4: additional filters | What-if inventory only | No experiment implemented |
| 5: final parent and 15D posterior | Existing pipeline available | Awaiting scientific validation |

Each new run exports `ROADMAP_STATUS.md` and `ROADMAP_STATUS.json` with execution
states; the report keeps production readiness false. Update this table from
persisted evidence after the run, not from job completion alone. The separate
[Pop-COSMOS band assessment](feniks_popcosmos_bands_what_if.md) records available
curves and what a matched comparison would require.
