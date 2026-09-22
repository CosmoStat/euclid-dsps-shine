# FENIKS r<29 failure-mode audit

This read-only campaign localizes the failure observed after the completed
forward-population and posterior-diagnostic runs. It does not train another
classifier, parent, or posterior. It reuses immutable checkpoints and banks,
and fails if any hashed input changes.

## Design

The five array cells run independently on one H100 each.

1. `decoder_observation` decodes 2,048 stratified blind-parent truths and
   compares the resulting noiseless flux with the catalogue `flux_true`. It
   separately recomputes the m5 uncertainty, standardizes the saved noise, and
   recomputes the exact noisy `lsst_r < 29` selector. This separates a decoder
   or unit mismatch from a population-learning failure.
2. `nuisance_sensitivity` holds the five physical latent coordinates fixed and
   replaces the ten nuisance latent coordinates with eight independent draws
   from the implemented reference conditional `N(0,I)`. It reports flux shifts
   in observational-sigma units and the resulting change in selection
   probability over 32 noise realizations. The decision uses aggregate rates,
   globally and in redshift bins; paired per-object changes are sensitivity
   diagnostics only. This is an oracle diagnostic using truth, not a proposed
   production estimator.
3. `posterior_original` evaluates the original supervised 15D flow on 8,192
   independent selected simulations and the complete available held-out
   selected catalogue cohort (up to 8,192 objects).
4. `posterior_continued` repeats exactly the same evaluation with the continued
   posterior checkpoint. Coverage is conditioned on true redshift, observed r
   magnitude, r-band S/N, and distance to the selection boundary. The derived
   metrics use 256 joint draws per object. Joint draws for 1,024 objects per
   cohort are retained for later distribution-level inspection.
5. `population_identifiability` bootstraps the fixed observed classifier-ratio
   matrix 24 times, resolves the simplex-constrained selected mixture each
   time, applies the explicit `v/alpha` parent correction, and evaluates the
   parent in physical coordinates. It also computes the tangent Hessian
   spectrum and converts the existing known-mixture closure from latent to
   physical coordinates before calculating Wasserstein distances.

The dependent report runs only after all five cells succeed. It writes a
machine-readable `DECISION.json`; no failed gate is hidden by an average over
parameters.

## Scientific contracts

- Parent truth comparisons use the catalogue `population_weight`; the target
  is not silently changed to the empirical unweighted sample.
- Basis samples are transformed from latent x to physical theta before being
  compared with physical truth.
- The five physical coordinates are selected by name, not by column position.
- The exact catalogue selection identity is read from
  `selection_identities.parquet` and independently recomputed from noisy flux.
- The individual posterior remains joint and 15D. No posterior median is used
  as a replacement for the distribution.
- Population fitting is not rerun from q samples and no q draw becomes a
  population or posterior training target.

## Gates

Thresholds are frozen in
`configs/experiments/feniks_failure_modes_r29.yaml` before submission. They
cover decoder residuals, uncertainty/noise consistency, selector identity,
nuisance-conditional sensitivity, per-physical-parameter 68/95% coverage,
parent selection normalization, known-mixture density closure, and bootstrap
stability. The nuisance test is not a demand for precise individual SFH
reconstruction. Broad, prior-like SFH posteriors are expected for an
underconstrained inverse problem. It is nevertheless a parent-prior gate: it
tests whether the fixed standard-normal nuisance conditional used by every
population component changes the `r<29` selection probability relative to the
true conditional. A failure can bias `alpha_j` and therefore the reconstructed
parent weights even when the five physical coordinates are represented well.

The original posterior is a diagnostic baseline and is not part of the
production-readiness conjunction. The continued posterior is. A likely useful
decision pattern is:

- decoder or noise gate fails: repair the observation contract before any new
  training;
- known-mixture density closure fails: improve ratio estimation or component
  identifiability before changing the posterior;
- simulated posterior passes but catalogue posterior fails: investigate the
  simulation-to-catalogue observation law, support, and conditional nuisance
  model;
- parent alpha fails while known-mixture closure passes: the reference family
  or its selection efficiencies do not reproduce the true parent catalogue;
- all structural and continued-posterior gates pass: prepare one corrected
  production run, not another exploratory sweep.

Passing this audit is necessary, not sufficient, for a scientific parent-prior
claim. Truth enters only these post-hoc closure tests.

## Outputs

- `decoder_observation/band_summary.csv`
- `decoder_observation/per_object_band.parquet`
- `nuisance_sensitivity/band_sensitivity.csv`
- `nuisance_sensitivity/per_object_selection.parquet`
- `posterior_{original,continued}/marginal_summary.csv`
- `posterior_{original,continued}/conditional_calibration.csv`
- `posterior_{original,continued}/*_dense_posteriors.npz`
- `population_identifiability/bootstrap_metrics.csv`
- `population_identifiability/bootstrap_weights.npz`
- `population_identifiability/known_mixture_density_closure.csv`
- `failure_modes_summary.png`
- `DECISION.json`, `REPORT.md`, and `FINAL.json`

The dense posterior NPZ files are the largest audit products and may be
excluded from an initial rsync while retaining every scalar/table diagnostic.
