# Diffsky Amortized Collapse Fix Plan

## Objective

Build a validation ladder for the unsupervised encoder + DSPS decoder +
normalizing-flow prior path. The science target remains photometry-only
inference with free redshift and a learned degenerate population prior. Truth
columns in the Diffsky closure dataset are diagnostics only; they must not be
used as a training loss or fixed-redshift constraint.

## Current Failure Mode

The latest smoke run is technically healthy but scientifically collapsed:

- inference is sharded/finalized correctly;
- MAP-Adam runs to completion;
- the learned posterior/prior and MAP solution prefer a high-redshift basin;
- closure truth redshifts in the smoke sample are low, so photo-z coverage is
  essentially zero.

The root cause is not yet identified. The next implementation must separate:

- decoder/forward-model mismatch;
- likelihood landscape degeneracy;
- encoder initialization/posterior collapse;
- RealNVP learning from a wrong aggregate posterior;
- excessive latent dimensionality.

## Generalized Latent Initialization Risk

Any bounded physical parameter represented by an unconstrained latent and a
sigmoid transform has an implicit initial population set by the bounds and the
encoder/prior initialization. If bounds are broad or badly centered relative to
the target population, optimization can start in an artificial basin. A jointly
trained NF can then learn and reinforce that basin. This applies to redshift,
mass, dust, metallicity, and SFH parameters, not only `z_obs`.

## Phase A: Extended Truth Diagnostics And Plots

Deliverables:

- extend truth snapshots beyond `redshift_true`/`logsm_true` to include
  available Diffsky/Diffstar/Diffmah/dust/burst columns;
- write summary tables for posterior, MAP, and learned-prior population
  comparisons;
- generate plots:
  - truth vs posterior redshift;
  - truth vs MAP redshift;
  - truth vs posterior mass;
  - truth vs MAP mass;
  - proxy dust plots where available: `tau2` vs `dust_av / 1.086` and
    `dust_index_n` vs `dust_delta`;
  - population overlays for truth/prior/posterior/MAP;
  - bias vs truth redshift/mass;
  - compact corner-style population plot for the comparable/proxy parameters.

Gate:

- these diagnostics must exist for every inference/MAP run before interpreting
  photo-z metrics.

## Phase B: Validate The Actual Amortized Decoder

Problem:

- existing true-parameter closure validates `diffsky_basic`;
- amortized runs currently use `popcosmos_bins`;
- therefore we have not proved that the actual amortized decoder can represent
  the Diffsky photometry near truth/proxy parameters.

Deliverables:

- add `popcosmos_proxy_truth_closure`;
- map available closure truth into the compact PopCosmos latent:
  - `redshift_true -> z_obs`;
  - `logsm_true -> log10_stellar_mass`;
  - `logssfr_true/logsfr_true -> dlog10_sfr_*` proxy;
  - `dust_av -> tau2`;
  - `dust_delta -> dust_index_n`;
  - fixed config values for the remaining nuisance dimensions;
- write closure tables and plots by band, redshift, mass, and proxy dust.

Gate:

- if this closure has large systematic residuals or impossible colors, do not
  train the NF; fix decoder/parameterization first.

## Phase C: MAP-Adam Likelihood-Landscape Diagnostics

Role:

- MAP multistart is initially a diagnostic, not the final model.
- It answers whether the DSPS decoder + photometric likelihood have a
  low-redshift minimum reachable without the encoder.

Deliverables:

- add MAP start modes independent of the encoder:
  - `encoder`;
  - `prior`;
  - `z_grid`;
  - `lowz_grid`;
  - `latin_hypercube`;
  - `mixed`;
- support `prior_weight=0.0` explicitly;
- write per-start-family diagnostics:
  - `map_estimates_by_start.parquet`;
  - `map_best_by_start_family.csv`;
  - `map_start_family_summary.csv`;
  - `chi2_vs_z_start.png`;
  - MAP truth/proxy plots from Phase A.

Debug runs:

- `map_no_prior_zgrid_1k`: `prior_weight=0`, starts independent of encoder;
- `map_with_prior_zgrid_1k`: same starts with a small learned-prior weight.

Gate:

- if likelihood-only z-grid cannot recover low-z, the issue is decoder,
  likelihood, or parameterization;
- if likelihood-only works but prior-weighted MAP fails, the prior is too
  restrictive or already wrong.

## Phase D: Low-Dimensional Controlled Tests

Deliverables:

- configs:
  - `popcosmos_lowdim_z_mass_dust`;
  - `popcosmos_lowdim_lowz_bounds`;
  - `popcosmos_lowdim_fullz_bounds`;
- Slurm entry points must accept arbitrary config/run names so multiple
  diagnostics can run in parallel.

Purpose:

- determine whether the 9D compact latent is too degenerate;
- distinguish bad bounds from bad physics and excessive nuisance freedom.

Gate:

- low-dimensional likelihood-only MAP should recover redshift before running
  low-dimensional NF training.

## Phase E: NF Stabilization Only After Gates

Process:

1. decoder proxy closure passes;
2. likelihood-only MAP z-grid passes;
3. encoder-only or prior-frozen training visits plausible redshift modes;
4. train NF on the aggregate posterior;
5. fine-tune encoder + NF jointly;
6. use MAP under learned prior as the final optimization-based estimator.

Deliverables:

- automatic collapse gate after training/inference:
  - posterior/prior redshift quantiles;
  - posterior width and entropy/log-std summaries;
  - residual/chi2 thresholds;
  - optional truth-aware closure gate when truth exists;
- stabilized configs with longer prior freeze, stronger entropy floor, and
  explicit latent initialization controls.

Gate:

- do not scale a RealNVP run if prior/posterior population overlays or
  likelihood-only MAP diagnostics indicate a high-redshift collapse.

## Expected Execution Order After Implementation

1. `popcosmos_proxy_truth_closure_1k`
2. `map_no_prior_zgrid_1k`
3. `map_with_prior_zgrid_1k`
4. `lowdim_map_no_prior_zgrid_1k`
5. `lowdim_unsup_nf_5k_e10`
6. `full_unsup_nf_10k_e15`

## Implementation Status

- Phase A: implemented.
- Phase B: implemented.
- Phase C: implemented.
- Phase D: implemented.
- Phase E: implemented.
- Validation status: compileall, Ruff, targeted pytest, Slurm syntax checks,
  config-load smoke, CLI-help smoke, and a two-object local proxy-closure smoke
  passed on 2026-06-17.
