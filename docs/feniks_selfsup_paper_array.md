# FENIKS final self-supervised paper array

This matrix tests whether the selected `selfsup_rws_k8_t2` result is stable and
whether its gain comes from RWS or merely from the prior. Truth columns are
never used by an optimizer; they are read only for held-out closure metrics.

## Controlled contract

All four candidates use:

- the same 15-parameter spline SFH latent schema and immutable
  `mixed_log_shifted_asinh` normalization;
- the differentiable DSPS forward model and the same 18-band catalog;
- a four-layer conditional RealNVP encoder in independent `latent_x` space;
- catalog `flux_err` as the scale of a Student-t likelihood with two degrees
  of freedom, without an added floor or jitter;
- 120 encoder epochs, the same validation cadence, 5,000 held-out objects, 128
  posterior samples per object, and 16,384 prior samples.

The candidates are:

1. `rws_k8_t2_seed2`: learned RealNVP population prior and RWS K=8.
2. `rws_k8_t2_seed3`: the same model with an independent training seed.
3. `fixed_prior_rws_k8_t2`: RWS K=8 with the serialized spline15d reference
   prior frozen.
4. `avi_joint_t2`: stochastic-ELBO AVI with reverse KL and a jointly learned
   RealNVP prior.

The two earlier and two new RWS seeds support a post-run seed-stability
analysis. The array itself does not overwrite or depend on the earlier run.

## Jacobian Lens contract

The Lens samples the complete conditional posterior after the flow. Its
decoder point is the empirical posterior mean and its direction variances use
the full empirical 15D covariance, including off-diagonal terms. The
autoencoder Jacobian remains deterministic and differentiable by pushing the
encoder base mean through the configured flow.

Each candidate uses four Lens shards over 512 stratified held-out objects. The
finalizer refuses to create `JLENS_DONE` unless the spectrum, effective-rank,
and autoencoder amplification artifacts exist.

## Required outputs

Each candidate writes training curves, inference metrics, posterior
calibration tables, full corner plots, photometric posterior-predictive plots,
prior marginal/correlation plots, and finalized Lens tables and figures.

`comparison/` contains:

- `experiment_metrics.csv`;
- `coverage_by_parameter.csv` and `coverage_by_parameter.png`;
- `speed_vs_coverage.png`;
- `scientific_scoreboard.png`;
- `jacobian_spectrum_comparison.png`;
- `photoz_by_redshift.png`;
- `prior_marginal_error_by_parameter.png`;
- `README.md` linking the per-run photometry fits and corner plots.

The production array is released only after the complete four-candidate smoke
chain has trained, inferred, generated Lens shards, finalized them, and built
the comparison report.
