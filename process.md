# Process: Euclid FS2 to DSPS Inference

This document explains the current pipeline: what data enters, how DSPS is called, what is inferred, what is fixed, what is arbitrary, and what outputs are produced.

## High-Level Objective

The goal is to test whether native `dsps` can reproduce simulated Euclid photometry for galaxies in the local Euclid FS2 catalog.

For each galaxy, the pipeline:

1. Reads observed or simulated catalog fluxes.
2. Selects the redshift value to pass into DSPS.
3. Builds a simple physical galaxy model.
4. Runs DSPS to produce a rest-frame SED.
5. Applies dust attenuation.
6. Projects the SED through Euclid filters to model photometry.
7. Compares model photometry to catalog photometry.
8. Optionally optimizes physical parameters to minimize the mismatch.

The default catalog is:

```text
Data/Euclid FS2 LC galaxy catalog_phz1.parquet
```

The default config is:

```text
configs/fs2_phz1.yaml
```

## Main Commands

```bash
euclid-dsps --config configs/fs2_phz1.yaml eda --out outputs/eda_phz1
```

Runs catalog inspection: schema, missing values, flux distributions, colors, and redshift diagnostics.

```bash
euclid-dsps --config configs/fs2_phz1.yaml run-one --out outputs/runs/one
```

Runs one forward DSPS model for one selected galaxy. No fitting.

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-one --out outputs/runs/fit_one
```

Fits one galaxy using JAX Adam.

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-one --bayesian --out outputs/runs/sample_one
```

Samples one galaxy posterior using NumPyro NUTS/HMC.

```bash
euclid-dsps --config configs/fs2_phz1.yaml run-batch --limit 1024 --batch-size 64 --out outputs/runs/run_batch
```

Runs forward DSPS predictions for many galaxies in JAX-vmapped chunks.

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-batch --limit 1024 --batch-size 64 --out outputs/runs/fit_batch
```

Fits many galaxies independently, one JAX-vmapped Adam optimizer per chunk.

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-batch --bayesian --limit 5 --batch-size 1 --out outputs/runs/sample_batch
```

Samples independent posteriors for a small batch. This is intentionally for small validation sets, not full-catalog production.

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-population --limit 1024 --batch-size 64 --out outputs/runs/population
```

Runs a chunked hierarchical MAP approximation: per-galaxy parameters plus shared population parameters.

Use `--all` instead of `--limit` to process the full parquet catalog.

## Data Contract

The default photometry columns are:

```text
euclid_vis
euclid_nisp_y
euclid_nisp_j
euclid_nisp_h
```

They are interpreted as:

```text
Fnu in erg / s / cm^2 / Hz
```

These fluxes are converted to AB magnitudes with:

```text
m_AB = -2.5 log10(Fnu) - 48.6
```

The default redshift passed into DSPS is read from the catalog column:

```text
z_phz
```

The catalog also contains:

```text
z_true
```

That column is used only for diagnostics. It is not used as the DSPS redshift unless the config is changed.

Important: DSPS does not estimate or infer redshift in the default scientific configuration. The pipeline chooses a redshift before calling DSPS. By default it takes `z_phz`, clips it to the configured range, and passes it to DSPS as fixed `z_obs`.

In code, this happens in `parameters_for_row` and `resolve_redshift`:

```text
catalog row -> z_phz -> z_obs -> DSPS
```

So "redshift handling" currently means:

- read `z_phz` from the Euclid catalog
- check that it is finite
- clip it to `[0.0001, 6.0]`
- use it in DSPS cosmology and observed-frame photometry
- compare it to `z_true` only after the run

If `z_phz` is missing or invalid, the config fallback is used:

```yaml
redshift:
  fixed_value: 0.5
```

That fallback is not a measurement. It is only a safety value.

## Filters

All Euclid bands (VIS/Y/J/H) now use the real Euclid throughput files supplied in `filters/` as ASCII/DAT files:

- `filters/Euclid_VIS.vis.dat`
- `filters/Euclid_NISP.Y.dat`
- `filters/Euclid_NISP.J.dat`
- `filters/Euclid_NISP.H.dat`

The code loads these directly from the ASCII format. Wavelengths are expected in Angstrom. Throughput values are clipped to [0, 1].

## Forward Model

The core model lives in:

```text
euclid_dsps/model.py
```

The JAX forward model is:

```text
parameters -> SFH -> DSPS rest SED -> dust attenuation -> observed photometry
```

The function `run_dsps_model_jax` does the differentiable work.

For a galaxy at redshift `z_obs`, the code computes the cosmic age:

```text
t_obs = age_at_z(z_obs)
```

Then it builds a time grid:

```text
gal_t_table = linspace(0.05, t_obs, n_sfh_bins)
```

Default:

```text
n_sfh_bins = 96
```

The star formation history is currently a simple lognormal-like curve:

```text
SFR(t) = 10^log10_sfr * exp(-0.5 * ((log(t) - log(t_peak)) / tau)^2)
```

This is a modeling choice. It is not taken from the Euclid catalog. The fitted `log10_sfr` is the amplitude of this curve. Reports compute the SFR compared to catalog truth by evaluating the fitted SFH at observation time:

```text
t_obs = age_at_z(z_obs)
fit_log10_sfr_at_obs = log10(SFR(t_obs))
```

That is why the truth comparison uses `log10_sfr_at_obs`, not `log10_sfr`.

DSPS then receives:

- `gal_t_table`
- `gal_sfr_table`
- `log10_metallicity`
- `metallicity_scatter`
- SSP metallicity grid
- SSP age grid
- SSP flux table
- `t_obs`

The DSPS call is:

```text
calc_rest_sed_sfh_table_lognormal_mdf
```

That returns a rest-frame SED.

Dust attenuation is applied with the DSPS Salim+2018-style curve:

```text
sbl18_k_lambda
_frac_transmission_from_k_lambda
```

Finally, DSPS photometry is computed with:

```text
calc_obs_mag
```

for each configured filter.

## Model Parameters

The default fixed parameter values are:

```yaml
log10_sfr: 0.0
sfh_t_peak: 4.0
sfh_tau: 0.6
log10_metallicity: -2.0
metallicity_scatter: 0.2
dust_av: 0.2
dust_slope: -0.7
```

The redshift is not fixed to the default value when `z_phz` exists. For each row:

```text
z_obs = z_phz clipped to [0.0001, 6.0]
```

The fallback redshift is:

```text
fixed_value: 0.5
```

but this fallback is only used if the redshift column is missing or invalid.

## Free Parameters in the Fit

The default fitted parameters are:

```yaml
log10_sfr:
  bounds: [-4.0, 3.0]
dust_av:
  bounds: [0.0, 1.0]
log10_metallicity:
  bounds: [-4.2, -1.4]
```

`log10_sfr` is the amplitude of the simple lognormal SFH used internally by the DSPS wrapper. It is not directly compared to catalog `log_sfr_true`. Reports compute the derived current SFR at observation time:

```text
fit_log10_sfr_at_obs = log10(SFR(t_obs))
```

and compare that derived value to `log_sfr_true`.

Everything else stays fixed:

- `z_obs` is fixed from `z_phz`
- `sfh_t_peak` is fixed at `4.0`
- `sfh_tau` is fixed at `0.6`
- `metallicity_scatter` is fixed at `0.2`
- `dust_slope` is fixed at `-0.7`
- cosmology uses DSPS `DEFAULT_COSMOLOGY`

These fixed choices are mostly pragmatic. With only VIS/Y/J/H photometry, fitting too many parameters is underconstrained.

## Likelihood and Objective

The fit compares model AB magnitudes to catalog AB magnitudes.

A likelihood is a function that says how probable the observed data are if a given model parameter set were true.

Here, the data are the catalog magnitudes:

```text
observed_mag = [VIS, Y, J, H]
```

The model prediction is:

```text
model_mag(parameters) = DSPS(parameters, z_obs, filters)
```

The parameters are things like:

```text
log10_sfr, dust_av, log10_metallicity
```

The likelihood answers:

```text
If these parameters were true, how likely is it that we would observe these catalog magnitudes?
```

The current assumption is Gaussian measurement error in magnitude space. That means each observed magnitude is assumed to be drawn from a normal distribution centered on the DSPS model magnitude:

```text
observed_mag_band ~ Normal(model_mag_band, sigma_mag)
```

This is a modeling assumption. The current catalog workflow does not read per-object flux errors, so `sigma_mag` is fixed by config.

For each band:

```text
chi = (observed_mag - model_mag) / sigma_mag
```

Default:

```text
sigma_mag = 0.05
```

The single-galaxy objective is:

```text
chi2 = sum_bands chi^2
```

The Gaussian log-likelihood is:

```text
log L = -0.5 * chi2 + constant
```

Because the constant does not depend on the model parameters, maximizing this likelihood is equivalent to minimizing `chi2`.

So in practice:

```text
best parameters = parameters that minimize chi2
```

For one galaxy with four bands, the fit asks:

```text
Which log10_sfr, dust_av, and log10_metallicity make DSPS VIS/Y/J/H closest to catalog VIS/Y/J/H?
```

The current code optimizes chi2 or a MAP objective. It does not sample the full posterior.

This distinction matters:

- Optimization returns one best-fit point.
- Posterior sampling would return a distribution of possible parameter values.
- Current code gives best-fit values and residual diagnostics, not full uncertainty intervals.

## Independent Fitting

Independent fitting is used by:

```text
fit-one
fit-batch
```

For `fit-one`, the code wraps one galaxy into a batch of size one and uses the same vectorized fitting code as `fit-batch`.

For `fit-batch`, the parquet catalog is streamed in chunks:

```text
batch_size galaxies at a time
```

For each chunk:

1. Convert fluxes to AB magnitudes.
2. Build a base parameter matrix.
3. Build an initial free-parameter matrix.
4. Warm-start `log10_sfr` using the median offset between initial model magnitudes and observed magnitudes.
5. Transform bounded parameters to unconstrained variables with a sigmoid inverse.
6. Run JAX Adam for `maxiter` iterations.
7. Keep the best parameters found per galaxy.
8. Write fit results, model photometry, residuals, and plots.

The optimizer is JAX-native:

```text
jax.jit
jax.vmap
jax.value_and_grad
jax.lax.scan
```

The code runs on the active JAX device. In the current environment this is:

```text
gpu:0
```

## Bounded Parameter Transform

Adam optimizes unconstrained variables `y`.

The physical bounded parameter `theta` is recovered with:

```text
theta = lower + (upper - lower) * sigmoid(y)
```

This prevents Adam from hard-clipping directly on parameter bounds, which made optimization less stable.

## Population Fitting

Population fitting is used by:

```text
fit-population
```

This is a hierarchical MAP approximation, not full posterior sampling.

The motivation is that galaxies are not treated as completely unrelated objects. Instead, we assume their fitted physical parameters come from a shared population distribution.

For independent fitting, each galaxy has its own parameters:

```text
theta_1, theta_2, theta_3, ...
```

where:

```text
theta_i = [log10_sfr_i, dust_av_i, log10_metallicity_i]
```

Each galaxy is optimized separately. Galaxy 1 does not inform galaxy 2.

In hierarchical population fitting, we introduce population-level parameters:

```text
mu    = population mean of each fitted parameter
sigma = population scatter of each fitted parameter
```

Then each galaxy parameter vector is assumed to come from that population:

```text
theta_i ~ Normal(mu, sigma)
```

This creates two linked levels:

```text
population level: mu, sigma
galaxy level:     theta_i for each galaxy
data level:       observed VIS/Y/J/H for each galaxy
```

The model structure is:

```text
mu, sigma -> theta_i -> DSPS(theta_i, z_i) -> model photometry_i -> likelihood_i
```

This is "hierarchical" because galaxy parameters are below population parameters.

For each parquet chunk, it jointly optimizes:

- per-galaxy free parameters
- population mean `mu` for each free parameter
- population scatter `sigma` for each free parameter

The population prior is Gaussian:

```text
theta_i ~ Normal(mu, sigma)
```

This prior regularizes the galaxy fits. If one galaxy has weak or ambiguous photometry, its parameters are gently pulled toward the population distribution inferred from the other galaxies in the chunk.

The optimized loss is:

```text
0.5 * sum_galaxies chi2_i
+ prior_weight * sum_galaxies population_prior_i
+ weak hyper-prior on mu
```

In words:

- The first term rewards fitting each galaxy's photometry.
- The second term rewards galaxy parameters that look plausible under the shared population.
- The third term prevents the population mean from drifting to extreme values without evidence.

MAP means "maximum a posteriori". It optimizes:

```text
posterior ∝ likelihood * prior
```

Taking negative logs gives:

```text
negative log posterior = negative log likelihood + negative log prior + constant
```

That is why the population objective has both:

- a photometry chi2 term
- a population prior term

The code minimizes this negative log posterior approximation. The result is one best joint solution:

```text
best theta_i for each galaxy
best mu for the chunk
best sigma for the chunk
```

The population scatter is constrained with:

```text
sigma = softplus(raw_sigma) + sigma_floor
```

Default population settings:

```yaml
prior_weight: 1.0
sigma_floor: 0.03
hyper_mu_scale: 5.0
```

This gives a first paper-style population layer: individual galaxy parameters are no longer completely independent; they are regularized by a shared population distribution.

Important limitation: this is chunk-local. Each chunk gets its own population `mu` and `sigma`. It is not yet a global full-catalog hierarchical posterior.

## Bayesian HMC/NUTS Sampling

Bayesian sampling is enabled with:

```bash
fit-one --bayesian
fit-batch --bayesian
```

This uses NumPyro HMC/NUTS. Both methods use gradients of the DSPS photometry
log posterior. The important speed difference is that fixed-step HMC has a
known number of leapfrog steps per draw, while NUTS adaptively builds a tree and
can use many more DSPS forward/gradient evaluations.

For a fast one-galaxy debug posterior:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-one \
  --index 0 \
  --bayesian \
  --sampler hmc \
  --num-warmup 120 \
  --num-samples 400 \
  --num-chains 1 \
  --num-steps 8 \
  --target-accept-prob 0.8 \
  --no-progress \
  --out outputs/runs/phz1_mcmc_row_0_fast
```

For a more careful single-galaxy posterior:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-one \
  --index 0 \
  --bayesian \
  --sampler nuts \
  --num-warmup 300 \
  --num-samples 800 \
  --num-chains 1 \
  --max-tree-depth 6 \
  --target-accept-prob 0.8 \
  --no-progress \
  --out outputs/runs/phz1_mcmc_row_0_nuts
```

This writes:

- `posterior_samples.csv`: raw sampled free parameters
- `posterior_derived_samples.csv`: derived values, including `log10_sfr_at_obs`
- `posterior_comparable_samples.csv`: posterior columns with configured truth/proxy values
- `posterior_corner.png`: raw sampled-parameter corner
- `posterior_corner_with_truth.png`: comparable posterior corner with truth/proxy markers
- `posterior_truth_values.json`: catalog truth/proxy values used for overlays
- `posterior_predictive_photometry.png`: posterior predictive photometry check

By default, Bayesian mode does not start from a random prior draw. It first runs the existing Adam/MAP fit for the selected galaxy, then passes the fitted free parameters to NumPyro as constrained initial values:

```text
catalog row -> Adam/MAP best point -> NUTS/HMC posterior sampling
```

This is controlled by:

```yaml
sample:
  init_from_map: true
```

Disable it only for debugging:

```bash
fit-one --bayesian --no-map-init
```

MAP initialization does not change the posterior definition. It only gives HMC a better starting point, which usually reduces wasted warmup time and avoids bad initial states.

Runtime guidance:

- prefer `--sampler hmc --num-steps 6` or `--num-steps 8` for quick posterior
  shape checks;
- use `--sampler nuts` only for rows where the HMC posterior looks suspicious
  or where publication-grade diagnostics are needed;
- keep redshift fixed to the configured `z_phz` value unless redshift is the
  parameter under study;
- use `--no-progress` for timing runs, because progress-bar synchronization can
  slow NumPyro and make JAX/GPU timing noisy;
- inspect `mcmc_diagnostics.json`: for NUTS, large `mean_num_steps` means
  the sampler is spending many DSPS gradient evaluations per accepted draw.

Adam/MAP and Bayesian sampling use the same forward model:

```text
parameters -> DSPS SED -> model photometry
```

They differ in the inference step.

Adam/MAP asks:

```text
Which parameter vector gives the best objective value?
```

NUTS/HMC asks:

```text
What region of parameter space has high posterior probability?
```

The posterior is:

```text
posterior(parameters | data) ∝ likelihood(data | parameters) * prior(parameters)
```

The likelihood is still the Gaussian photometric likelihood:

```text
observed_mag_band ~ Normal(model_mag_band, sigma_mag)
```

The priors come from the `sample.priors` config. By default they are truncated normal distributions inside the same physical bounds used by Adam:

```yaml
sample:
  priors:
    log10_sfr:
      type: truncated_normal
      loc: 0.0
      scale: 1.75
    dust_av:
      type: truncated_normal
      loc: 0.2
      scale: 0.6
    log10_metallicity:
      type: truncated_normal
      loc: -2.0
      scale: 0.7
```

A prior means:

```text
what parameter values are plausible before seeing this galaxy's photometry
```

A likelihood means:

```text
how compatible the observed photometry is with parameters after running DSPS
```

A posterior means:

```text
what parameter values are plausible after combining prior knowledge and photometry
```

NUTS/HMC uses JAX gradients of the log posterior. This is why it is a good match for DSPS in this repo: the forward model and likelihood are differentiable.

The output is not one best parameter set. It is a cloud of posterior samples:

```text
posterior_samples.csv
```

From these samples, the code computes:

- posterior median
- posterior mean
- 5/16/50/84/95 percentiles
- posterior predictive photometry
- corner plot
- trace plot
- MCMC diagnostics

This adds information Adam cannot give:

- parameter uncertainty
- parameter correlations
- degeneracies, such as dust-metallicity tradeoff
- multi-modal or broad solutions
- posterior predictive uncertainty

This matters for the current Euclid VIS/Y/J/H setup. A MAP point can fit photometry well while being physically misleading because several parameters produce similar colors. MCMC does not magically make the four bands more informative, but it exposes the uncertainty and degeneracy:

- broad posterior means the data do not constrain that parameter
- tilted corner contours reveal dust/SFR/metallicity tradeoffs
- posterior predictive checks show whether many parameter combinations fit the photometry
- comparison with truth/proxy labels can be interpreted probabilistically rather than as a single-point failure

Recommended use:

```text
large MAP or population run -> choose representative/problem rows -> MCMC subset
```

Use rows with high chi-square, large residuals, extreme parameter values, and typical good fits. Full-catalog MCMC is too expensive, but a stratified subset can tell whether the MAP solution is unique or just one point in a broad degenerate posterior.

For batch Bayesian mode, the code samples one galaxy at a time and writes:

- `batch_posterior_summary.csv`
- `batch_posterior_samples.csv`
- `batch_posterior_predictive.csv`
- `batch_mcmc_diagnostics.csv`
- batch posterior interval plots

Full-catalog NUTS/HMC is intentionally not the default. Sampling thousands of galaxies is much more expensive than Adam/MAP. The intended workflow is:

```text
Adam/MAP for full catalog
NUTS/HMC for selected galaxies or small validation batches
population MAP for scalable population-level regularization
```

## Analysis of `outputs/runs/phz1_population_3000`

The existing 3000-galaxy population run fits photometry well:

```text
median_abs_residual_mag = 0.026 mag
median_reduced_chi2    = 0.711
n_valid_galaxies       = 3000
```

The independent MAP run is similar:

```text
median_abs_residual_mag = 0.031 mag
median_reduced_chi2    = 0.634
```

So the DSPS wrapper can reproduce VIS/Y/J/H photometry under this simplified model. The physical truth diagnostics from that run are weaker:

```text
z_obs RMSE                = 0.626
log10_sfr RMSE            = 0.883
dust_av proxy RMSE        = 0.572
log10_metallicity RMSE    = 0.596
```

Important interpretation: that run was produced before the scientific-comparison cleanup. It had `z_obs` as a free parameter and compared catalog `log_sfr_true` to the fitted SFH amplitude `log10_sfr`. Both choices make the physical truth diagnostics harder to interpret.

The trace plot shows the key failure mode:

```text
photometry loss decreases
truth RMSE increases
```

This is not an optimizer bug. The optimizer minimizes photometric chi-square, not truth RMSE. Truth/proxy RMSE is logged only as a diagnostic. If the RMSE rises while chi-square falls, the model has found parameters that fit four-band photometry better but move away from the available truth/proxy labels. That is expected when:

- redshift is allowed to absorb color mismatch
- SFR, dust, and metallicity are degenerate with only VIS/Y/J/H
- dust truth is `E(B-V)` while DSPS fits `A_V`
- metallicity truth is gas-phase oxygen while DSPS fits stellar metallicity
- the SFH is a fixed-shape lognormal, while the catalog SFR truth may come from a different SFH model

The implemented fix is therefore:

- keep `z_obs` fixed to `z_phz` by default
- compare `log_sfr_true` to derived `fit_log10_sfr_at_obs`, not to internal `fit_log10_sfr`
- keep dust and metallicity comparisons but document them as proxies
- use the truth-overlaid corner plot only for paired fit/truth quantities

The rerun `outputs/runs/phz1_population_3000_fixed_z` shows the cleaner interpretation:

```text
median_abs_residual_mag        = 0.032 mag
median_reduced_chi2           = 0.622
z_phz vs z_true RMSE          = 0.423
log10_sfr_at_obs RMSE         = 0.762
dust A_V proxy RMSE           = 0.587
log10_metallicity proxy RMSE  = 0.628
```

The SFR distribution is broadly aligned and the median bias is small. Dust does not match well: inferred `A_V` remains much lower than the `4.05 * E(B-V)` proxy for many galaxies, with weak correlation. Metallicity has moderate rank correlation but the inferred distribution is much broader than the gas-oxygen proxy. These are scientific/model limitations, not evidence that the plotting code is mixing columns.

## Ground Truth

Current usable truth:

```text
z_true
log_sfr_true
metallicity_true
dust_ebv_true
```

These are used for diagnostics only. They are not injected into the DSPS fit objective.

Redshift diagnostics compare:

```text
delta_z_obs_minus_truth = z_phz - z_true
```

SFR diagnostics compare catalog SFR to a derived DSPS quantity:

```text
truth_log10_sfr_at_obs = log_sfr_true
fit_log10_sfr_at_obs = log10(SFR(t_obs)) from the fitted lognormal SFH
```

This is more accurate than comparing `log_sfr_true` to fitted `log10_sfr`, because fitted `log10_sfr` is an internal amplitude parameter of the simplified SFH.

Metallicity diagnostics use a proxy conversion from gas-phase oxygen abundance to total metallicity:

```text
log10_metallicity_true = metallicity_true - 10.61
```

This assumes solar oxygen abundance `12 + log10(O/H) = 8.69` and `Z_sun = 0.012`. It is useful, but not exact: DSPS fits stellar metallicity while the catalog truth is gas-phase oxygen abundance.

Dust diagnostics use an `A_V` proxy:

```yaml
dust_av:
  column: dust_ebv_true
  scale: 4.05
```

This assumes `A_V = R_V * E(B-V)` with `R_V = 4.05`, matching the DSPS attenuation-curve convention. It is still a proxy if the catalog dust was generated with a different attenuation law, geometry, or `R_V`.

The current pipeline can honestly claim:

- DSPS can fit the simulated VIS/Y/J/H photometry well under the configured model.
- `z_phz` differs from `z_true` by the reported redshift diagnostics.
- derived current SFR, dust `A_V` proxy, and metallicity proxy can be compared to catalog truth diagnostics.

It should not claim:

- full SFH recovery
- strict stellar metallicity recovery from gas-phase oxygen truth
- strict dust-law recovery from `E(B-V)` truth
- individual physical-parameter recovery without acknowledging four-band degeneracies

Truth columns are configured under:

```yaml
truth:
  redshift_column: z_true
  parameter_columns:
    log10_metallicity:
      column: metallicity_true
      offset: -10.61
    log10_sfr_at_obs: log_sfr_true
    dust_av:
      column: dust_ebv_true
      scale: 4.05
```

Those values are propagated into batch summaries, truth metrics, truth-vs-inferred plots, and population corner overlays.

## What Is Arbitrary

These choices are currently arbitrary or pragmatic:

- `sigma_mag = 0.05` is assumed, not read from per-object catalog uncertainties.
- SFH is a simple lognormal shape.
- `sfh_t_peak = 4.0` and `sfh_tau = 0.6` are fixed.
- `metallicity_scatter = 0.2` is fixed.
- `dust_slope = -0.7` is fixed.
- The fit uses only VIS/Y/J/H, not LSST bands.
- Population fitting is chunk-local MAP, not full posterior inference.
- Parameter bounds are chosen to keep optimization stable, not derived from priors in a survey paper.
- Dust truth comparison assumes `A_V = 4.05 * E(B-V)`.
- Metallicity truth comparison treats gas-phase oxygen abundance as a proxy for stellar metallicity.

## What Is Fixed by Data

These pieces come from data or files:

- catalog fluxes in VIS/Y/J/H
- `z_phz` redshift used by DSPS
- `z_true` redshift for diagnostics
- `log_sfr_true`, `metallicity_true`, and `dust_ebv_true` truth/proxy diagnostics
- VIS and NISP Y/J/H throughput curves
- SSP template file `ssp_data_fsps_v3.2_lgmet_age.h5`

## Outputs

Single-galaxy runs write:

- `selected_galaxy.json`
- `model_parameters.json`
- `sed.csv`
- `sed.png`
- `photometry_comparison.csv`
- `photometry_comparison.png`

Fit runs also write:

- `fit_result.json`
- `fit_trace.csv`
- `fit_trace.png`

Batch forward runs write:

- `batch_photometry_comparison.csv`
- `batch_summary.json`
- `batch_summary_by_band.csv`
- `batch_summary_by_galaxy.csv`
- batch diagnostic plots

Batch fit runs write:

- `batch_fit_results.csv`
- `batch_fit_photometry_comparison.csv`
- `batch_fit_trace.csv`
- `batch_fit_truth_metrics.csv`
- `batch_fit_parameter_truth.png`
- `batch_fit_trace_truth.png`
- batch diagnostic plots

Population runs write:

- `population_fit_results.csv`
- `population_fit_photometry_comparison.csv`
- `population_hyperparameters.csv`
- `population_fit_trace.csv`
- `population_parameter_truth_metrics.csv`
- `population_corner_parameters_with_truth.png`
- `population_parameter_distributions_with_truth.png`
- `population_fit_parameter_truth.png`
- `population_fit_trace_truth.png`
- population diagnostic plots

## Diagnostic Plots

The SED plot shows:

- intrinsic rest-frame SED
- dust-attenuated rest-frame SED
- vertical effective wavelengths for each band
- passband shapes overlaid on the SED plot

The photometry plot shows:

- observed simulated catalog magnitudes
- DSPS model magnitudes
- residuals in magnitudes

Batch plots show:

- residuals by band
- observed versus model photometry
- reduced chi2 distribution
- redshift used by DSPS versus redshift truth

## How to Read Results

Low residuals mean:

```text
within this simplified DSPS model, the fitted parameters can reproduce the catalog photometry
```

Low residuals do not mean:

```text
the recovered physical parameters are physically true
```

Reason: the catalog truth columns are only partly like-for-like with the DSPS fit parameters. With four bands and several correlated physical effects, many parameter combinations can produce similar photometry.

## Main Limitations and Next Steps

The largest scientific limitations are:

1. Add per-object photometric uncertainties instead of fixed `sigma_mag`.
2. Use more bands if the science goal needs stronger physical constraints.
3. Fit or parameterize more SFH flexibility only after adding more data or stronger priors.
4. Move population inference from chunk-local MAP to global posterior inference if needed.
5. Replace gas oxygen and dust `E(B-V)` proxies with like-for-like stellar metallicity and attenuation truth if available.
6. Use posterior sampling on validation subsets to quantify degeneracies, not just MAP points.

The current code is therefore best understood as:

```text
a fast, differentiable, GPU-ready DSPS photometry fitting pipeline for simulated Euclid fluxes
```

not yet:

```text
a complete physical recovery pipeline with fully validated like-for-like truth labels
```
