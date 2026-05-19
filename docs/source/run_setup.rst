Parameters And Run Setup
========================

Default Configuration
---------------------

The default Euclid FS2 PHZ setup is:

.. code-block:: text

   configs/fs2_phz1.yaml

This config is Euclid-only. Ten-band LSST+Euclid runs are opt-in:

.. code-block:: text

   configs/fs2_phz1_10band.yaml

Use this config for LSST ``ugrizy`` in the photometric likelihood and
pseudo-SED diagnostic. The default config remains Euclid VIS/Y/J/H only.

Every CLI command accepts ``--config``:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml run-one --out outputs/runs/dev_one

Top-Level Paths
---------------

``catalog_path``
  Local parquet catalog path. For FS2 PHZ this is
  ``Data/Euclid FS2 LC galaxy catalog_phz1.parquet``.

``ssp_path``
  DSPS SSP template file. The default is
  ``Data/ssp_data_fsps_v3.2_lgmet_age.h5``.

Runtime
-------

``runtime`` controls JAX backend setup before DSPS-heavy modules are imported:

.. code-block:: yaml

   runtime:
     jax_platforms: "cpu"
     disable_jax_plugin_autoload: true
     xla_python_client_preallocate: false
     require_gpu: false
     expected_gpu_name:

The default is WSL-safe for the local ``shine`` environment. For a working CUDA
JAX stack, use ``jax_platforms: "cuda"`` and
``disable_jax_plugin_autoload: false``. Set ``require_gpu: true`` or
``EUCLID_DSPS_REQUIRE_GPU=1`` when a run must fail instead of silently falling
back to CPU. ``expected_gpu_name`` or ``EUCLID_DSPS_EXPECTED_GPU_NAME`` checks
the JAX GPU ``device_kind`` string, for example ``NVIDIA``.

Before a long GPU run, verify the environment:

.. code-block:: bash

   /home/maxime/miniforge3/envs/shine/bin/python \
     scripts/check_jax_gpu.py \
     --require-nvidia \
     --hold-seconds 10

This script prints ``nvidia-smi`` visibility, JAX devices, selected backend, and
runs a real JAX matrix multiplication while NVIDIA memory is allocated. If JAX
shows only ``cpu`` devices, the pipeline will not use the RTX card.

Batch fitting and COSMOS DSPS comparison are JAX-vectorized over each parquet
chunk, so increasing ``--batch-size`` uses more accelerator memory when a GPU
backend is active.

For repeated GPU runs, enable the persistent JAX compilation cache:

.. code-block:: yaml

   runtime:
     jax_compilation_cache_dir: outputs/jax_cache
     jax_persistent_cache_min_compile_time_secs: 1.0

This does not make the first compilation free, but it reduces repeated-run
startup cost when shapes, filters, SSP assets, and fit mode stay unchanged.

Selection
---------

``selection.index``
  Catalog row index for one-galaxy workflows.

``selection.require_positive_flux``
  Reject rows with non-positive configured photometry when selecting a row.

``selection.sort_by_flux``
  Optional photometry column used to sort before selecting.

Redshift
--------

The FS2 configs set initial DSPS ``z_obs`` from the NNPZ PDF median. The
70/90/95 percent intervals are still loaded so diagnostics and explicit
photo-z-prior experiments can use them, but production fast mode does not use
them as hard bounds by default:

.. code-block:: yaml

   redshift:
     column: phz_median
     truth_column: z_true_gal
     fixed_value: 0.5
     min: 0.0001
     max: 6.0
     prior_interval:
       min_column: phz_min_70
       max_column: phz_max_70
       probability: 0.70
       sigma_floor: 0.01
       sigma_ceiling: 0.6
     prior_intervals:
       - min_column: phz_min_70
         max_column: phz_max_70
         probability: 0.70
       - min_column: phz_min_90
         max_column: phz_max_90
         probability: 0.90
       - min_column: phz_min_95
         max_column: phz_max_95
         probability: 0.95

``column`` initializes the DSPS redshift. ``truth_column`` is diagnostic only.
``fixed_value`` is a fallback when the row value is missing or invalid.
``prior_interval`` still writes the compatibility field ``z_obs_prior_sigma``.
``prior_intervals`` writes ``z_obs_phz_min/max_70/90/95`` base parameters. They
are consumed only when a config explicitly enables ``type: phz_interval`` or
``fit.fast_grid_use_phz_bounds: true``.

Bands
-----

Each band entry defines the catalog column, units, uncertainty, and passband:

.. code-block:: yaml

   bands:
     - name: euclid_vis
       column: euclid_vis
       units: fnu_cgs
       sigma_mag: 0.05
       error_column: euclid_vis_el_model3_ext_odonnell_ext_error
       error_units: fnu_cgs
       sigma_mag_floor: 0.01
       sigma_mag_ceiling: 0.5
       filter:
         path: filters/Euclid_VIS.vis.dat

Supported photometry units are:

.. list-table::
   :header-rows: 1

   * - Unit
     - Meaning
   * - ``fnu_cgs``
     - ``Fnu`` in ``erg/s/cm^2/Hz``.
   * - ``abmag``
     - AB magnitude.
   * - ``microjy`` or ``ujy``
     - MicroJansky.

``sigma_mag`` is the fallback likelihood uncertainty. When ``error_column`` is
present in a catalog row, the pipeline converts that flux-density uncertainty
to a local AB-magnitude uncertainty and uses it instead. ``sigma_mag_floor``
and ``sigma_mag_ceiling`` prevent unrealistically tiny or huge weights.

Model Parameters
----------------

``model.fixed_parameters`` contains the baseline DSPS parameter dictionary.
Default free parameters can override these values during fitting.

.. code-block:: yaml

   model:
     n_sfh_bins: 96
     fixed_parameters:
       log10_sfr: 0.0
       log10_formed_mass_msun: 10.0
       sfh_t_peak: 4.0
       sfh_tau: 0.6
       sfh_burst_fraction: 0.0
       sfh_burst_time: 1.0
       sfh_burst_width: 0.12
       sfh_quench_time: 12.0
       sfh_quench_width: 0.5
       sfh_quench_depth: 0.0
       log10_metallicity: -2.0
       metallicity_scatter: 0.2
       dust_av: 0.2
       dust_slope: -0.7
     parameter_columns: {}

``parameter_columns`` can map model parameters to catalog columns when a value
should come from each row instead of the fixed config.

In the 10-band COSMOS comparison config, ``parameter_columns`` maps
``cosmos_ebv_*``, ``cosmos_frac_*``, and ``cosmos_ext_curve_*`` directly to the
catalog so DSPS uses the same two-component attenuation family as the COSMOS
template proxy.

Parameter meanings:

``log10_sfr``
  Historical logarithmic SFH amplitude in ``Msun/yr``. When
  ``log10_formed_mass_msun`` is present, the SFH is renormalized to the formed
  mass and ``log10_sfr`` no longer controls the luminosity amplitude.

``log10_formed_mass_msun``
  Base-10 logarithm of the formed stellar mass in ``Msun``. This is now the
  preferred DSPS amplitude parameter. Catalog ``log_stellar_mass`` is stored as
  ``log10(Msun h^-2)`` and is converted for diagnostics using
  ``log10(Msun) = log_stellar_mass + 2 log10(h)``.

``sfh_t_peak`` and ``sfh_tau``
  Peak time and width of the lognormal SFH baseline in Gyr. Lognormal and
  delayed families are common compact SFH parameterizations, but they are too
  restrictive for many galaxies.

``sfh_bin_log_sfr_*`` and ``sfh_bin_transition_width``
  Optional smooth non-parametric SFH shape bins. They are not enabled in the
  default local configs because the local broad-band data do not constrain many
  SFH degrees of freedom. If explicitly configured, bin values are
  mean-centered before formed-mass normalization.

``sfh_burst_fraction``, ``sfh_burst_time``, ``sfh_burst_width``
  Smooth Gaussian burst component added to the baseline SFH. This is motivated
  by the PROVABGS model family, which uses a richer SFH basis plus burst terms
  for DESI BGS SED inference. In this project it is experimental and fixed to
  zero in the local production configs.

``sfh_quench_time``, ``sfh_quench_width``, ``sfh_quench_depth``
  Smooth late-time suppression applied after a quench time. This is a compact
  differentiable proxy for the flexibility normally provided by non-parametric
  SFHs or NMF SFH bases. It is currently added-but-not-used for the local
  broad-band data: ``sfh_quench_depth`` is fixed to zero, and quench parameters
  should not be interpreted as inferred galaxy properties.

``log10_metallicity`` and ``metallicity_scatter``
  Stellar metallicity center and scatter for DSPS SSP weighting. The catalog
  ``metallicity_true`` column is gas-phase ``12 + log(O/H)``; any conversion to
  DSPS stellar metallicity is diagnostic only.

``dust_av`` and ``dust_slope``
  Scalar fallback dust parameters. The 10-band COSMOS config prefers
  row-resolved two-component COSMOS attenuation through ``cosmos_ebv_*``,
  ``cosmos_ext_curve_*``, and ``cosmos_frac_*``.

``cosmos_ebv_*``, ``cosmos_ext_curve_*``, ``cosmos_frac_*``
  Per-row COSMOS attenuation parameters copied from the catalog. They are used
  when ``cosmos_sed.use_cosmos_dust_in_dsps`` is true.

The default numbers are starting values and safe fallbacks, not final physical
priors. The scientific motivation is:

* `PROVABGS mock challenge <https://arxiv.org/abs/2202.01809>`__: richer SFH
  bases, burst terms, and explicit priors matter for physical parameter
  inference.
* `How to Measure Galaxy SFHs II <https://arxiv.org/abs/1811.03637>`__:
  non-parametric SFHs are more flexible, but their inference is prior
  dependent.
* `pop-cosmos <https://arxiv.org/abs/2402.00935>`__: population-level priors
  calibrated on COSMOS photometry are a better long-term target than fully
  independent broad-band fits.

The current values keep the optimization stable while leaving the scientific
assumptions visible in YAML. The default fits infer formed mass, redshift,
lognormal SFH shape where configured, dust where applicable, and metallicity.
Burst/quench modifiers and binned SFH parameters remain fixed by default
because broad-band photometry alone does not robustly identify those features.

Current "extra" model pieces:

* burst modifiers are implemented but inactive in the production configs;
* quench modifiers are implemented but inactive because ``sfh_quench_depth`` is
  zero;
* binned SFH parameters are implemented but absent from the local configs;
* scalar Salim dust remains as fallback, while the 10-band COSMOS config uses
  row-injected two-component COSMOS dust;
* emission-line targets exist in the catalog, but no local line-enabled SSP
  asset is configured, so emission lines are not part of the main likelihood.

Fit Parameters
--------------

``fit.free_parameters`` declares fitted parameters, initial values, and hard
bounds:

.. code-block:: yaml

   fit:
     method: jax_adam
     maxiter: 80
     learning_rate: 0.1
     tolerance: 1.0e-5
     patience: 18
     prior_weight: 1.0
     priors:
       z_obs:
         type: phz_interval
         tail_scale: 0.05
         weight: 1.0
     free_parameters:
       z_obs:
         initial: from_base
         bounds: [0.001, 6.0]
       log10_formed_mass_msun:
         initial: 10.0
         bounds: [6.0, 13.0]
       dust_av:
         initial: 0.2
         bounds: [0.0, 3.0]
       log10_metallicity:
         initial: -2.25
         bounds: [-3.9, -1.6]

Use ``initial: from_base`` when the initial value should come from the resolved
base parameter dictionary for each row.

``fit.priors`` adds differentiable penalties to the JAX objective. Supported
types are ``uniform``, ``normal``, ``truncated_normal``, ``scaled_beta``, and
``phz_interval``. For Gaussian priors, ``scale: from_base`` reads a row-resolved
``<parameter>_prior_sigma`` value from the base parameter dictionary. For
``phz_interval``, the prior is flat inside the 70 percent interval and steepens
through the 90/95 percent intervals. The reported ``chi2`` remains the
photometric chi-square; the prior only guides the optimization.

Population relations can be configured under ``fit.population.relations``:

.. code-block:: yaml

   population:
     relations:
       log10_metallicity:
         predictor: log10_formed_mass_msun
         pivot: 10.0
         intercept_initial: -2.25
         slope_initial: 0.15
         sigma_initial: 0.35

This replaces the independent Gaussian population prior for the target
parameter with ``target ~ Normal(intercept + slope * (predictor - pivot),
sigma)`` in the population MAP objective.

Bayesian Sampling
-----------------

``sample`` controls NumPyro HMC/NUTS:

.. code-block:: yaml

   sample:
     sampler: nuts
     num_warmup: 100
     num_samples: 200
     num_chains: 1
     chain_method: parallel
     target_accept_prob: 0.85
     max_tree_depth: 10
     num_steps: 8
     seed: 42
     init_from_map: true
     priors:
       z_obs:
         type: phz_interval
         tail_scale: 0.05
         weight: 1.0
       log10_sfr:
         type: truncated_normal
         loc: 0.0
         scale: 1.3
       dust_av:
         type: scaled_beta
         alpha: 1.2
         beta: 3.0
       log10_metallicity:
         type: uniform

Use ``--sampler hmc`` and a small ``--num-steps`` for predictable debugging.
Use ``--sampler nuts`` for more adaptive posterior checks on selected rows.
Supported prior types are ``uniform``, ``normal``, ``truncated_normal``,
``scaled_beta``, and ``phz_interval``. ``loc: from_base`` centers a Gaussian
prior on the row-resolved base parameter. ``phz_interval`` uses the same PHZ
interval penalty as MAP and does not need ``loc``/``scale``.

COSMOS Template SED Setup
-------------------------

``cosmos_sed`` controls LePhare COSMOS-template reconstruction:

.. code-block:: yaml

   cosmos_sed:
     value_added_data_dir: "Data/value_added_data"
     catalog_h: 0.67
     lephare_data_dir: "/home/maxime/.cache/lephare/data"
     template_subdir: "sed/GAL/COSMOS_SED"
     template_list: "COSMOS_MOD.list"
     expected_template_count: 31
     extinction_dir: "ext"
     extinction:
       curves:
         0: none
         1: SMC_prevot
         2: SB_calzetti
         3: SB_calzetti_bump1
         4: SB_calzetti_bump2
     component_fraction_policy: strict
     filter_response_kind: photon
     sample_plot_count: 12
     observed_photometry_target_sets:
       - continuum_internal_dust
     use_cosmos_dust_in_dsps: true

The current local parquet contains ``frac_cosmos_1`` and ``frac_cosmos_2``.
The default policy is therefore strict. Older exports can use
``component_fraction_policy: equal_if_missing`` explicitly; the fallback is
recorded in output diagnostics.

When ``value_added_data_dir`` is configured, COSMOS templates are loaded from
``galaxy_seds`` and attenuation curves are derived from ``galaxy_extincts``
using the SciPIC formula documented with the data release. This is the preferred
local source because it is the same value-added library shipped with the
catalog. If it is absent, the workflow falls back to the LePhare cache.

``observed_photometry_target_sets`` defaults to continuum-only in the current
science config. This avoids scoring a continuum+dust DSPS model against
catalog columns that include emission lines. ``use_cosmos_dust_in_dsps`` loads
the configured extinction curves into the JAX context and applies the
row-resolved two-component COSMOS attenuation inside DSPS.

Truth Comparisons
-----------------

Truth entries are report diagnostics. They do not constrain the default forward
model or MAP fit:

.. code-block:: yaml

   truth:
     redshift_column: z_true_gal
     parameter_columns:
       log10_formed_mass_msun:
         column: log_stellar_mass
         transform: log_stellar_mass_h2_to_msun
         h: 0.67
       log10_metallicity:
         column: metallicity_true
         offset: -10.61
       log10_sfr_at_obs: log_sfr_true
       dust_av:
         column: dust_ebv_true
         scale: 4.05

CLI Workflows And Outputs
-------------------------

``download-assets``
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml download-assets --out Data

Purpose:
  Download small native DSPS smoke-test assets into the requested directory.

Main outputs:
  Asset files under ``--out``. No science plots.

``eda``
~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml eda --out outputs/eda_phz1

Purpose:
  Inspect configured catalog columns before modelling.

Exported plots:
  ``flux_distributions.png``
    Histogram of ``log10(Fnu)`` for configured photometric bands. Used to catch
    unit mistakes, negative fluxes, and extreme tails.

  ``color_distributions.png``
    Histograms of adjacent-band AB colors, computed as
    ``-2.5 log10(Fnu_i/Fnu_j)``.

  ``redshift_diagnostics.png``
    Redshift histogram for configured ``redshift.column`` and optional
    comparison against ``redshift.truth_column``.

  ``physical_parameters.png``
    Histograms of available physical proxy columns such as metallicity, SFR,
    and dust labels.

Exported tables:
  ``catalog_schema.json``, ``catalog_stats.csv``, ``missing_values.csv``.

``run-one``
~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml run-one \
     --index 0 \
     --out outputs/runs/phz1_one

Purpose:
  Run the configured DSPS model for one catalog row without fitting.

Exported plots:
  ``sed.png``
    Intrinsic and attenuated DSPS rest-frame SED versus rest wavelength.

  ``photometry_comparison.png``
    Catalog AB magnitudes versus DSPS predicted AB magnitudes by band.
    Residuals are ``model_mag - observed_mag``.

Exported tables:
  ``selected_galaxy.json``, ``model_parameters.json``, ``sed.csv``,
  ``photometry_comparison.csv``, ``sed_diagnostic_dsps_sed.csv``,
  ``sed_diagnostic_photometry.csv``. With ``--plot-ground-truth``, the command
  also writes ``sed_diagnostic_ground_truth_sed.csv`` when local COSMOS
  resources are available.

``forward``
~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1_10band.yaml forward \
     --index 0 \
     --plot-ground-truth \
     --out outputs/runs/phz1_forward_row0

Purpose:
  Simpler no-fit entry point. With ``--index`` it behaves like ``run-one``.
  Without ``--index`` it streams rows like ``run-batch``.

Useful diagnostics:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1_10band.yaml forward \
     --limit 100 \
     --batch-size 50 \
     --save-sed-samples 8 \
     --plot-ground-truth \
     --out outputs/runs/phz1_forward_diag

``--save-sed-samples N`` writes ``sed_diagnostics/`` and
``sed_diagnostics_manifest.csv`` for the first ``N`` processed rows.
``--plot-filters`` is enabled by default and overlays passbands in rest-frame
wavelength. ``--plot-ground-truth`` overlays the COSMOS proxy SED only when the
local columns and template resources are present.

``fit-one``
~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml fit-one \
     --index 0 \
     --out outputs/runs/phz1_fit_one

Purpose:
  Fit configured free parameters for one row with JAX MAP optimization.

Additional exported plot:
  ``fit_trace.png``
    Optimizer objective and fitted parameters versus iteration. Used to detect
    stalled or unstable MAP solutions.

Additional exported tables:
  ``fit_result.json`` and ``fit_trace.csv``. ``run-one`` outputs are also
  written for the best-fit model.

``fit-one --bayesian``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml fit-one \
     --index 0 \
     --bayesian \
     --sampler hmc \
     --num-warmup 120 \
     --num-samples 400 \
     --num-chains 1 \
     --num-steps 8 \
     --no-progress \
     --out outputs/runs/phz1_mcmc_row_0_fast

Purpose:
  Sample one-row posterior with NumPyro HMC/NUTS.

Exported plots:
  ``posterior_trace.png``
    Chain traces for sampled parameters.

  ``posterior_corner.png``
    Posterior parameter corner plot.

  ``posterior_corner_with_truth.png``
    Same corner plot with available truth/proxy values overlaid.

  ``posterior_predictive_photometry.png``
    Catalog photometry versus posterior predictive magnitude intervals.

Exported tables:
  ``posterior_samples.csv``, ``posterior_derived_samples.csv``,
  ``posterior_summary.csv``, ``posterior_predictive_photometry.csv``,
  ``mcmc_diagnostics.json``.

``run-batch``
~~~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml run-batch \
     --limit 1000 \
     --batch-size 500 \
     --out outputs/runs/phz1_batch

Purpose:
  Run the configured DSPS model over many rows without fitting.

Exported plots:
  ``batch_dashboard.png``
    Four-panel diagnostic: residuals by band, model-vs-observed AB magnitude,
    ``log10`` reduced chi-square histogram, and redshift truth scatter when
    truth exists.

  ``batch_residuals_by_band.png``
    Boxplots of ``model_mag - observed_mag`` by band.

  ``batch_observed_vs_model.png``
    Scatter of observed AB magnitude versus model AB magnitude with one-to-one
    reference line.

  ``batch_redshift_truth.png``
    ``z_obs`` versus truth redshift, when configured.

  ``batch_parameter_truth.png``
    Inferred/fixed parameter values versus available truth/proxy columns.

Exported tables:
  ``batch_photometry_comparison.csv``, ``batch_summary.json``,
  ``batch_summary_by_band.csv``, ``batch_summary_by_galaxy.csv``,
  ``batch_truth_metrics.csv`` when truth/proxy pairs exist. With
  ``--save-sed-samples N``, ``sed_diagnostics_manifest.csv`` and per-sample SED
  CSV/PNG files are also written.

``fit-batch``
~~~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml fit-batch \
     --limit 1024 \
     --batch-size 64 \
     --out outputs/runs/phz1_fit_batch

Purpose:
  Fit independent MAP solutions for many rows with JAX-vmapped Adam.

Fast production mode:
  The 10-band config defaults to ``fit.fast_grid_search: true``. In this mode
  each galaxy is fit with coordinate grids for the fast configured parameters
  and an analytic luminosity-amplitude warm start. It is intended for large GPU
  runs where full per-galaxy Adam would be too expensive.

  Inferred in fast-grid mode:

  * ``z_obs``: selected from a small row-level grid around the initialized
    redshift using photometry and the configured redshift bounds; PHZ hard
    interval bounds are disabled unless ``fit.fast_grid_use_phz_bounds: true``;
  * ``log10_formed_mass_msun``: adjusted analytically from the broadband
    magnitude offset;
  * ``log10_metallicity``: selected on a small prior-bounded grid;
  * ``sfh_t_peak`` and ``sfh_tau``: selected on small prior-bounded grids;
  * derived ``sfr_at_obs_msun_per_yr`` and ``log10_sfr_at_obs``: inferred from
    the fitted formed mass plus fitted SFH shape.

  ``log10_sfr`` is not the main amplitude when
  ``log10_formed_mass_msun`` is present; it is retained for backward
  compatibility. The SFR quantity to use is derived ``fit_log10_sfr_at_obs``.

  The default fast grid axes are controlled by:

  .. code-block:: yaml

     fit:
       fast_grid_parameters:
         - z_obs
         - log10_metallicity
         - sfh_t_peak
         - sfh_tau
       redshift_grid_size: 5
       fast_grid_prior_width: 1.0

  The same axes can be overridden from the CLI:

  .. code-block:: bash

     euclid-dsps --config configs/fs2_phz1_10band.yaml fit-batch \
       --fast-grid \
       --fast-grid-parameters z_obs,log10_metallicity,sfh_t_peak,sfh_tau \
       --redshift-grid-size 5 \
       --fast-grid-prior-width 1.0 \
       --out outputs/runs/fast_grid

  Priors used by the 10-band config:

  * redshift: uniform by default in current production configs; PHZ interval
    prior is available only for explicit comparison runs;
  * mass: broad normal prior on ``log10_formed_mass_msun``;
  * metallicity: broad normal stellar-metallicity prior;
  * SFR: no direct independent ``log10_sfr`` prior in fast mode; SFR is derived
    from the fitted SFH shape, whose priors are broad normals on
    ``sfh_t_peak`` and ``sfh_tau``.

  Use ``--full-adam`` for a validation subset when all configured free
  parameters should be optimized with gradients:

  .. code-block:: bash

     euclid-dsps --config configs/fs2_phz1_10band.yaml fit-batch \
       --limit 100 \
       --batch-size 32 \
       --full-adam \
       --fit-maxiter 30 \
       --reporting-level light \
       --output-format parquet \
       --out outputs/runs/validation_full_adam

  Use ``--fast-warmstart`` for the fastest diagnostic pass. It adjusts only the
  amplitude and does not scan redshift.

Exported plots:
  Same aggregate plot family as ``run-batch``, with prefix ``batch_fit_*``.
  ``batch_fit_trace_truth.png`` is added when configured truth/proxy values
  are available; it shows diagnostic RMSE versus optimizer iteration.

Exported tables:
  ``batch_fit_results.csv``, ``batch_fit_photometry_comparison.csv``,
  ``batch_fit_trace.csv``, ``batch_fit_residuals_by_property.csv``,
  ``batch_fit_summary*.csv/json``,
  ``batch_fit_truth_metrics.csv`` when available.

During batch fitting, per-chunk checkpoints are written under ``_chunks/``.
These files make long million-row runs recoverable even if a later chunk fails.

``fit-batch --bayesian``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml fit-batch \
     --bayesian \
     --limit 8 \
     --batch-size 1 \
     --out outputs/runs/phz1_hmc_batch

Purpose:
  Run independent HMC/NUTS posterior checks for a small row subset.

Exported plots:
  ``batch_posterior_parameter_intervals.png``
    Per-row posterior median and 16--84 percent intervals for each parameter.

  ``batch_posterior_predictive.png``
    Posterior median photometry residuals versus observed magnitude.

  ``batch_mcmc_diagnostics.png``
    Divergences and acceptance probability by row.

Exported tables:
  ``batch_posterior_summary.csv``, ``batch_posterior_predictive.csv``,
  ``batch_mcmc_diagnostics.csv``, ``batch_posterior_samples.csv``,
  ``batch_mcmc_summary.json``.

``fit-population``
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml fit-population \
     --limit 1024 \
     --batch-size 64 \
     --out outputs/runs/phz1_population

Purpose:
  Fit chunked population MAP model with Gaussian regularizers and configured
  physical relation priors, such as mass-metallicity.

Fast mode behavior:
  If ``fit.fast_grid_search`` or ``fit.fast_warmstart_only`` is enabled,
  ``fit-population`` does not run the slow joint population Adam objective.
  It runs the same fast per-galaxy fit as ``fit-batch`` and writes empirical
  chunk-level population summaries. ``population_hyperparameters`` rows with
  ``kind=fast_empirical`` contain mean/scatter of fitted values; rows with
  ``kind=fast_relation`` contain a post-fit linear regression such as
  mass-metallicity. These are diagnostics, not jointly optimized population
  priors.

  Use ``--full-adam`` to run the true joint population MAP optimizer.

Exported plots:
  Same aggregate plot family as ``run-batch``, with prefix
  ``population_fit_*``.

  ``population_corner_parameters.png``
    Corner plot of population MAP parameter estimates.

  ``population_parameter_distributions.png``
    One-dimensional histograms of population MAP parameters.

  ``population_corner_parameters_with_truth.png`` / ``population_parameter_distributions_with_truth.png``
    Same plots with configured truth/proxy distributions overlaid.

Exported tables:
  ``population_fit_results.csv``,
  ``population_fit_photometry_comparison.csv``,
  ``population_hyperparameters.csv`` with Gaussian and relation hyperparameters,
  ``population_fit_trace.csv``, ``population_fit_residuals_by_property.csv``,
  ``population_map_parameters.csv``, ``population_map_parameter_summary.csv``,
  truth/proxy summary tables when available.

``fit-workflow``
~~~~~~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml fit-workflow \
     --limit 1000 \
     --batch-size 64 \
     --hmc-n 20 \
     --out outputs/runs/phz1_workflow

Purpose:
  Run independent MAP, HMC subset, population MAP, then comparison reports.

Exported comparison plots under ``comparison/``:
  ``map_vs_population_parameters.png``
    MAP versus population MAP parameter values per galaxy.

  ``map_vs_population_chi2.png``
    Fit-quality difference between independent MAP and population MAP.

  ``hmc_vs_map_population.png``
    Posterior medians versus MAP and population MAP for HMC subset.

  ``corner_*``
    Parameter distribution overlays for MAP, population MAP, and HMC samples.

Exported comparison tables:
  ``map_vs_population_parameters.csv``,
  ``map_vs_population_fit_quality.csv``, ``hmc_vs_map_population.csv``,
  ``workflow_comparison_summary.json``.

``report-workflow``
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml report-workflow \
     --run-dir outputs/runs/phz1_workflow

Purpose:
  Regenerate ``fit-workflow`` comparison tables and plots from an existing run.

``cosmos-sed``
~~~~~~~~~~~~~~

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1_10band.yaml cosmos-sed \
     --limit 2000 \
     --batch-size 512 \
     --population-dsps \
     --plot-samples 10 \
     --out outputs/runs/cosmos_sed_population_dsps_10band_2000_gpu

Purpose:
  Reconstruct COSMOS-template pseudo-ground-truth SEDs, normalize them with
  Euclid ``*_abs`` fluxes, optionally fit DSPS, then compare COSMOS proxy SEDs
  and DSPS outputs.

Standalone plots:
  ``cosmos_sed_example.png``
    First reconstructed COSMOS proxy SED. Shows scaled template ``Lnu`` versus
    rest wavelength.

  ``cosmos_sed_sample_set.png``
    Visual grid of sampled COSMOS SEDs. With DSPS comparison enabled, each row
    shows COSMOS proxy versus inferred DSPS plus log residuals.

  ``synthetic_vs_catalog_abs_flux.png``
    Synthetic Euclid absolute flux after normalization versus catalog
    ``euclid_*_abs``. One-to-one alignment validates normalization.

  ``cosmos_template_pair_heatmap.png``
    Counts of ``sed_cosmos_1`` versus ``sed_cosmos_2`` template pairs.

  ``cosmos_fraction_diagnostics.png``
    Component fraction distribution and normalization ``alpha`` versus
    normalized first-component fraction.

Branch-1 plots:
  ``branch1_rest_sed_comparison_example.png``
    One COSMOS proxy versus DSPS rest-SED comparison.

  ``branch1_rest_sed_metrics.png``
    Distribution of RMS log SED residuals by ``color_kind``.

  ``branch1_rest_color_residuals.png``
    DSPS minus COSMOS Euclid rest-color residuals in magnitudes.

  ``branch1_worst_sed_grid.png``
    Sixteen worst SED comparisons selected by ``rms_log_sed_residual``.

  ``branch1_rms_residual_heatmap.png``
    Median RMS log SED residual by ``z_true_gal`` bin and ``color_kind``.

  ``branch1_population_validation.png``
    Median rest-SED residual grouped by ``color_kind``, redshift, apparent
    magnitude, SFR proxy, metallicity proxy, stellar mass, template pair, and
    dust-curve pair. The CSV contains the same grouped values.

Branch-2 plots:
  ``branch2_observed_flux_residuals.png``
    Observed-frame flux residuals by band and target set. Continuum sets use
    clipped fractional residuals. Noisy target sets use clipped
    ``(model-observed)/sigma``.

  ``branch2_population_validation.png``
    Same population grouping axes as branch 1, but for observed-frame
    photometry residuals.

Population-mode report:
  ``cosmos_dsps_population_report.json``
    Explicitly records that population mode is chunk-regularized MAP. It is not
    a learned galaxy-population prior like pop-cosmos.

Fit-likelihood plots:
  ``cosmos_dsps_likelihood_dashboard.png``
    Classic four-panel fit diagnostic for the photometric likelihood.

  ``cosmos_dsps_likelihood_observed_vs_model.png``
    Model versus catalog AB magnitude.

  ``cosmos_dsps_likelihood_residuals_by_band.png``
    Magnitude residuals by band.

  ``cosmos_dsps_likelihood_redshift_truth.png``
    Fitted/input redshift versus ``z_true_gal`` when available.

  ``cosmos_dsps_likelihood_parameter_truth.png``
    Fitted parameters versus available truth/proxy columns.

Use ``--all`` only for full parquet processing. Use small ``--limit`` values
while changing model code.
