Architecture
============

Current Layout
--------------

The repository is organized as a small Python package plus local experiment
assets:

.. code-block:: text

   euclid_dsps/
     assets.py       Download small DSPS smoke-test assets.
     cli.py          Command-line parser and command dispatch.
     config.py       YAML loading and default normalization.
     cosmos.py       COSMOS-template proxy SED reconstruction.
     filters.py      Euclid/LSST transmission curve loading.
     fit.py          MAP and population optimization.
     io.py           Parquet, row, unit, JSON, and CSV helpers.
     jax_runtime.py  Conservative JAX runtime setup for local WSL/shine.
     likelihood.py   Shared likelihood helpers.
     mcmc.py         NumPyro and experimental BlackJAX posterior sampling.
     model.py        Native DSPS boundary.
     nebular.py      Diagnostic-only SSP emission-line tables and crossings.
     observation_arrays.py  Batch photometry arrays for training workflows.
     parameter_vectors.py   Public theta-vector to DSPS JAX helper layer.
     performance.py  Runtime, throughput, and device-cost summaries.
     photometry.py   Central AB magnitude and Fnu flux conversions.
     posterior_target.py  Pure-JAX posterior target for BlackJAX samplers.
     amortized/      FS2-only amortized posterior prototype.
     pipeline.py     Deprecated compatibility facade for workflow imports.
     reports.py      Deprecated compatibility facade for reporting imports.
     selection.py    Single-row catalog selection.
     reporting/
       cosmos.py     COSMOS SED diagnostic plots.
       eda.py        EDA report exports.
       fit.py        MAP/population report exports.
       forward.py    Forward-model report exports.
     posterior.py  Posterior report exports.
     workflow.py   Composite workflow report exports.
     core.py       Report tables and plots.
     diffsky_data/  Remote listing, bounded download, inventory, preparation,
                    validation, and reports for Diffsky/OpenCosmo HLTDS data.
     workflows/
       bayesian.py   Bayesian workflow exports.
       cosmos.py     COSMOS SED reconstruction workflow.
       eda.py        EDA workflow exports.
       forward.py    Forward-model workflow exports.
       map_fit.py    MAP workflow exports.
       population.py Population workflow exports.
       workflow.py   Composite workflow exports.
       core.py       End-to-end CLI workflows.
   configs/
     fs2_gpu.yaml                              Euclid FS2 GPU baseline.
     diffsky_hltds_04_14_simple_gpu.yaml      Main Diffsky HLTDS simple fit.
     diffsky_hltds_04_14_fixedz_closure_gpu.yaml
                                               Diffsky fixed-redshift closure.
     amortized_fs2_realnvp.yaml               FS2 amortized RealNVP prior.
   scripts/
     generate_fsps_ssp_grid.py
     generate_fsps_gas_grid.py
     generate_fsps_agn_component_grid.py
     benchmark_against_fsps_prospector.py
   Data/             Local data and DSPS assets, not source.
   outputs/          Generated run outputs, not source.

The active runtime path is intentionally narrow: Euclid FS2 MAP/posterior,
Diffsky HLTDS simple MAP recovery, and FS2 amortized RealNVP prior learning.
Generated data and spectral assets stay in ``Data/``; generated reports and
run outputs stay in ``outputs/``.

Layer Responsibilities
----------------------

``config.py``
  Loads YAML, applies defaults, and keeps run setup explicit. It should not
  read catalog data or call DSPS.

``io.py``
  Owns the catalog contract: parquet reads, required columns, row index
  handling, truth value transforms, photometry unit conversion, and JSON
  serialization.

``filters.py``
  Loads exact passbands from ASCII, HDF5, or FITS. Approximate top-hat filters
  are a fallback for smoke tests only.

``model.py``
  Contains the native DSPS boundary. Other modules should pass normalized
  dataclasses and parameter dictionaries into this layer rather than importing
  DSPS directly. This is where SSP interpolation, SFH weighting, dust, gas,
  AGN, IGM, and filter integration are combined.

``parameter_vectors.py``
  Owns the public JAX contract for converting physical ``theta`` vectors into
  DSPS parameter dictionaries and evaluating model magnitudes from arrays with
  shape ``[D]``, ``[N,D]``, or ``[K,N,D]``. It calls the same fast
  ``model_mags_jax_dynamic`` boundary used by MAP/MCMC code and preserves JAX
  gradients. Training-oriented code should use this module instead of private
  helpers in ``fit.py`` or NumPy report helpers such as ``predict_batch_mags``.

``observation_arrays.py``
  Provides array-based photometry extraction for training workflows. It keeps
  FS2 fluxes, flux errors, masks, and object identifiers in batch arrays so
  amortized training does not construct ``GalaxyObservation`` objects in the
  hot path.

``cosmos.py``
  Reconstructs template-level COSMOS proxy SEDs from ``sed_cosmos_*``,
  ``ebv_cosmos_*``, ``ext_curve_cosmos_*``, and ``frac_cosmos_*``.
  It owns SciPIC value-added or LePhare template/extinction loading,
  attenuation, synthetic photometry, rest-frame absolute-flux normalization,
  population validation, and COSMOS-vs-DSPS metrics.

``diffsky_data/``
  Owns the HLTDS dataset workflow: remote NERSC directory listing, bounded
  downloads, HDF5/parquet inventory, truth/photometry detection, normalized
  parquet preparation, validation for prior learning, dataset diagnostics, and
  Diffsky MAP fit reports. It must not silently invent unavailable truth
  columns or native photometric errors.

``jax_runtime.py``
  Applies config/env JAX runtime choices before JAX-heavy modules are imported.
  Auto switch between cpu if GPU not found. GPU runs are enabled by
  changing ``runtime.jax_platforms`` and plugin autoload settings.

``fit.py``, ``mcmc.py``, and ``posterior_target.py``
  Own optimizer and sampler behavior. ``mcmc.py`` keeps the public posterior
  workflow contract, while ``posterior_target.py`` exposes the unconstrained
  pure-JAX log-density used by BlackJAX MCLMC. They should depend on the model
  boundary and observation dataclasses, not on parquet or report-writing
  concerns.

``amortized/``
  Owns the FS2-only amortized posterior prototype: latent transforms,
  encoder features, Student-t flux likelihood, Equinox encoder, RealNVP prior,
  negative ELBO, synthetic smoke, training, inference, catalog export, and
  diagnostics. The encoder and RealNVP prior are optimized jointly by one
  Optax update, while the DSPS decoder remains fixed behind
  ``parameter_vectors.py``. Encoder features keep the 10 flux + 10 error
  contract, using ``asinh(flux / flux_scale)`` for robust bright-object flux
  normalization and log-normalized errors. Training writes progressive logs,
  gradient-norm diagnostics, ``best``/``last`` checkpoints, and epoch
  checkpoints controlled by ``amortized.output.checkpoint_every``. Inference
  writes normalized posterior predictive residuals, top chi-square objects,
  feature diagnostics, redshift proxy comparisons, redshift PIT diagnostics,
  catalog-proxy mass/SFR comparisons, contour-style posterior corners, and
  learned RealNVP prior samples/corners. It reuses the same DSPS model boundary
  and does not replace the MAP or MCMC baselines.

``nebular.py``
  Reads line metadata already loaded by ``model.py`` and writes diagnostic
  line/filter crossing artifacts. It must not alter the science likelihood
  until a no-double-count line model exists.

``performance.py``
  Owns wall-time, throughput, memory, JAX device, and GPU-hour reporting. It
  should stay lightweight and never require a GPU to import.

``workflows/*.py``
  Composes workflows from the layers above. It is allowed to orchestrate, but
  should avoid complex scientific logic that belongs in ``model.py``,
  ``fit.py``, or ``io.py``. Focused modules expose stable entry points by
  workflow type, while ``core.py`` keeps the shared implementation and helpers.

``reporting/*.py``
  Owns artifact writing. Focused modules expose stable entry points by report
  type, while ``core.py`` keeps shared plotting/table implementation.

``pipeline.py`` and ``reports.py``
  Deprecated compatibility facades retained for existing scripts and notebooks.
  They contain no workflow or plotting implementation. New source code should
  import from ``euclid_dsps.workflows`` and ``euclid_dsps.reporting``. They can
  be removed after local scripts such as ``scripts/quickstart_one_galaxy.py``
  and downstream notebooks no longer import them.

Design Rules
------------

* Keep DSPS imports isolated in ``model.py``.
* Keep catalog-specific aliases and truth transforms in config or ``io.py``.
* Keep output files deterministic and named with snake_case.
* Treat ``Data/`` and ``outputs/`` as local runtime state.
* Add tests or smoke commands when changing model, fit, sampling, or catalog
  contracts.
* Prefer new config keys over hidden constants when changing scientific setup.

Remaining Cleanup
-----------------

The main architectural risk is the size of the shared implementation modules.
``workflows/core.py`` still owns many orchestration helpers, and
``reporting/core.py`` still owns many plot families. Future refactors should
move those internals while keeping the stable ``euclid_dsps.workflows`` and
``euclid_dsps.reporting`` imports.
