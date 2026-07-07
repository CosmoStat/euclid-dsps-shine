Architecture
============

Current Layout
--------------

The repository is organized as a small Python package plus local experiment
assets:

.. code-block:: text

   euclid_dsps/
     cli.py                 Command-line parser and command dispatch.
     config.py              YAML loading, inheritance, and normalization.
     model.py               Native DSPS boundary.
     io.py                  Parquet rows, units, truth transforms, JSON/CSV.
     filters.py             Euclid, LSST, Roman filter loading.
     photometry.py          AB magnitude and Fnu flux conversions.
     fit.py                 MAP and population optimization.
     mcmc.py                NumPyro and experimental BlackJAX sampling.
     posterior_target.py    Pure-JAX posterior target for samplers.
     parameter_vectors.py   Public theta-vector to DSPS JAX helper layer.
     observation_arrays.py  Batch photometry arrays for training workflows.
     diffsky_data/          HLTDS listing, download, preparation, validation.
     synthetic_diffsky/     FENIKS proposal generation, resampling, closure
                            photometry, manifests, diagnostics, validation.
     prior_learning/        Supervised and inferred RealNVP prior workflows.
     amortized/             FS2, HLTDS, and synthetic-closure amortized
                            posterior workflows.
     openuniverse/          OpenUniverse preparation and diagnostics helpers.
     reporting/             CSV, JSON, Markdown, and plot artifact writers.
     workflows/             CLI orchestration entry points.
     pipeline.py            Deprecated compatibility facade for workflows.
     reports.py             Deprecated compatibility facade for reports.
   configs/
     diffsky_synthetic_feniks_260617_50k.yaml
     diffsky_synthetic_feniks_260617_trueparam_closure.yaml
     prior_diffsky_synthetic_feniks_full_realnvp.yaml
     amortized_diffsky_synthetic_feniks_full_gpu.yaml
     diffsky_dataset_hltds_04_14.yaml
     diffsky_hltds_04_14_simple_gpu.yaml
     fs2_gpu.yaml
   scripts/
     diffsky_synthetic_feniks_50k_h100.slurm
     diffsky_amortized_train_h100.slurm
     diffsky_amortized_infer_h100.slurm
     build_diffsky_lowz_projected_truth_dataset.py
     benchmark_against_fsps_prospector.py
   Data/                    Local data and DSPS assets, not source.
   outputs/                 Generated run outputs, not source.

The active production path is intentionally narrow: synthetic
Diffsky/FENIKS proposal pools, local DSPS closure photometry, strict closure
validation, supervised 18D prior learning, amortized posterior inference, and
held-out closure evaluation. HLTDS and Euclid FS2 remain supported reference
paths. Generated data and spectral assets stay in ``Data/``; generated reports
and run outputs stay in ``outputs/``.

Production Data Flow
--------------------

.. code-block:: text

   configs/*.yaml
        |
        v
   cli.py
        |
        +--> synthetic_diffsky/
        |       Diffsky/FENIKS proposals -> selection -> resampling
        |       -> local DSPS true flux -> noisy closure catalogue
        |       -> manifest/schema/diagnostics/validation_report
        |
        +--> prior_learning/
        |       18D closure truths -> bounded latent transform
        |       -> RealNVP supervised prior checkpoint
        |
        +--> amortized/
        |       noisy flux/error features + fixed DSPS decoder
        |       -> posterior samples, predictive residuals, calibration tables
        |
        +--> reporting/
                Markdown reports, CSV/JSON metrics, plots

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
  configured fluxes, flux errors, masks, and object identifiers in batch arrays
  so amortized training does not construct ``GalaxyObservation`` objects in the
  hot path. FS2 uses ten bands; Diffsky HLTDS uses fourteen LSST+Roman bands.

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

``synthetic_diffsky/``
  Owns the production closure dataset workflow: Diffsky/FENIKS backend imports,
  independent proposal generation, proposal-level selection, weighted
  resampling, metallicity convention handling, local DSPS closure photometry,
  flux-error injection, manifest/schema writing, population diagnostics, and
  validation gates. It must keep proposal photometry, closure truth, true
  photometry, noisy photometry, and provenance as separate data products.

``prior_learning/``
  Owns population-density learning from truth or inferred parameters. It
  builds bounded latent transforms from config, trains RealNVP priors, samples
  trained priors, and writes truth-vs-prior diagnostics. It does not own
  photometric encoders or the DSPS decoder.

``jax_runtime.py``
  Applies config/env JAX runtime choices before JAX-heavy modules are imported.
  Production GPU configs should fail fast when CUDA is required. CPU or
  auto-selection modes are for local smoke tests and import-safe diagnostics.

``fit.py``, ``mcmc.py``, and ``posterior_target.py``
  Own optimizer and sampler behavior. ``mcmc.py`` keeps the public posterior
  workflow contract, while ``posterior_target.py`` exposes the unconstrained
  pure-JAX log-density used by BlackJAX MCLMC. They should depend on the model
  boundary and observation dataclasses, not on parquet or report-writing
  concerns.

``amortized/``
  Owns amortized posterior workflows for FS2 and Diffsky HLTDS: latent
  transforms, encoder features, Student-t flux likelihood, Equinox encoder,
  standard-normal / supervised-checkpoint / joint RealNVP priors, negative
  ELBO, synthetic smoke, training, inference, catalog export, and diagnostics.
  The DSPS decoder remains fixed behind ``parameter_vectors.py``. Encoder
  features use configured ``flux_B + err_B`` arrays, with
  ``asinh(flux / flux_scale)`` for robust bright-object flux normalization and
  log-normalized errors. Training writes progressive logs, gradient-norm
  diagnostics, ``best``/``last`` checkpoints, and epoch checkpoints controlled
  by ``amortized.output.checkpoint_every``. Inference writes likelihood-
  normalized posterior predictive residuals, top chi-square objects, feature
  diagnostics, redshift comparisons, PIT diagnostics, posterior/prior corners,
  and learned RealNVP prior samples. It reuses the same DSPS model boundary and
  does not replace the MAP or MCMC baselines.

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
