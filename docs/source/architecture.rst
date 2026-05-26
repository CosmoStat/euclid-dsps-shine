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
     mcmc.py         NumPyro posterior sampling.
     model.py        Native DSPS boundary.
     nebular.py      Diagnostic-only SSP emission-line tables and crossings.
     performance.py  Runtime, throughput, and device-cost summaries.
     photometry.py   Central AB magnitude and Fnu flux conversions.
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
     fs2_phz1_science.yaml Active LSST+Euclid science setup.
     legacy/         Old config examples, not active workflow defaults.
     smoke_test.yaml Lightweight smoke-test setup.
   scripts/
     quickstart_one_galaxy.py
     convert_euclid_filters.py
   Data/             Local data and DSPS assets, not source.
   outputs/          Generated run outputs, not source.

The current package has good high-level boundaries. The main cleanup need is
not a rewrite; it is reducing module size and documenting contracts so new
science experiments stay local to config, model, fit, or reporting layers.

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
  DSPS directly.

``cosmos.py``
  Reconstructs template-level COSMOS proxy SEDs from ``sed_cosmos_*``,
  ``ebv_cosmos_*``, ``ext_curve_cosmos_*``, and ``frac_cosmos_*``.
  It owns SciPIC value-added or LePhare template/extinction loading,
  attenuation, synthetic photometry, rest-frame absolute-flux normalization,
  population validation, and COSMOS-vs-DSPS metrics.

``jax_runtime.py``
  Applies config/env JAX runtime choices before JAX-heavy modules are imported.
  Auto switch between cpu if GPU not found. GPU runs are enabled by
  changing ``runtime.jax_platforms`` and plugin autoload settings.

``fit.py`` and ``mcmc.py``
  Own optimizer and sampler behavior. They should depend on the model boundary
  and observation dataclasses, not on parquet or report-writing concerns.

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
