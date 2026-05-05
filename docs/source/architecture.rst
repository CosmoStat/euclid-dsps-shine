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
     filters.py      Euclid/LSST transmission curve loading.
     fit.py          MAP and population optimization.
     io.py           Parquet, row, unit, JSON, and CSV helpers.
     likelihood.py   Shared likelihood helpers.
     mcmc.py         NumPyro posterior sampling.
     model.py        Native DSPS boundary.
     pipeline.py     Compatibility facade for workflow imports.
     reports.py      Compatibility facade for reporting imports.
     selection.py    Single-row catalog selection.
     reporting/
       eda.py        EDA report exports.
       fit.py        MAP/population report exports.
       forward.py    Forward-model report exports.
       posterior.py  Posterior report exports.
       workflow.py   Composite workflow report exports.
       core.py       Report tables and plots.
     workflows/
       bayesian.py   Bayesian workflow exports.
       eda.py        EDA workflow exports.
       forward.py    Forward-model workflow exports.
       map_fit.py    MAP workflow exports.
       population.py Population workflow exports.
       workflow.py   Composite workflow exports.
       core.py       End-to-end CLI workflows.
   configs/
     fs2_phz1.yaml   Default Euclid FS2 PHZ setup.
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

``fit.py`` and ``mcmc.py``
  Own optimizer and sampler behavior. They should depend on the model boundary
  and observation dataclasses, not on parquet or report-writing concerns.

``workflows/*.py``
  Composes workflows from the layers above. It is allowed to orchestrate, but
  should avoid complex scientific logic that belongs in ``model.py``,
  ``fit.py``, or ``io.py``. Focused modules expose stable entry points by
  workflow type, while ``core.py`` keeps the shared implementation and helpers.

``reporting/*.py``
  Owns artifact writing. Focused modules expose stable entry points by report
  type, while ``core.py`` keeps shared plotting/table implementation.

``pipeline.py`` and ``reports.py``
  Compatibility facades retained for existing scripts. New code should import
  from ``euclid_dsps.workflows`` and ``euclid_dsps.reporting``.

Design Rules
------------

* Keep DSPS imports isolated in ``model.py``.
* Keep catalog-specific aliases and truth transforms in config or ``io.py``.
* Keep output files deterministic and named with snake_case.
* Treat ``Data/`` and ``outputs/`` as local runtime state.
* Add tests or smoke commands when changing model, fit, sampling, or catalog
  contracts.
* Prefer new config keys over hidden constants when changing scientific setup.

Current Technical Debt
----------------------

``workflows/core.py`` and ``reporting/core.py`` retain shared implementation to
avoid risky movement of coupled helper functions. The public modules are now
split by workflow/report type, so future internal movement can happen behind
stable imports. The safer sequence is documented in :doc:`refactor_roadmap`.
