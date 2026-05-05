Refactor Roadmap
================

Assessment
----------

The current architecture is workable. DSPS calls are isolated, config is
centralized, and command workflows are explicit. The remaining quality risks are:

* Native DSPS import currently segfaults in the local ``shine`` environment,
  which blocks full forward-model smoke tests until the environment is repaired.
* Plot-level tests remain intentionally light. They verify EDA artifact creation
  but do not inspect image contents.
* Local runtime data and cloned helper repositories can clutter the project
  root if not ignored.

Recommended Sequence
--------------------

1. Add project hygiene. Done.

   Sphinx docs, black/ruff config, pytest wiring, CI workflow, and stronger
   ignore rules for local artifacts are in place.

2. Add lightweight tests before moving code. Done.

   Start with tests that do not require the full FS2 parquet or GPU:

   * config normalization
   * photometry unit conversions
   * truth value transforms
   * required catalog column discovery
   * filter wavelength unit conversion
   * row-index file parsing
   * synthetic parquet schema validation
   * EDA output smoke testing

3. Add explicit config schema validation. Done.

   ``euclid_dsps.config.validate_config`` checks required paths, band entries,
   units, redshift bounds, fit bounds, sample settings, and truth transforms
   immediately after YAML loading.

4. Split reporting. Done.

   Report code now lives in a package with focused public modules:

   .. code-block:: text

      euclid_dsps/reporting/
        eda.py
        fit.py
        forward.py
        posterior.py
        workflow.py
        core.py

   ``euclid_dsps.reports`` remains as a compatibility facade.

5. Split workflows. Done.

   Workflow orchestration now lives in focused public modules:

   .. code-block:: text

      euclid_dsps/workflows/
        bayesian.py
        eda.py
        forward.py
        map_fit.py
        population.py
        workflow.py
        core.py

   ``euclid_dsps.pipeline`` remains as a compatibility facade.

6. Introduce shared domain dataclasses. Partially done.

   ``euclid_dsps.columns.CatalogColumn`` now documents the selected CosmoHub
   columns. Runtime config still uses dictionaries because it maps directly to
   YAML. Add typed config dataclasses only if validation errors or IDE support
   become a real bottleneck.

7. Add reproducible smoke fixtures. Done.

   ``tests/data/synthetic_catalog.parquet`` has a few deterministic rows. CI
   validates the configured schema, row selection, derived metallicity helper,
   and EDA outputs without shipping private or large CosmoHub data.

Non-Goals
---------

Do not vendor native DSPS inside this repository. Keep DSPS as a dependency and
keep local clones outside the project root.

Do not move large parquet files, SSP assets, or generated outputs into source
control.

Do not split modules only for appearance. Split after tests exist or when a file
has a clear independent responsibility that can move with low behavior risk.
