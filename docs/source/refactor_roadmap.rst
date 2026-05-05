Refactor Roadmap
================

Assessment
----------

The current architecture is workable. DSPS calls are isolated, config is
centralized, and command workflows are explicit. The biggest quality risks are:

* ``reporting/core.py`` is still large and mixes table shaping with plotting.
* ``workflows/core.py`` is still large and mixes orchestration for many commands.
* There is no synthetic end-to-end parquet fixture yet.
* Local runtime data and cloned helper repositories can clutter the project
  root if not ignored.

Recommended Sequence
--------------------

1. Add project hygiene. Done.

   Sphinx docs, black/ruff config, pytest wiring, CI workflow, and stronger
   ignore rules for local artifacts are in place.

2. Add lightweight tests before moving code. Done for low-level contracts.

   Start with tests that do not require the full FS2 parquet or GPU:

   * config normalization
   * photometry unit conversions
   * truth value transforms
   * required catalog column discovery
   * filter wavelength unit conversion
   * row-index file parsing

3. Add explicit config schema validation. Done.

   ``euclid_dsps.config.validate_config`` checks required paths, band entries,
   units, redshift bounds, fit bounds, sample settings, and truth transforms
   immediately after YAML loading.

4. Split reporting. First pass done.

   Report code now lives in a package:

   .. code-block:: text

      euclid_dsps/reporting/
        core.py

   ``euclid_dsps.reports`` remains as a compatibility facade.

5. Split workflows. First pass done.

   Workflow orchestration now lives in:

   .. code-block:: text

      euclid_dsps/workflows/
        core.py

   ``euclid_dsps.pipeline`` remains as a compatibility facade.

6. Introduce shared domain dataclasses. Partially done.

   ``euclid_dsps.columns.CatalogColumn`` now documents the selected CosmoHub
   columns. Runtime config still uses dictionaries because it maps directly to
   YAML. Add typed config dataclasses only if validation errors or IDE support
   become a real bottleneck.

7. Add reproducible smoke fixtures.

   Keep a tiny synthetic parquet fixture under ``tests/data/``. It should have a
   few rows and deterministic photometry values. This will enable CI tests for
   EDA and row-selection workflows without shipping private or large CosmoHub
   data.

Non-Goals
---------

Do not vendor native DSPS inside this repository. Keep DSPS as a dependency and
keep local clones outside the project root.

Do not move large parquet files, SSP assets, or generated outputs into source
control.

Do not split modules only for appearance. Split after tests exist or when a file
has a clear independent responsibility that can move with low behavior risk.
