Refactor Roadmap
================

Assessment
----------

The current architecture is workable. DSPS calls are isolated, config is
centralized, and command workflows are explicit. The biggest quality risks are:

* ``reports.py`` is too large and mixes table shaping with plotting.
* ``pipeline.py`` is too large and mixes orchestration for many commands.
* There is no formal test suite or small synthetic fixture.
* Config validation is implicit. Missing or misspelled YAML keys fail late.
* Local runtime data and cloned helper repositories can clutter the project
  root if not ignored.

Recommended Sequence
--------------------

1. Add project hygiene.

   Done in this cleanup: Sphinx docs, black/ruff config, CI workflow, and
   stronger ignore rules for local artifacts.

2. Add lightweight tests before moving code.

   Start with tests that do not require the full FS2 parquet or GPU:

   * config normalization
   * photometry unit conversions
   * truth value transforms
   * required catalog column discovery
   * filter wavelength unit conversion
   * row-index file parsing

3. Add explicit config schema validation.

   A small validation layer should check required top-level keys, band entries,
   free-parameter bounds, truth transforms, and path types immediately after
   YAML loading. This can stay lightweight with dataclasses, or use Pydantic if
   richer error messages become important.

4. Split reporting.

   Move report code into a package:

   .. code-block:: text

      euclid_dsps/reporting/
        tables.py
        plots.py
        posterior.py
        workflow.py

   Keep old public functions as small wrappers during the transition so the CLI
   does not change.

5. Split workflows.

   Move orchestration into:

   .. code-block:: text

      euclid_dsps/workflows/
        eda.py
        forward.py
        map_fit.py
        bayesian.py
        population.py

   ``pipeline.py`` can then become a compatibility facade imported by
   ``cli.py`` and scripts.

6. Introduce shared domain dataclasses.

   If parameter dictionaries become hard to reason about, add:

   .. code-block:: text

      euclid_dsps/types.py

   Candidate dataclasses:

   * ``ResolvedConfig``
   * ``BandConfig``
   * ``FitParameterSpec``
   * ``TruthColumnSpec``
   * ``WorkflowRunMetadata``

7. Add reproducible smoke fixtures.

   Keep a tiny synthetic parquet fixture under ``tests/data/``. It should have a
   few rows and deterministic photometry values. This enables CI tests without
   shipping private or large CosmoHub data.

Non-Goals
---------

Do not vendor native DSPS inside this repository. Keep DSPS as a dependency and
keep local clones outside the project root.

Do not move large parquet files, SSP assets, or generated outputs into source
control.

Do not split modules only for appearance. Split after tests exist or when a file
has a clear independent responsibility that can move with low behavior risk.
