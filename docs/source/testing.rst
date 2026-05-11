Testing
=======

Test Scope
----------

The test suite is intentionally split into fast, deterministic unit tests and
small smoke tests that do not import native DSPS. This keeps CI useful even when
the local scientific runtime or accelerator stack is unavailable.

Current coverage:

.. list-table::
   :header-rows: 1

   * - File
     - Scope
   * - ``tests/test_config.py``
     - Config normalization, COSMOS SED config validation failures, configured catalog columns.
   * - ``tests/test_cosmos.py``
     - LePhare COSMOS template loading, extinction, synthetic photometry, fraction policy, validation.
   * - ``tests/test_columns.py``
     - Catalog metadata uniqueness and unit documentation for key columns.
   * - ``tests/test_io.py``
     - Photometry unit conversions, truth transforms, row-index parsing, observation building.
   * - ``tests/test_mcmc.py``
     - Row-centered priors and scaled beta prior support.
   * - ``tests/test_filters.py``
     - Wavelength unit conversion, effective wavelength, ASCII filter sorting/clipping.
   * - ``tests/test_imports.py``
     - Public workflow/reporting facades and compatibility imports.
   * - ``tests/test_sed.py``
     - Broad-band pseudo-SED reconstruction, luminosity conversion, SED report outputs.
   * - ``tests/test_workflows_smoke.py``
     - Synthetic parquet schema validation, row selection, EDA artifact creation.

Synthetic Fixture
-----------------

``tests/data/synthetic_catalog.parquet`` is a tiny deterministic parquet file
with three rows. It covers the columns needed for schema validation, row
selection, truth transforms, metallicity derivation, and EDA reporting without
using private or large CosmoHub data.

Run Tests
---------

.. code-block:: bash

   python -m pytest tests

The repository also sets ``testpaths = ["tests"]`` in ``pyproject.toml`` so
plain ``python -m pytest`` does not collect local cloned repositories such as a
developer checkout of native DSPS.

Full Quality Gate
-----------------

Run the same checks as CI:

.. code-block:: bash

   find euclid_dsps scripts tests -name '*.py' -exec python -m black --check {} \;
   python -m ruff check euclid_dsps scripts tests
   python -m pytest tests
   python -m compileall euclid_dsps scripts/quickstart_one_galaxy.py
   python -m sphinx -W --keep-going -b html docs/source docs/build/html

Runtime Notes
-------------

Native DSPS/JAX workflows are still heavier than unit tests, so CI focuses on
fast deterministic tests. The local ``shine`` WSL environment segfaults when
JAX probes the CUDA13 plugin directly; ``configs/fs2_phz1.yaml`` defaults
project DSPS workflows to CPU and disables JAX plugin autoload before importing
JAX-heavy modules. Manual smoke commands should include:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml run-one --out outputs/runs/dev_one
   euclid-dsps --config configs/fs2_phz1.yaml cosmos-sed --limit 10 --plot-samples 12 --out outputs/runs/dev_cosmos_sed
   euclid-dsps --config configs/fs2_phz1.yaml cosmos-sed --limit 2 --compare-dsps --out outputs/runs/dev_cosmos_sed_dsps
   euclid-dsps --config configs/fs2_phz1.yaml cosmos-sed --limit 4 --batch-size 4 --population-dsps --out outputs/runs/dev_cosmos_sed_population
