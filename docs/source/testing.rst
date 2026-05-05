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
     - Config normalization, validation failures, configured catalog columns.
   * - ``tests/test_columns.py``
     - Catalog metadata uniqueness and unit documentation for key columns.
   * - ``tests/test_io.py``
     - Photometry unit conversions, truth transforms, row-index parsing, observation building.
   * - ``tests/test_filters.py``
     - Wavelength unit conversion, effective wavelength, ASCII filter sorting/clipping.
   * - ``tests/test_imports.py``
     - Public workflow/reporting facades and compatibility imports.
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

Known Runtime Gap
-----------------

The full DSPS forward-model workflows are not part of CI yet because native
``dsps`` import segfaults in the current local ``shine`` environment. Once the
environment is repaired, add a CPU-only smoke test for ``run-one`` against a
tiny SSP/filter fixture or a mocked DSPS context.
