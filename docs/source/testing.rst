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
fast deterministic tests.
Manual smoke commands should include:

.. code-block:: bash

   python -m euclid_dsps.cli --config configs/popcosmos_binned_compressed.yaml fit --index 0 --fit-maxiter 20 --out outputs/runs/dev_popcosmos_compressed_fullagn_one_short --sed-samples 1
   python -m euclid_dsps.cli --config configs/popcosmos_binned_compressed.yaml fit --limit 20 --batch-size 8 --sed-samples 0 --reporting-level light --out outputs/runs/dev_popcosmos_compressed_fullagn_batch
   python -m euclid_dsps.cli --config configs/popcosmos_binned_noagn.yaml fit --limit 20 --batch-size 5 --sed-samples 4 --out outputs/runs/dev_popcosmos_noagn_batch
   python -m euclid_dsps.cli --config configs/popcosmos_diffstar_compressed.yaml fit --index 0 --fit-maxiter 20 --out outputs/runs/dev_popcosmos_diffstar_compressed_fullagn_one_short --sed-samples 1
   python -m euclid_dsps.cli --config configs/popcosmos_binned_compressed.yaml check --kind cosmos --limit 10 --out outputs/runs/dev_cosmos_check

Benchmark smoke:

.. code-block:: bash

   MPLCONFIGDIR=outputs/matplotlib_cache python scripts/benchmark_against_fsps_prospector.py \
     --runtime cpu \
     --config configs/popcosmos_binned.yaml \
     --agn-component-grid Data/popcosmos_chabrier_agn_component_ssp_grid.h5 \
     --agn-host-attenuation fsps_diffuse_unit_tau \
     --agn-igm-order fsps_after_igm \
     --agn-baked-attenuation fsps_powerlaw_unit_tau \
     --agn-baked-dust-index -0.7 \
     --levels stellar_only stellar_plus_dust stellar_plus_gas full_noagn stellar_plus_agn stellar_plus_dust_plus_agn stellar_plus_gas_plus_agn full_agn \
     --n 5 \
     --seed 0 \
     --out outputs/benchmarks/smoke_popcosmos_full_forward
