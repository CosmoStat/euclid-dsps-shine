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
   * - ``tests/test_parameter_vectors.py``
     - Public JAX theta-vector helpers and array photometry extraction.
   * - ``tests/test_amortized_*.py``
     - FS2 amortized latent transforms, features, likelihood, optional
       Equinox encoder/RealNVP modules, ELBO, synthetic smoke, and a small
       true-DSPS decoder end-to-end path with a synthetic ten-filter context.
   * - ``tests/test_filters.py``
     - Wavelength unit conversion, effective wavelength, ASCII filter sorting/clipping.
   * - ``tests/test_imports.py``
     - Public workflow/reporting facades and compatibility imports.
   * - ``tests/test_workflows_smoke.py``
     - Synthetic parquet schema validation, row selection, EDA artifact creation.
   * - ``tests/test_synthetic_diffsky_closure.py``
     - FENIKS/DSPS closure config loading, truth extraction, metallicity
       convention, selection gates, manifest/validation behavior, and CLI
       smoke coverage.
   * - ``tests/test_prior_learning_supervised.py``
     - Supervised prior schemas, bounded transforms, RealNVP training helpers,
       and prior reporting contracts.

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
   uv run --with sphinx --with sphinx-rtd-theme python -m sphinx \
     -W --keep-going -b html docs/source docs/build/html

Runtime Notes
-------------

Native DSPS/JAX workflows are heavier than unit tests, so CI focuses on fast
deterministic tests. Manual smoke commands should use the public GPU configs:

.. code-block:: bash

   export JAX_PLATFORMS=cuda
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export TF_GPU_ALLOCATOR=cuda_malloc_async

   python -m euclid_dsps.cli \
     --config configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml \
     fit --limit 8 --batch-size 8 --fit-maxiter 40 \
     --sed-samples 0 --reporting-level light \
     --out outputs/runs/dev_diffsky_fixedz_smoke

   python -m euclid_dsps.cli \
     --config configs/fs2_gpu.yaml \
     fit --index 0 --fit-maxiter 20 \
     --sed-samples 1 \
     --out outputs/runs/dev_fs2_gpu_one_short

Synthetic FENIKS closure smoke:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-generate-dsps-closure \
     --smoke \
     --overwrite

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_trueparam_closure.yaml \
     diffsky-validate-dsps-closure \
     --dataset-dir Data/diffsky/synthetic/feniks_260617_dsps_closure \
     --sample-size 24 \
     --batch-size 24 \
     --runtime cpu

Amortized FS2 smoke commands:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_fs2_realnvp.yaml \
     amortized-synthetic-smoke \
     --mock-decoder \
     --n-objects 64 \
     --epochs 2 \
     --out outputs/runs/dev_amortized_synthetic

   python -m euclid_dsps.cli \
     --config configs/amortized_fs2_realnvp.yaml \
     amortized-train-fs2 \
     --limit 32 \
     --batch-size 8 \
     --epochs 2 \
     --n-samples 1 \
     --out outputs/runs/dev_amortized_fs2

The amortized training outputs should include progressive checkpoints and
diagnostics:

.. code-block:: text

   training_log.csv
   training_progress.json
   training_summary.json
   checkpoints/best.eqx
   checkpoints/last.eqx
   checkpoints/epoch_0001.eqx
   encoder_grad_norm.png
   prior_grad_norm.png

``training_log.csv`` should contain nonzero ``encoder_grad_norm`` and
``prior_grad_norm`` once the KL term is active. That is the lightweight runtime
check that the RealNVP prior is trained jointly with the encoder.

Benchmark smoke:

.. code-block:: bash

   MPLCONFIGDIR=outputs/matplotlib_cache python scripts/benchmark_against_fsps_prospector.py \
     --runtime cpu \
     --config configs/fs2_gpu.yaml \
     --agn-component-grid Data/popcosmos_chabrier_agn_component_ssp_grid.h5 \
     --agn-host-attenuation fsps_diffuse_unit_tau \
     --agn-igm-order fsps_after_igm \
     --agn-baked-attenuation fsps_powerlaw_unit_tau \
     --agn-baked-dust-index -0.7 \
     --levels stellar_only stellar_plus_dust stellar_plus_gas full_noagn stellar_plus_agn stellar_plus_dust_plus_agn stellar_plus_gas_plus_agn full_agn \
     --n 5 \
     --seed 0 \
     --out outputs/benchmarks/smoke_popcosmos_full_forward
