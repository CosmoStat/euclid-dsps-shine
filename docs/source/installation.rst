Installation
============


.. code-block:: bash

   conda activate shine
   python -m pip install -e .

Future ``uv`` workflow target:

.. code-block:: bash

   uv sync
   uv run euclid-dsps --help
   uv run python -m compileall euclid_dsps scripts/quickstart_one_galaxy.py

GPU JAX may require a different install command under ``uv`` than under
``conda activate shine``. Keep the exact CPU/GPU commands documented before
using ``uv`` for production GPU runs.

For documentation and quality tooling:

.. code-block:: bash

   python -m pip install sphinx sphinx-rtd-theme
   python -m pip install pytest ruff black

Core Checks
-----------

Run the checks used while developing:

.. code-block:: bash

   python -m compileall euclid_dsps scripts/quickstart_one_galaxy.py
   euclid-dsps --config configs/fs2_phz1_science.yaml fit --index 0 --out outputs/runs/dev_fit_one
   euclid-dsps --config configs/fs2_phz1_science.yaml fit --limit 20 --batch-size 5 --out outputs/runs/dev_fit_batch

When posterior code changes, also run a tiny posterior smoke:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1_science.yaml posterior --index 0 --num-warmup 10 --num-samples 10 --out outputs/runs/dev_posterior_one

Documentation Build
-------------------

Build Sphinx documentation locally:

.. code-block:: bash

   python -m sphinx -W --keep-going -b html docs/source docs/build/html

Quality Tooling
---------------

Run the same formatting and lint checks as CI:

.. code-block:: bash

   python -m black --check euclid_dsps scripts
   python -m ruff check euclid_dsps scripts tests
   python -m pytest tests

Format code before committing:

.. code-block:: bash

   python -m black euclid_dsps scripts
   python -m ruff check --fix euclid_dsps scripts tests
