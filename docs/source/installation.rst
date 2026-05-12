Installation
============


.. code-block:: bash

   conda activate shine
   python -m pip install -e .

For documentation and quality tooling:

.. code-block:: bash

   python -m pip install sphinx sphinx-rtd-theme
   python -m pip install pytest ruff black

Core Checks
-----------

Run the checks used while developing:

.. code-block:: bash

   python -m compileall euclid_dsps scripts/quickstart_one_galaxy.py
   euclid-dsps --config configs/fs2_phz1.yaml run-one --out outputs/runs/dev_one
   euclid-dsps --config configs/fs2_phz1.yaml run-batch --limit 20 --batch-size 5 --out outputs/runs/dev_batch

When fitting code changes, also run:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml fit-one --out outputs/runs/dev_fit_one
   euclid-dsps --config configs/fs2_phz1.yaml fit-batch --limit 3 --batch-size 3 --out outputs/runs/dev_fit_batch

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
