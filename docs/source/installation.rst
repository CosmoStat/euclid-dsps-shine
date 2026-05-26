Installation
============

Python Environment
------------------

Use the existing ``shine`` environment:

.. code-block:: bash

   conda activate shine
   python -m pip install -e .

The codebase should still import under ``uv`` for tests and packaging checks:

.. code-block:: bash

   uv sync
   uv run python -m compileall euclid_dsps scripts
   uv run euclid-dsps --help

Diffstar SFH Optional Extra
---------------------------

``configs/popcosmos_diffstar.yaml`` requires ``diffstar`` and ``diffmah``. In
the runtime environment:

.. code-block:: bash

   python -m pip install -e '.[diffstar]'

The standard binned-SFH config does not require this optional extra.

FSPS And python-FSPS
--------------------

The PopCosmos-like config requires generated FSPS gas and AGN grids. Install
FSPS and python-FSPS in the runtime environment used for generation:

.. code-block:: bash

   cd "$HOME/src"
   export SPS_HOME="$HOME/src/fsps"
   git clone https://github.com/cconroy20/fsps.git "$SPS_HOME"

   cd /home/maxime/src/DSPS-pop-cosmos
   export SPS_HOME="$HOME/src/fsps"
   uv pip install fsps

Check that python-FSPS sees the expected libraries:

.. code-block:: bash

   python -c "import fsps; sp=fsps.StellarPopulation(sfh=0); print(len(sp.wavelengths)); print(sp.isoc_library, sp.spec_library)"

Expected local output for the current assets is ``11149`` wavelength samples
with ``mist`` and ``c3k_a``.

Quality Checks
--------------

.. code-block:: bash

   uv run python -m compileall euclid_dsps scripts
   uv run pytest tests
   uv run python -m sphinx -W --keep-going -b html docs/source docs/build/html

GPU Note
--------

The default documented commands use CPU-safe JAX settings because the local
``shine`` environment may not have CUDA-enabled ``jaxlib``. For GPU production,
install a matching CUDA JAX stack first and keep large FSPS grids out of JAX
closures so XLA does not compile them as constants.
