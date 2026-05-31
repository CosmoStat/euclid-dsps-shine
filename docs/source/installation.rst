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

``configs/popcosmos_diffstar.yaml``,
``configs/popcosmos_diffstar_compressed.yaml``, and
``configs/popcosmos_diffstar_noagn.yaml`` require ``diffstar`` and ``diffmah``.
In the runtime environment:

.. code-block:: bash

   python -m pip install -e '.[diffstar]'

The standard binned-SFH configs do not require this optional extra.

FSPS And python-FSPS
--------------------

The PopCosmos-like configs require generated FSPS SSP, gas, AGN component, and
compressed runtime grids. Install FSPS and python-FSPS in the runtime
environment used for generation:

.. code-block:: bash

   cd "$HOME/src"
   export SPS_HOME="$HOME/src/fsps"
   git clone https://github.com/cconroy20/fsps.git "$SPS_HOME"

   cd /home/maxime/src/DSPS
   export SPS_HOME="$HOME/src/fsps"
   python -m pip install fsps

Check that python-FSPS sees the expected libraries:

.. code-block:: bash

   python -c "import fsps; sp=fsps.StellarPopulation(sfh=0); print(len(sp.wavelengths)); print(sp.isoc_library, sp.spec_library)"

Expected local output for the current assets is ``11149`` wavelength samples
with ``mist`` and ``c3k_a``. See :doc:`data_download` for the exact generation
commands.

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
install the project GPU extra, which follows the official JAX pip CUDA wheel
path and brings the CUDA 12 runtime libraries into the ``uv`` environment:

.. code-block:: bash

   uv sync --extra gpu
   uv run --extra gpu python -c "import jax; print(jax.devices())"

If ``nvidia-smi`` works but JAX still reports only ``CpuDevice``, check that the
command is running through ``uv run --extra gpu`` and that no stale CPU-only
``jaxlib`` install is shadowing the project environment. Keep large FSPS grids
out of JAX closures so XLA does not compile them as constants.
