Installation
============

Python environment
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

FSPS and python-FSPS
--------------------

The FS2 config requires generated FSPS SSP, gas, AGN component, and compressed
runtime grids. Install FSPS and python-FSPS only in the environment used for
asset generation:

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

Quality checks
--------------

.. code-block:: bash

   uv run python -m compileall euclid_dsps scripts
   uv run pytest tests
   uv run --with sphinx --with sphinx-rtd-theme python -m sphinx \
     -W --keep-going -b html docs/source docs/build/html

GPU runtime
-----------

The public fit configs are GPU configs. For production, use a CUDA-enabled JAX
installation and verify the backend before launching fits:

.. code-block:: bash

   uv sync --extra gpu
   uv run --extra gpu python -c "import jax; print(jax.devices())"

In the ``shine`` conda environment the equivalent check is:

.. code-block:: bash

   conda activate shine
   export JAX_PLATFORMS=cuda
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export TF_GPU_ALLOCATOR=cuda_malloc_async
   python -c "import jax; print(jax.default_backend()); print(jax.devices())"

If ``nvidia-smi`` works but JAX still reports only ``CpuDevice``, check that no
stale CPU-only ``jaxlib`` install is shadowing the active environment.
