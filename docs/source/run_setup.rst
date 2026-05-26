Run And Fit
===========

Single Config
-------------

There is one active config:

.. code-block:: text

   configs/popcosmos_binned.yaml

It is standalone and uses:

* LSST ``ugrizy`` + Euclid VIS/Y/J/H;
* flux-space likelihood with catalog flux errors;
* PopCosmos-like seven-bin SFH ratios;
* single stellar metallicity;
* Charlot-Fall age-dependent dust;
* FSPS gas SSP grid over gas metallicity and ionization;
* FSPS/CLUMPY AGN template grid over ``agn_tau``.

CPU Runtime
-----------

Use this on WSL or CPU-only environments:

.. code-block:: bash

   export JAX_PLATFORMS=cpu
   export XLA_PYTHON_CLIENT_PREALLOCATE=false

GPU Runtime
-----------

Use this only in an environment with CUDA-enabled JAX:

.. code-block:: bash

   export JAX_PLATFORMS=cuda
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export TF_GPU_ALLOCATOR=cuda_malloc_async

The fit and post-fit batch prediction paths pass large SSP/gas/AGN arrays as
dynamic JAX arguments, so JIT does not compile the full gas grid as a
closed-over constant. Gas-grid interpolation gathers only the four bracketing
gas-metallicity/gas-ionization slabs, but the grid plus optimizer/reporting
buffers still must fit in device memory. If a GPU run exhausts memory, reduce
``--batch-size`` first.

One Row
-------

Short optimizer smoke:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     fit --index 0 \
     --fit-maxiter 20 \
     --out outputs/runs/dev_popcosmos_one_short \
     --sed-samples 1

Production-style one-row run:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     fit --index 0 \
     --out outputs/runs/dev_popcosmos_one \
     --sed-samples 1

Batched MAP
-----------

Small batch:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     fit --limit 20 \
     --batch-size 5 \
     --out outputs/runs/dev_popcosmos_batch \
     --sed-samples 4

Larger batch:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     fit --limit 1000 \
     --batch-size 64 \
     --out outputs/runs/popcosmos_fit_1000 \
     --sed-samples 16

Forward Check
-------------

Run the model without optimization:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     fit --index 0 \
     --no-optimize \
     --out outputs/runs/dev_popcosmos_forward \
     --sed-samples 1

Posterior Smoke
---------------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     posterior --index 0 \
     --num-warmup 10 \
     --num-samples 10 \
     --out outputs/runs/dev_popcosmos_posterior_one

Parameter Vector
----------------

The active fit parameters are:

.. code-block:: text

   z_obs
   log10_stellar_mass
   dlog10_sfr_1 ... dlog10_sfr_6
   log10_stellar_metallicity
   tau2
   dust_index_n
   tau1_over_tau2
   log10_gas_metallicity
   log10_gas_ionization
   ln_fagn
   ln_tauagn

``ln_tauagn`` is restricted to the native FSPS/CLUMPY template range
``[ln(5), ln(150)]``.

Outputs
-------

MAP runs write normalized config, fit results, photometry comparisons,
optimizer diagnostics, performance summaries, and optional SED diagnostics under
the requested output directory.
