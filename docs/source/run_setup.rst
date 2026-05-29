Run And Fit
===========

Recommended Configs
-------------------

Use full AGN by default:

.. code-block:: text

   configs/popcosmos_binned.yaml
   configs/popcosmos_diffstar.yaml

Use no-AGN only for fallback/debug or controlled ablations:

.. code-block:: text

   configs/popcosmos_binned_noagn.yaml
   configs/popcosmos_diffstar_noagn.yaml

The default binned full-AGN config uses:

* LSST ``ugrizy`` plus Euclid VIS/Y/J/H;
* PopCosmos-like step SFH bins;
* Chabrier FSPS SSPs with ``z_sun=0.0142``;
* Prospector/FSPS-like dust;
* raw FSPS/CLOUDY gas grid;
* FSPS-native AGN component grid;
* FSPS-like AGN host attenuation and AGN/IGM ordering;
* flux-space Student-t likelihood with ``student_t_dof=2``.

Terminology
-----------

``full AGN`` means the galaxy SED includes stars, dust, gas, IGM, and an active
galactic nucleus component. The fit includes ``ln_fagn`` and ``ln_tauagn``.
``no-AGN`` means the AGN component is disabled and those two parameters are not
present. No-AGN is useful when checking the stellar+dust+gas model or reducing
memory pressure, but it is no longer the default science path.

Runtime
-------

CPU-safe local runtime:

.. code-block:: bash

   export JAX_PLATFORMS=cpu
   export XLA_PYTHON_CLIENT_PREALLOCATE=false

GPU runtime, only with CUDA-enabled JAX:

.. code-block:: bash

   uv sync --extra gpu
   export JAX_PLATFORMS=cuda
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export TF_GPU_ALLOCATOR=cuda_malloc_async

The full AGN component grid is about 3.9 GiB and the gas grid is about 2.7 GiB.
The benchmark script has a lazy component-grid path, but production fitting
loads the configured grids in the model context. Start with small batches and
increase only after checking memory.

One-Row Full AGN
----------------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     fit --index 0 \
     --fit-maxiter 20 \
     --out outputs/runs/dev_popcosmos_fullagn_one_short \
     --sed-samples 1

Production-style one-row run:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     fit --index 0 \
     --out outputs/runs/dev_popcosmos_fullagn_one \
     --sed-samples 4

Small Batch Full AGN
--------------------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     fit --limit 20 \
     --batch-size 2 \
     --out outputs/runs/dev_popcosmos_fullagn_batch20 \
     --sed-samples 4

Increase ``--batch-size`` only after the memory footprint is known on the target
machine.

No-AGN Fallback
---------------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned_noagn.yaml \
     fit --limit 20 \
     --batch-size 5 \
     --out outputs/runs/dev_popcosmos_noagn_batch20 \
     --sed-samples 4

The no-AGN config keeps the same SSP, dust, gas, filters, redshift, and
likelihood surface but removes ``ln_fagn`` and ``ln_tauagn``.

Diffstar
--------

Install the optional dependencies:

.. code-block:: bash

   python -m pip install -e '.[diffstar]'

Run a short Diffstar full-AGN smoke:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_diffstar.yaml \
     fit --index 0 \
     --fit-maxiter 20 \
     --out outputs/runs/dev_popcosmos_diffstar_fullagn_one_short \
     --sed-samples 1

Forward Check
-------------

Run the model and reporting path without optimization:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     fit --index 0 \
     --no-optimize \
     --out outputs/runs/dev_popcosmos_fullagn_forward \
     --sed-samples 1

Posterior Smoke
---------------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/popcosmos_binned.yaml \
     posterior --index 0 \
     --num-warmup 10 \
     --num-samples 10 \
     --out outputs/runs/dev_popcosmos_fullagn_posterior_one

Benchmark Against FSPS/Prospector
---------------------------------

Small smoke:

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
     --n 50 \
     --seed 0 \
     --out outputs/benchmarks/popcosmos_binned_full_forward_fsps_closure_n50

Regression-size benchmark:

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
     --n 500 \
     --seed 1 \
     --out outputs/benchmarks/popcosmos_binned_full_forward_fsps_closure_seed1_n500

Parameter Vector
----------------

Full AGN binned config:

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

No-AGN configs remove ``ln_fagn`` and ``ln_tauagn``. Diffstar configs replace
the six ``dlog10_sfr`` terms with the configured Diffstar parameters.

Outputs
-------

MAP runs write the normalized config, fit results, model/observed photometry,
optimizer diagnostics, performance summaries, and optional SED diagnostics
under the requested output directory. ``fit_quality`` follows the configured
photometric likelihood. ``chi2`` remains a Gaussian comparison diagnostic.
