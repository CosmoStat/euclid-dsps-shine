Run The Pipeline
================

Public Configs
--------------

The public config surface is intentionally small:

.. list-table::
   :header-rows: 1

   * - Config
     - Use
   * - ``configs/fs2_gpu.yaml``
     - Euclid FS2 MAP/posterior baseline on GPU.
   * - ``configs/diffsky_hltds_04_14_simple_gpu.yaml``
     - Main Diffsky HLTDS 04/14/2026 simple DSPS MAP fit on GPU.
   * - ``configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml``
     - Diffsky closure/debug fit with redshift fixed to ``redshift_true``.
   * - ``configs/amortized_fs2_realnvp.yaml``
     - FS2 amortized encoder plus learned RealNVP prior.

Old Diffstar, OpenUniverse fit-ready, non-GPU, and experimental ablation configs
were removed from the public ``configs/`` directory. The first-pass science
path is either Euclid FS2, the Diffsky HLTDS sample dataset, or FS2 prior
learning.

GPU Runtime
-----------

All public fit configs request CUDA. In the ``shine`` environment, set the
runtime explicitly before launching long jobs:

.. code-block:: bash

   conda activate shine
   export JAX_PLATFORMS=cuda
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export TF_GPU_ALLOCATOR=cuda_malloc_async

Check the visible devices:

.. code-block:: bash

   python -c "import jax; print(jax.default_backend()); print(jax.devices())"

If this prints only CPU devices, fix the JAX/CUDA environment before launching
fits. The configs use ``runtime.require_gpu: true`` so production commands fail
fast instead of silently falling back to CPU.

Diffsky HLTDS Simple Fit
------------------------

This is the recommended dataset for the current physical-recovery tests. It
uses the prepared file:

.. code-block:: text

   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet

and the HLTDS SSP asset:

.. code-block:: text

   Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/diffsky_hltds_cosmos_260215_04_14_2026_ssp_data.hdf5

Small smoke fit:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_hltds_04_14_simple_gpu.yaml \
     fit \
     --limit 16 \
     --batch-size 16 \
     --fit-maxiter 80 \
     --sed-samples 0 \
     --reporting-level light \
     --out outputs/runs/diffsky_hltds_simple_smoke_n16

Larger batch:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_hltds_04_14_simple_gpu.yaml \
     fit \
     --limit 1000 \
     --batch-size 128 \
     --fit-maxiter 220 \
     --sed-samples 0 \
     --reporting-level light \
     --out outputs/runs/diffsky_hltds_simple_n1000

Regenerate the Diffsky recovery report after a run:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-fit-report \
     --run outputs/runs/diffsky_hltds_simple_n1000 \
     --config configs/diffsky_hltds_04_14_simple_gpu.yaml \
     --label batch_fit \
     --reporting-level light

The simple config fits only parameters that have direct/basic truth columns in
the prepared dataset and are plausible for a broad-band DSPS fit:

.. code-block:: text

   z_obs
   log10_stellar_mass
   dlog10_sfr_1
   tau2
   dust_index_n

It does not fit Diffstar/Diffmah latents, AGN parameters, gas ionization, or
full SFH latent extensions.

Diffsky Fixed-Redshift Closure
------------------------------

Use this when redshift collapse or degeneracy dominates a free-redshift run:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml \
     fit \
     --limit 128 \
     --batch-size 128 \
     --fit-maxiter 180 \
     --sed-samples 0 \
     --reporting-level light \
     --out outputs/runs/diffsky_hltds_fixedz_closure_n128

This config reads ``redshift_true`` as the fixed catalog redshift and fits:

.. code-block:: text

   log10_stellar_mass
   dlog10_sfr_1
   tau2
   dust_index_n

It is a model/photometry closure diagnostic, not a blind photo-z experiment.

Euclid FS2 Fit
--------------

FS2 remains supported as the Euclid comparison and domain-shift dataset:

.. code-block:: text

   Data/Euclid FS2 LC galaxy catalog_phz1.parquet

One-row smoke:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/fs2_gpu.yaml \
     fit \
     --index 0 \
     --fit-maxiter 20 \
     --sed-samples 1 \
     --out outputs/runs/fs2_gpu_one_short

Small batch:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/fs2_gpu.yaml \
     fit \
     --limit 64 \
     --batch-size 64 \
     --fit-maxiter 200 \
     --sed-samples 0 \
     --reporting-level light \
     --out outputs/runs/fs2_gpu_n64

Amortized FS2 Prior Learning
----------------------------

Train the FS2 amortized model with a learned RealNVP prior:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_fs2_realnvp.yaml \
     amortized-train-fs2 \
     --limit 10000 \
     --batch-size 64 \
     --epochs 5 \
     --n-samples 2 \
     --out outputs/runs/amortized_fs2_realnvp_debug

Run amortized inference:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_fs2_realnvp.yaml \
     amortized-infer-fs2 \
     --checkpoint outputs/runs/amortized_fs2_realnvp_debug/checkpoints/best.eqx \
     --limit 10000 \
     --batch-size 64 \
     --posterior-samples 64 \
     --out outputs/runs/amortized_fs2_realnvp_infer

Outputs
-------

MAP runs write ``normalized_config.json``, fit result tables, photometry
comparison tables, optimizer traces, objective summaries, truth metrics when
truth columns exist, and optional plots under the requested output directory.
Diffsky reports add a compact Markdown summary with truth-recovery tables and
band residuals.
