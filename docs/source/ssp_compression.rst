SSP, Gas, And AGN Compression
=============================

Purpose
-------

The compression implemented in this repository is runtime compression, not
disk compression. The goal is to reduce the spectral tensors that are resident
in JAX/GPU memory so larger galaxy batches fit on the same GPU.

HDF5 gzip or shuffle compression can make files smaller on disk, but it does
not help if ``load_context`` expands the file into a dense ``float32`` tensor
before JAX sees it. The compressed configs therefore load compact arrays
directly:

.. code-block:: text

   basis[k, wave]
   coeff[physical_axes..., k]
   scale[physical_axes...]

The public FS2 config loads compact arrays directly and is the recommended
path for high-throughput FS2 MAP fits.

Recommended Configs
-------------------

Use the FS2 GPU config for production FS2 batches:

.. code-block:: text

   configs/fs2_gpu.yaml

The Diffsky HLTDS simple MAP configs can use the HLTDS SSP file directly. The
active Diffsky amortized configs use a compressed basis built from that same
HLTDS SSP file:

.. code-block:: text

   Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/diffsky_hltds_cosmos_260215_04_14_2026_ssp_basis_k64_coeff16.hdf5

They do not use the compressed FS2 gas/AGN runtime assets.

Compressed Assets
-----------------

The active FS2 config uses:

.. code-block:: yaml

   runtime: gpu

   fit:
     trace_mode: optimizer
     trace_interval: 20

   model:
     ssp_model: compressed_basis
     compressed_ssp_path: Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5
     nebular_model: compressed_gas_grid
     compressed_gas_grid_path: Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5
     agn_model: compressed_fsps_component_grid
     compressed_agn_component_grid_path: Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5

These assets are generated from the dense Chabrier FSPS assets. They keep the
same wavelength, age, metallicity, gas, and AGN axes, but replace the repeated
dense wavelength axis by a low-rank basis.

How The SVD Compression Works
-----------------------------

For the base SSP cube, the dense tensor is:

.. code-block:: text

   ssp_flux[Zstar, age, wave]

The compressor flattens ``Zstar`` and ``age`` into one curve axis:

.. code-block:: text

   F[curve, wave]

Each curve is normalized before SVD:

.. code-block:: text

   scale[curve] = norm(F[curve, :])
   F_norm[curve, wave] = F[curve, wave] / scale[curve]

SVD then finds a wavelength basis ordered by decreasing singular value:

.. code-block:: text

   F_norm ~= U_k S_k V_k.T

The HDF5 asset stores:

.. code-block:: text

   basis = V_k.T
   coeff = U_k S_k
   scale = scale

At runtime DSPS reconstructs only the metallicity-specific slice needed by the
current forward pass:

.. code-block:: python

   coeff_z = interpolate_Zstar(ssp_coeff, lgmet_abs)      # [age, k]
   scale_z = interpolate_Zstar(ssp_scale, lgmet_abs)      # [age]
   weighted_coeff = coeff_z * scale_z[:, None]
   ssp_flux_z = jnp.einsum("ak,kw->aw", weighted_coeff, ssp_basis)

This gives ``ssp_flux_z[age, wave]`` without loading the full
``ssp_flux[Zstar, age, wave]`` tensor into JAX.

Gas Compression
---------------

The dense gas grid is much larger:

.. code-block:: text

   ssp_flux[Zgas, Ugas, Zstar, age, wave]

where:

* ``Zgas`` is ``log10_gas_metallicity = log10(Zgas / Zsun)``;
* ``Ugas`` is ``log10_gas_ionization = log10 U``;
* ``Zstar`` is the stellar metallicity axis;
* ``age`` is the SSP age axis;
* ``wave`` is the wavelength axis.

The compressed gas asset stores:

.. code-block:: text

   gas_basis[k, wave]
   gas_coeff[Zgas, Ugas, Zstar, age, k]
   gas_scale[Zgas, Ugas, Zstar, age]

DSPS interpolates in gas metallicity, gas ionization, and stellar metallicity
in coefficient space, then expands only the requested ``[age, wave]`` slice.
This avoids materializing the dense gas grid or four dense interpolation
corners.

Gas spectra contain narrow emission lines, so the gas basis is harder to
compress than a smooth stellar continuum. The current production compromise is
``k=64`` with mixed precision storage.

AGN Compression
---------------

The dense FSPS-native AGN component grid is:

.. code-block:: text

   agn_lnu_per_mformed[fagn, tauagn, Zstar, age, wave]

The fitted AGN parameters are:

.. code-block:: text

   ln_fagn    -> fagn = exp(ln_fagn)
   ln_tauagn  -> tauagn = exp(ln_tauagn)

The AGN grid is effectively linear in ``fagn`` for the active asset family. The
compressed format therefore removes the ``fagn`` axis from storage and applies
``fagn`` as a runtime multiplier:

.. code-block:: text

   agn_basis[k, wave]
   agn_coeff[tauagn, Zstar, age, k]
   agn_scale[tauagn, Zstar, age]

Runtime reconstruction:

.. code-block:: text

   agn_flux[age, wave]
     = fagn * agn_scale[tauagn, Zstar, age]
            * agn_coeff[tauagn, Zstar, age, k] @ agn_basis[k, wave]

This is why AGN compression is especially effective: the dense ``fagn`` axis
does not need to be resident on GPU.

Why SVD, Not Wavelets
---------------------

A sparse Haar-wavelet benchmark was added in
``scripts/benchmark_wavelet_compression.py``. It compares the current SVD-style
compressed assets to an oracle sparse Haar representation that stores the top
wavelet coefficients and their indices per spectrum.

The result was:

* AGN: current fagn-factored SVD is much smaller and much more accurate.
* Gas: current SVD ``k64`` mixed precision is smaller and more accurate than
  the tested sparse Haar candidates.
* Stellar SSP: Haar is competitive on a sampled median spectral metric, but its
  worst-tail error is worse and it has no validated JAX sparse runtime path.

Decision: keep SVD/factored-SVD as the production compression method. Wavelets
remain a research option only after photometric closure and JAX sparse-layout
benchmarks.

Build Compressed Assets
-----------------------

The dense input assets must already exist. Then build:

.. code-block:: bash

   python scripts/build_compressed_ssp_grid.py \
     --input Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --output Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5 \
     --k 64 --basis-dtype float32 --coeff-dtype float16 --overwrite

   python scripts/build_compressed_gas_grid.py \
     --input Data/popcosmos_chabrier_gas_ssp_grid.h5 \
     --output Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5 \
     --k 64 --basis-dtype float16 --coeff-dtype float16 --overwrite

   python scripts/build_compressed_agn_component_grid.py \
     --input Data/popcosmos_chabrier_agn_component_ssp_grid.h5 \
     --output Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5 \
     --k 12 --factor-fagn --basis-dtype float32 --coeff-dtype float16 --overwrite

Validate compressed assets:

.. code-block:: bash

   python scripts/validate_compressed_spectral_asset.py \
     Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5

   python scripts/validate_compressed_spectral_asset.py \
     Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5

   python scripts/validate_compressed_spectral_asset.py \
     Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5

Production Fit Command
----------------------

Use the FS2 GPU config for large MAP batches:

.. code-block:: bash

   conda activate shine

   python -m euclid_dsps.cli \
     --config configs/fs2_gpu.yaml \
     fit \
     --limit 1000 \
     --batch-size 128 \
     --fit-maxiter 200 \
     --out outputs/runs/fs2_gpu_map_n1000_bs128 \
     --sed-samples 0 \
     --reporting-level light

If the target GPU is smaller, reduce ``--batch-size`` first. The compressed
config already requests the GPU runtime preset and ``TF_GPU_ALLOCATOR`` setting
used in the local successful runs.

Checks
------

Run the lightweight checks after changing compression code or configs:

.. code-block:: bash

   python -m compileall euclid_dsps scripts
   pytest tests/test_config.py
   pytest tests/test_model.py

For scientific closure, run dense-vs-compressed and FSPS/Prospector benchmarks
before using a new compressed asset family for final science.
