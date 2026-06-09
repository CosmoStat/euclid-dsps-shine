Data And FSPS Assets
====================

Catalog
-------

The current validation priority is OpenUniverse/Diffsky. FS2 remains supported
as a comparison and domain-shift diagnostic dataset.

OpenUniverse SkyCatalogs
~~~~~~~~~~~~~~~~~~~~~~~~

OpenUniverse SkyCatalog data is organized by nside=32 HEALPix. For a small
validation subset, provide only the HEALPix ids you want to process:

.. code-block:: text

   Data/openuniverse/raw/galaxy_<hpix>.parquet
   Data/openuniverse/raw/galaxy_flux_<hpix>.parquet
   Data/openuniverse/raw/galaxy_sed_<hpix>.hdf5  # optional

For the public preview S3 layout used in local smoke checks, a single HEALPix
can be downloaded without credentials:

.. code-block:: bash

   aws s3 cp --no-sign-request \
     s3://nasa-irsa-simulations/openuniverse2024/roman/preview/roman_rubin_cats_v1.1.2_faint/galaxy_10307.parquet \
     Data/openuniverse/raw/

   aws s3 cp --no-sign-request \
     s3://nasa-irsa-simulations/openuniverse2024/roman/preview/roman_rubin_cats_v1.1.2_faint/galaxy_flux_10307.parquet \
     Data/openuniverse/raw/

   aws s3 cp --no-sign-request \
     s3://nasa-irsa-simulations/openuniverse2024/roman/preview/roman_rubin_cats_v1.1.2_faint/galaxy_sed_10307.hdf5 \
     Data/openuniverse/raw/

Prepare a compact LSST+Roman 14-band parquet with:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/openuniverse_lsst_roman_14.yaml \
     openuniverse-prepare \
     --input-root Data/openuniverse/raw \
     --hpix 9812 9813 \
     --limit 10000 \
     --out Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet

The command refuses to run without explicit HEALPix ids from ``--hpix`` or
``openuniverse.hpix_ids``. It does not download or scan a full OpenUniverse
release by default.

After preparation, inventory the truth-like fields and optional SED HDF5:

.. code-block:: bash

   python -m euclid_dsps.openuniverse.cli inventory-truth \
     --input Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
     --input-root Data/openuniverse/raw \
     --hpix 10307 \
     --sed \
     --sed-sample-limit 3 \
     --out outputs/reports/openuniverse_truth_inventory_10307

For DSPS fit smoke tests, derive the explicit ``fnu_cgs`` fit-ready parquet:

.. code-block:: bash

   python -m euclid_dsps.openuniverse.cli make-fit-ready \
     --input Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
     --main Data/openuniverse/raw/galaxy_10307.parquet \
     --out Data/openuniverse/processed/ou_lsst_roman_14_subset_fit_ready.parquet

This derived file keeps the public photon fluxes in audit columns, computes
``mu_lensing`` from convergence/shear, divides by magnification in the default
``unlensed`` mode, and writes standard ``flux_*``/``fluxerr_*`` columns in
``fnu_cgs``.

Euclid FS2
~~~~~~~~~~

The FS2 comparison configs expect the Euclid FS2 PHZ parquet at:

.. code-block:: text

   Data/Euclid FS2 LC galaxy catalog_phz1.parquet

The catalog must provide LSST ``ugrizy`` and Euclid VIS/Y/J/H fluxes, flux
errors, redshift truth/proxy columns, and basic physical diagnostics. See
:doc:`catalog_columns` for the column contract.

FSPS Setup
----------

Asset generation requires a local FSPS checkout and python-FSPS in the same
environment:

.. code-block:: bash

   cd "$HOME/src"
   export SPS_HOME="$HOME/src/fsps"
   git clone https://github.com/cconroy20/fsps.git "$SPS_HOME"

   conda activate shine
   python -m pip install fsps

Check the local FSPS build:

.. code-block:: bash

   python -c "import fsps; sp=fsps.StellarPopulation(sfh=0); print(len(sp.wavelengths)); print(sp.isoc_library, sp.spec_library)"

The assets used in the current benchmark were generated with 11149 wavelength
samples, MIST isochrones, C3K spectra, Chabrier IMF, and ``z_sun=0.0142``.

Active Assets
-------------

The production full-AGN path uses:

.. code-block:: text

   Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5
   Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5
   Data/popcosmos_chabrier_gas_ssp_grid.h5
   Data/popcosmos_chabrier_agn_component_ssp_grid.h5
   Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5
   Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5
   Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5

Generated HDF5 files are runtime assets and should not be committed.

What The Files Represent
------------------------

The assets are precomputed so the fit-time DSPS/JAX path does not need to call
FSPS repeatedly:

* the SSP files contain spectra for idealized single-age, single-metallicity
  stellar populations;
* the gas grid repeats those SSP axes over gas metallicity and ionization using
  the FSPS/CLOUDY nebular model;
* the AGN component grid stores the AGN contribution that FSPS adds for each
  ``fagn`` and ``agn_tau`` value;
* the compressed assets store SVD ``basis``, ``coeff``, and ``scale`` arrays
  consumed directly by JAX for production fits;
* all three spectral assets share the same wavelength, stellar age, and stellar
  metallicity axes so they can be combined safely.

Reference SSP
-------------

Generate the fixed-nebular reference SSP:

.. code-block:: bash

   python scripts/generate_fsps_ssp_grid.py \
     --output Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --overwrite

This file is the base axis contract for the PopCosmos-like model. It stores:

.. code-block:: text

   ssp_wave[wave]
   ssp_lg_age_gyr[age]
   ssp_lgmet[stellar_metallicity]
   ssp_flux[stellar_metallicity, age, wave]
   ssp_surviving_mstar[stellar_metallicity, age]

It also carries metadata for IMF, ``z_sun``, FSPS libraries, dust/gas settings,
and units.

Pure-Stellar SSP
----------------

Generate the no-nebular-emission SSP:

.. code-block:: bash

   python scripts/generate_fsps_ssp_grid.py \
     --stellar-only \
     --output Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5 \
     --overwrite

This file has:

.. code-block:: text

   add_neb_emission = 0
   add_neb_continuum = 0
   imf_type = 1
   imf_name = chabrier

It is used by the benchmark ``stellar_only`` and ``stellar_plus_dust`` levels so
those levels are genuinely gas-free.

Gas Grid
--------

Generate the raw FSPS/CLOUDY gas-varying SSP grid:

.. code-block:: bash

   python scripts/generate_fsps_gas_grid.py \
     --output Data/popcosmos_chabrier_gas_ssp_grid.h5 \
     --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --base-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --overwrite

Expected main dataset:

.. code-block:: text

   ssp_flux[gas_metallicity, gas_ionization, stellar_metallicity, age, wave]
   shape: (7, 7, 12, 107, 11149)

The grid axes are:

.. code-block:: text

   gas_lgmet_grid = log10(Zgas/Zsun)
   gas_lgu_grid = log10 U
   ssp_lgmet = log10(Zstar absolute)
   ssp_lg_age_gyr = log10(age/Gyr)
   ssp_wave = Angstrom

The current production config uses ``emission_line_corrections: none``. That is
raw FSPS/CLOUDY, not official PopCosmos line-calibrated gas.

AGN Component Grid
------------------

Generate the FSPS-native AGN component grid:

.. code-block:: bash

   python scripts/generate_fsps_agn_component_grid.py \
     --output Data/popcosmos_chabrier_agn_component_ssp_grid.h5 \
     --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --overwrite

Expected main dataset:

.. code-block:: text

   agn_lnu_per_mformed[fagn, agn_tau, stellar_metallicity, age, wave]

The generator computes the component as:

.. code-block:: text

   FSPS(fagn, agn_tau) - FSPS(fagn=0)

for each age and metallicity. This is the validated AGN asset for the current
full model. The older ``popcosmos_chabrier_agn_template_grid.h5`` is a legacy
diagnostic template and is not the default science path.

Compressed Runtime Assets
-------------------------

After the dense SSP, gas, and AGN assets exist, build the compressed production
assets:

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

These files are the assets used by ``configs/popcosmos_binned_compressed.yaml``
and ``configs/popcosmos_diffstar_compressed.yaml``. See
:doc:`ssp_compression` for the SVD format and benchmark status.

Validation Commands
-------------------

Validate existing files without importing python-FSPS:

.. code-block:: bash

   python scripts/generate_fsps_ssp_grid.py \
     --output Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --validate-only

   python scripts/generate_fsps_ssp_grid.py \
     --stellar-only \
     --output Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5 \
     --validate-only

   python scripts/generate_fsps_gas_grid.py \
     --output Data/popcosmos_chabrier_gas_ssp_grid.h5 \
     --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --base-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --validate-only

   python scripts/generate_fsps_agn_component_grid.py \
     --output Data/popcosmos_chabrier_agn_component_ssp_grid.h5 \
     --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --validate-only

Validate compressed files:

.. code-block:: bash

   python scripts/validate_compressed_spectral_asset.py \
     Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5

   python scripts/validate_compressed_spectral_asset.py \
     Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5

   python scripts/validate_compressed_spectral_asset.py \
     Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5

Files To Avoid
--------------

Do not use old Kroupa-named assets for PopCosmos-like configs. Do not use the
legacy AGN template grid for the current validated full-AGN path unless you are
explicitly reproducing an old diagnostic run.
