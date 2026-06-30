Data And Assets
===============

Directory Contract
------------------

Runtime data lives under ``Data/`` and generated reports under ``outputs/``.
The repository does not download large survey products by default.

The current public paths are:

.. code-block:: text

   Data/Euclid FS2 LC galaxy catalog_phz1.parquet
   Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/
   Data/diffsky/raw/hltds_cosmos_260215_03_31_2026/
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr_projected_truth.parquet
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr_projected_truth_nokl_trainval20k.parquet
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet
   Data/diffsky/processed/hltds_cosmos_260215_03_31_2026_zmax335_m5depth.parquet
   Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5
   Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5
   Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5
   Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5
   Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5

Diffsky HLTDS Downloader
------------------------

The main validation dataset is:

.. code-block:: text

   https://portal.nersc.gov/cfs/hacc/aphearin/diffsky_data/hltds_cosmos_260215_04_14_2026/

First list the remote directory. This downloads only the HTML listing:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-list-remote \
     --url https://portal.nersc.gov/cfs/hacc/aphearin/diffsky_data/hltds_cosmos_260215_04_14_2026/ \
     --max-depth 1 \
     --out outputs/diffsky_hltds_04_14_listing.json

Rank likely useful files:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-inventory-remote \
     --listing outputs/diffsky_hltds_04_14_listing.json \
     --out outputs/diffsky_hltds_04_14_candidates.csv

Download a bounded subset. The command requires ``--yes`` and enforces the
``--max-files`` and ``--max-total-gb`` limits:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-download-subset \
     --listing outputs/diffsky_hltds_04_14_listing.json \
     --out-dir Data/diffsky/raw/hltds_cosmos_260215_04_14_2026 \
     --max-files 12 \
     --max-total-gb 2 \
     --include diffsky_gals \
     --include param \
     --include ssp \
     --include transmission \
     --include t_table \
     --include yaml \
     --yes

Inspect local HDF5 files:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-inventory-local \
     --root Data/diffsky/raw/hltds_cosmos_260215_04_14_2026 \
     --out outputs/diffsky_hltds_04_14_local_inventory.json

Build the full normalized source parquet with deterministic synthetic
``m5_depth`` photometric errors:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-prepare-dataset \
     --raw-root Data/diffsky/raw/hltds_cosmos_260215_04_14_2026 \
     --inventory outputs/diffsky_hltds_04_14_local_inventory.json \
     --out Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet \
     --error-model m5_depth

Then build the current continuous-redshift low-z intermediate with the explicit
flux-dependent error model:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-redshift-subset \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet \
     --out Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet \
     --redshift-min 0.0 \
     --redshift-max 0.35 \
     --error-model m5_depth

Add DSPS projected-truth columns to make the default modeling dataset:

.. code-block:: bash

   conda activate shine
   python scripts/build_diffsky_lowz_projected_truth_dataset.py --force

This writes
``Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr_projected_truth.parquet``
with ``78651`` rows in the current local build.

Build the canonical truth-rich high-redshift subset with the same
``m5_depth`` error contract:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-redshift-subset \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_03_31_2026_photometry_truth.parquet \
     --out Data/diffsky/processed/hltds_cosmos_260215_03_31_2026_zmax335_m5depth.parquet \
     --redshift-min 0.0 \
     --redshift-max 3.35 \
     --error-model m5_depth

This is the preferred dataset when generated truth is the reference population.
The local 03/31 source currently reaches ``z = 3.0319715``, so the ``3.35`` cut
keeps all available high-redshift rows while making the intended upper bound
explicit.

Validate readiness for prior-learning experiments:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-validate-dataset \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet \
     --manifest Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.manifest.yaml \
     --out outputs/reports/diffsky_hltds_04_14/prior_learning_validation_report.md

Write dataset diagnostics:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-dataset-diagnostics \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet \
     --manifest Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.manifest.yaml \
     --out outputs/reports/diffsky_hltds_04_14/dataset

The prepared files keep native HLTDS AB magnitudes and truth columns such as
``redshift_true``, ``logsm_true``, ``logssfr_true``, ``logsfr_true``,
``logmp_true``, central flags, and size proxies when they are present. The
source parquet and low-z intermediate both create ``fluxerr_*`` columns with
the deterministic ``m5_depth`` model and a PhotErr-style
``sigma_sys_mag=0.005`` systematic floor. The default projected-truth parquet
adds DSPS truth columns and is the default input for Diffsky training, MAP,
and posterior-predictive diagnostics.

For the exact ``m5_depth``/``photo_err``/PhotErr-style formula, the separate
2% likelihood floor, and the current ``worst100`` huge-error-bar failure
examples, see ``diffsky_dataset.rst`` under "Photometry Contract".

Diffsky SSP Asset
-----------------

The Diffsky simple configs use the SSP file downloaded with the HLTDS subset:

.. code-block:: text

   Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/diffsky_hltds_cosmos_260215_04_14_2026_ssp_data.hdf5

This asset does not carry the same PopCosmos metadata as the locally generated
FSPS assets, so the Diffsky configs set ``model.asset_metadata_policy:
permissive``. The science contract is explicit: this is a simple DSPS recovery
test against HLTDS photometry and direct/basic truth columns, not a claim that
the DSPS parameterization exactly matches the HLTDS generator.

For the amortized Diffsky path, build a lightweight compressed SSP basis from
the downloaded HLTDS dense SSP before training:

.. code-block:: bash

   python scripts/build_compressed_ssp_grid.py \
     --input Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/diffsky_hltds_cosmos_260215_04_14_2026_ssp_data.hdf5 \
     --output Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/diffsky_hltds_cosmos_260215_04_14_2026_ssp_basis_k64_coeff16.hdf5 \
     --k 64 \
     --basis-dtype float32 \
     --coeff-dtype float16 \
     --overwrite

The compressed file keeps physical axes at source precision and stores only the
large spectral basis/coefficient payload in reduced precision. The public
amortized Diffsky config expects:

.. code-block:: text

   Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/diffsky_hltds_cosmos_260215_04_14_2026_ssp_basis_k64_coeff16.hdf5

Euclid FS2 Data
---------------

FS2 remains the Euclid comparison dataset. The GPU config expects:

.. code-block:: text

   Data/Euclid FS2 LC galaxy catalog_phz1.parquet

The catalog must provide LSST ``ugrizy`` and Euclid VIS/Y/J/H fluxes and flux
errors. See :doc:`catalog_columns` for the FS2 column contract.

FS2 SSP Runtime Assets
----------------------

``configs/fs2_gpu.yaml`` uses compressed runtime assets:

.. code-block:: text

   Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5
   Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5
   Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5

These are generated from local FSPS grids. If they already exist, validate
them:

.. code-block:: bash

   python scripts/validate_compressed_spectral_asset.py \
     Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5

   python scripts/validate_compressed_spectral_asset.py \
     Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5

   python scripts/validate_compressed_spectral_asset.py \
     Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5

Generating those assets requires FSPS and is separate from downloading the
Diffsky HLTDS sample. Keep generated HDF5 assets out of source control.
