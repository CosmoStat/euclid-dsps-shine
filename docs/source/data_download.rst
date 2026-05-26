Data And FSPS Assets
====================

Catalog
-------

The single active config expects the Euclid FS2 PHZ parquet at:

.. code-block:: text

   Data/Euclid FS2 LC galaxy catalog_phz1.parquet

The local query is kept in ``querry.sql`` and targets CosmoHub catalog 353:

.. code-block:: text

   https://cosmohub.pic.es/catalogs/353

It should export LSST ``ugrizy`` and Euclid VIS/Y/J/H continuum fluxes,
per-band flux errors, redshift truth, and basic physical proxy columns. See
:doc:`catalog_columns` for the selected column contract.

Reference SSP
-------------

Download the reference DSPS SSP used to align the generated grids:

.. code-block:: bash

   python scripts/manage_ssp.py download fsps_v0.4.7_u-2.0
   python scripts/manage_ssp.py test Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5

The config path is:

.. code-block:: text

   Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5

FSPS Setup
----------

Generation requires python-FSPS and ``SPS_HOME``:

.. code-block:: bash

   conda activate shine
   export SPS_HOME="$HOME/src/fsps"

   python -c "import fsps; sp=fsps.StellarPopulation(sfh=0); print(len(sp.wavelengths)); print(sp.isoc_library, sp.spec_library)"

The expected local setup reports ``11149`` wavelengths, ``mist`` isochrones,
and ``c3k_a`` spectra.

Gas Grid
--------

Generate the production gas SSP grid:

.. code-block:: bash

   python scripts/generate_fsps_gas_grid.py \
     --output Data/popcosmos_gas_ssp_grid.h5 \
     --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --base-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --overwrite

Validate it:

.. code-block:: bash

   python scripts/generate_fsps_gas_grid.py \
     --output Data/popcosmos_gas_ssp_grid.h5 \
     --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --base-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --validate-only

Expected shape:

.. code-block:: text

   ssp_flux: (7, 7, 12, 107, 11149)

The axes are gas metallicity, gas ionization, stellar metallicity, SSP age, and
wavelength. The spectra use the DSPS ``Lsun/Hz/Msun formed`` convention.

AGN Grid
--------

Generate the AGN template grid:

.. code-block:: bash

   python scripts/generate_fsps_agn_grid.py \
     --output Data/popcosmos_agn_template_grid.h5 \
     --base-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --agn-tau-grid 5 10 20 30 40 60 80 100 150 \
     --fagn-normalization 1.0 \
     --tage-gyr 1.0 \
     --stellar-logzsol 0.0 \
     --overwrite

Validate it:

.. code-block:: bash

   python scripts/generate_fsps_agn_grid.py \
     --output Data/popcosmos_agn_template_grid.h5 \
     --base-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --validate-only

Expected shape:

.. code-block:: text

   template_lnu_per_lbol: (9, 11149)

The AGN grid uses the native FSPS/Nenkova optical depths
``5, 10, 20, 30, 40, 60, 80, 100, 150``. The fit parameter ``ln_tauagn`` is
bounded to ``[ln(5), ln(150)]`` in ``configs/popcosmos_binned.yaml``.

Files Not Tracked
-----------------

All generated ``.h5`` assets, parquet catalogs, and run outputs stay under
``Data/`` or ``outputs/`` and should not be committed.
