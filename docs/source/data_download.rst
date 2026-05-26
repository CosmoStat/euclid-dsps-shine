Data Download
=============

Catalog Source
--------------

The default workflow expects the Euclid FS2 PHZ parquet file at:

.. code-block:: text

   Data/Euclid FS2 LC galaxy catalog_phz1.parquet

The data used by the default configuration was queried from CosmoHub catalog
353:

.. code-block:: text

   https://cosmohub.pic.es/catalogs/353

``Data/`` is local runtime state and is ignored by git. Keep large parquet
files, SSP templates, downloaded DSPS assets, and local data manifests there.

CosmoHub SQL Query
------------------

Run the repository SQL in CosmoHub and export the result as parquet. The query
aliases catalog columns to the names consumed by the configs and includes
continuum fluxes, rest-frame fluxes, survey-like fluxes, flux errors, and noisy
realizations for LSST ``ugrizy`` plus Euclid VIS/Y/J/H:

.. literalinclude:: ../../querry.sql
   :language: sql

Column Contract
---------------

See :doc:`catalog_columns` for the complete documented subset selected by the
SQL query, including source names, aliases, units, and project usage notes.

The default config requires these canonical columns:

.. list-table::
   :header-rows: 1

   * - Column
     - Purpose
   * - ``z_true_gal``
     - Preferred truth redshift for galaxy-level diagnostics.
   * - ``lsst_u`` ... ``lsst_y``, ``euclid_vis`` ... ``euclid_nisp_h``
     - Continuum photometry interpreted as ``Fnu`` in ``erg/s/cm^2/Hz`` and
       used by the active DSPS fit.
   * - ``*_el_model3_ext_odonnell_ext_error``
     - Per-band flux uncertainties used as the likelihood denominator.
   * - ``sed_cosmos_1``, ``sed_cosmos_2``
     - COSMOS template IDs in local LePhare ``COSMOS_MOD.list`` order.
   * - ``frac_cosmos_1``, ``frac_cosmos_2``
     - Component fractions for COSMOS proxy SED reconstruction. The current
       local parquet contains them, so the default reconstruction policy is
       strict and reports fraction diagnostics.
   * - ``*_abs``
     - Rest-frame flux density at 10 parsec, used for COSMOS proxy SED
       diagnostics and normalization checks.
   * - ``*_el_model3_ext*``
     - Forward-modelled flux target variants for observed-frame diagnostics.
   * - ``metallicity_true``
     - Gas-phase oxygen abundance truth. Reports convert it to a metallicity proxy with ``offset: -10.61``.
   * - ``log_stellar_mass``
     - Stellar mass in ``log10(Msun h^-2)``. Reports convert it to
       ``log10(Msun)`` using the configured catalog ``h`` value.
   * - ``log_sfr_true``
     - Catalog log SFR truth. Reports compare it with derived ``log10_sfr_at_obs``.
   * - ``dust_ebv_true``
     - Intrinsic dust color-excess proxy. Reports convert it to ``A_V`` with ``scale: 4.05``.

Local DSPS Assets
-----------------

Download small DSPS smoke-test assets into ``Data/``:

.. code-block:: bash

   euclid-dsps --config configs/smoke_test.yaml download-assets --out Data

The production FS2 config expects:

.. code-block:: text

   Data/ssp_data_fsps_v3.2_lgmet_age.h5

The active 10-band config loads passbands from the repository ``filters/``
directory:

.. code-block:: text

   filters/LSST_LSST.u.dat
   filters/LSST_LSST.g.dat
   filters/LSST_LSST.r.dat
   filters/LSST_LSST.i.dat
   filters/LSST_LSST.z.dat
   filters/LSST_LSST.y.dat
   filters/Euclid_VIS.vis.dat
   filters/Euclid_NISP.Y.dat
   filters/Euclid_NISP.J.dat
   filters/Euclid_NISP.H.dat

Optional Rest-Frame Flux Columns
--------------------------------

CosmoHub tooltips expose rest-frame flux columns such as ``lsst_u_abs`` and
``euclid_nisp_h_abs`` with the description "rest-frame flux at 10 parsec".

When these ``*_abs`` columns are present in the parquet row, the SED diagnostic
uses them to anchor the rest-frame pseudo-SED directly. If they are absent, the
diagnostic falls back to converting observed fluxes to rest-frame luminosity
density with the luminosity distance.
