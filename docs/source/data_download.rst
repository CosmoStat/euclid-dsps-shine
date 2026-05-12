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
aliases catalog columns to the names consumed by the configs and includes the
new value-added diagnostics used for stellar-mass and photo-z validation:

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
   * - ``phz_median``
     - Default redshift estimate used as DSPS ``z_obs``.
   * - ``phz_min_70``, ``phz_max_70``
     - NNPZ interval used to derive the row-level redshift prior width.
   * - ``z_true_gal``
     - Preferred truth redshift for galaxy-level diagnostics.
   * - ``euclid_vis``, ``euclid_nisp_y``, ``euclid_nisp_j``, ``euclid_nisp_h``
     - Euclid photometry, interpreted as ``Fnu`` in ``erg/s/cm^2/Hz``.
   * - ``sed_cosmos_1``, ``sed_cosmos_2``
     - COSMOS template IDs in local LePhare ``COSMOS_MOD.list`` order.
   * - ``frac_cosmos_1``, ``frac_cosmos_2``
     - Component fractions for COSMOS proxy SED reconstruction. The current
       local parquet contains them, so the default reconstruction policy is
       strict and reports fraction diagnostics.
   * - ``euclid_*_abs``
     - Rest-frame Euclid flux density at 10 parsec, used to normalize the
       COSMOS proxy SED.
   * - ``euclid_*_el_model3_ext*``
     - Forward-modelled Euclid flux target variants for observed-frame branch-2
       diagnostics.
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

Euclid passbands are loaded from ``filters/`` in the Euclid-only config. The
10-band config uses the SciPIC ``value_added_data/filters`` CSV passbands for
LSST and Euclid so the photometry and pseudo-SED resources share the same local
data release.

.. code-block:: text

   filters/Euclid_VIS.vis.dat
   filters/Euclid_NISP.Y.dat
   filters/Euclid_NISP.J.dat
   filters/Euclid_NISP.H.dat

Optional Rest-Frame Flux Columns
--------------------------------

CosmoHub tooltips expose rest-frame Euclid flux columns such as
``euclid_nisp_h_abs`` with the description "rest-frame flux at 10 parsec".

When these ``*_abs`` columns are present in the parquet row, the SED diagnostic
uses them to anchor the rest-frame pseudo-SED directly. If they are absent, the
diagnostic falls back to converting observed fluxes to rest-frame luminosity
density with the luminosity distance.
