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

Run this query in CosmoHub and export the result as parquet. The query aliases
the catalog columns to the names consumed by ``configs/fs2_phz1.yaml``:

.. code-block:: sql

   SELECT
     -- coordinates
     `ra_gal`,
     `dec_gal`,
     `ra_mag_gal`,
     `dec_mag_gal`,

     -- redshift labels
     `true_redshift_halo` AS `z_true`,
     `phz_mode_1` AS `z_phz`,

     -- Euclid photometry
     `euclid_vis`,
     `euclid_nisp_y`,
     `euclid_nisp_j`,
     `euclid_nisp_h`,

     -- LSST photometry
     `lsst_u`,
     `lsst_g`,
     `lsst_r`,
     `lsst_i`,
     `lsst_z`,
     `lsst_y`,

     -- physical truth labels
     `metallicity` AS `metallicity_true`,
     `log_sfr` AS `log_sfr_true`,
     POW(10, `log_sfr`) AS `sfr_true`,

     -- dust raw components
     `ebv_cosmos_1`,
     `ebv_cosmos_2`,
     `ext_curve_cosmos_1`,
     `ext_curve_cosmos_2`,
     `mw_extinction`,

     -- one scalar intrinsic dust target, computed in SQL
     CASE
       WHEN `bulge_fraction` IS NOT NULL
         AND `ebv_cosmos_1` IS NOT NULL
         AND `ebv_cosmos_2` IS NOT NULL
       THEN
         `bulge_fraction` * `ebv_cosmos_1`
         + (1.0 - `bulge_fraction`) * `ebv_cosmos_2`

       WHEN `ebv_cosmos_1` IS NOT NULL
         AND `ebv_cosmos_2` IS NULL
       THEN `ebv_cosmos_1`

       WHEN `ebv_cosmos_1` IS NULL
         AND `ebv_cosmos_2` IS NOT NULL
       THEN `ebv_cosmos_2`

       WHEN `ebv_cosmos_1` IS NOT NULL
         AND `ebv_cosmos_2` IS NOT NULL
       THEN 0.5 * (`ebv_cosmos_1` + `ebv_cosmos_2`)

       ELSE NULL
     END AS `dust_ebv_true`,

     -- morphology
     `bulge_fraction`,
     `disk_r50`,
     `bulge_r50`,
     `eps1_gal`,
     `eps2_gal`,
     `disk_ellipticity`,
     `bulge_ellipticity`,
     `bulge_nsersic`,
     `disk_nsersic`,

     -- halo properties
     `lm_halo`,
     `lmbound_halo`,
     `r_halo`,
     `x_halo`,
     `y_halo`,
     `z_halo`,
     `vx_halo`,
     `vy_halo`,
     `vz_halo`,
     `n_sats_halo`,
     `num_p_halo`,
     `conc_vir_halo`,
     `rs_halo`,
     `rvir_halo`

   FROM
     euclid_fs2_mock_dr_v1_1_phz

   WHERE
     `ra_gal` > 230
     AND `ra_gal` < 232
     AND `dec_gal` > 65
     AND `dec_gal` < 66

     -- required photometry
     AND `euclid_vis` IS NOT NULL
     AND `euclid_nisp_y` IS NOT NULL
     AND `euclid_nisp_j` IS NOT NULL
     AND `euclid_nisp_h` IS NOT NULL

     -- required redshift labels
     AND `phz_mode_1` IS NOT NULL
     AND `true_redshift_halo` IS NOT NULL

     -- required physical labels
     AND `metallicity` IS NOT NULL
     AND `log_sfr` IS NOT NULL

     -- required dust information
     AND (
       `ebv_cosmos_1` IS NOT NULL
       OR `ebv_cosmos_2` IS NOT NULL
     )

     -- safe morphology range for weighted dust
     AND (
       `bulge_fraction` IS NULL
       OR (
         `bulge_fraction` >= 0.0
         AND `bulge_fraction` <= 1.0
       )
     )

Column Contract
---------------

The default config requires these canonical columns:

.. list-table::
   :header-rows: 1

   * - Column
     - Purpose
   * - ``z_phz``
     - Fixed redshift used as DSPS ``z_obs``.
   * - ``z_true``
     - Truth redshift used only in diagnostics.
   * - ``euclid_vis``, ``euclid_nisp_y``, ``euclid_nisp_j``, ``euclid_nisp_h``
     - Euclid photometry, interpreted as ``Fnu`` in ``erg/s/cm^2/Hz``.
   * - ``metallicity_true``
     - Gas-phase oxygen abundance truth. Reports convert it to a metallicity proxy with ``offset: -10.61``.
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

Euclid passbands are loaded from ``filters/``. The default FS2 setup uses the
ASCII passbands:

.. code-block:: text

   filters/Euclid_VIS.vis.dat
   filters/Euclid_NISP.Y.dat
   filters/Euclid_NISP.J.dat
   filters/Euclid_NISP.H.dat
