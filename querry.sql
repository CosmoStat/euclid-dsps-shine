SELECT
  -- coordinates
  `ra_gal`,
  `dec_gal`,
  `ra_mag_gal`,
  `dec_mag_gal`,

  -- redshift labels
  `true_redshift_halo` AS `z_true`,
  `phz_mode_1` AS `z_phz`,
  `true_redshift_gal` AS `z_true_gal`,
  `observed_redshift_gal` AS `z_obs_gal`,
  `redshift_step`,
  `zp` AS `z_deepz`,
  `phz_flags`,
  `phz_median`,
  `phz_min_70`,
  `phz_max_70`,
  `phz_min_90`,
  `phz_max_90`,
  `phz_min_95`,
  `phz_max_95`,
  `phz_mode_1_area`,
  `phz_mode_2`,
  `phz_mode_2_area`,

  -- COSMOS SED latent templates
  `sed_cosmos_1`,
  `sed_cosmos_2`,
  `frac_cosmos_1`,
  `frac_cosmos_2`,
  `color_kind`,

  -- Euclid VIS
  `euclid_vis`,
  `euclid_vis_abs`,
  `euclid_vis_el_model3_ext`,
  `euclid_vis_el_model3_ext_odonnell_ext`,
  `euclid_vis_el_model3_ext_odonnell_ext_error`,
  `euclid_vis_el_model3_ext_odonnell_ext_error_realization`,

  -- Euclid NISP-Y
  `euclid_nisp_y`,
  `euclid_nisp_y_abs`,
  `euclid_nisp_y_el_model3_ext`,
  `euclid_nisp_y_el_model3_ext_odonnell_ext`,
  `euclid_nisp_y_el_model3_ext_odonnell_ext_error`,
  `euclid_nisp_y_el_model3_ext_odonnell_ext_error_realization`,

  -- Euclid NISP-J
  `euclid_nisp_j`,
  `euclid_nisp_j_abs`,
  `euclid_nisp_j_el_model3_ext`,
  `euclid_nisp_j_el_model3_ext_odonnell_ext`,
  `euclid_nisp_j_el_model3_ext_odonnell_ext_error`,
  `euclid_nisp_j_el_model3_ext_odonnell_ext_error_realization`,

  -- Euclid NISP-H
  `euclid_nisp_h`,
  `euclid_nisp_h_abs`,
  `euclid_nisp_h_el_model3_ext`,
  `euclid_nisp_h_el_model3_ext_odonnell_ext`,
  `euclid_nisp_h_el_model3_ext_odonnell_ext_error`,
  `euclid_nisp_h_el_model3_ext_odonnell_ext_error_realization`,

  -- LSST photometry
  `lsst_u`,
  `lsst_g`,
  `lsst_r`,
  `lsst_i`,
  `lsst_z`,
  `lsst_y`,

  -- physical truth labels
  `log_stellar_mass`,
  `log_ml_r01`,
  `abs_mag_r01`,
  `log_luminosity_r01`,
  `abs_mag_uv_unextincted`,
  `metallicity` AS `metallicity_true`,
  `log_sfr` AS `log_sfr_true`,
  POW(10, `log_sfr`) AS `sfr_true`,

  -- COSMOS dust / extinction components
  `ebv_cosmos_1`,
  `ebv_cosmos_2`,
  `ext_curve_cosmos_1`,
  `ext_curve_cosmos_2`,
  `mw_extinction`,

  -- one scalar intrinsic dust target
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

  -- required SED info
  AND `sed_cosmos_1` IS NOT NULL
  AND `sed_cosmos_2` IS NOT NULL

  -- required Euclid continuum photometry
  AND `euclid_vis` IS NOT NULL
  AND `euclid_nisp_y` IS NOT NULL
  AND `euclid_nisp_j` IS NOT NULL
  AND `euclid_nisp_h` IS NOT NULL

  -- required Euclid absolute/rest-frame fluxes
  AND `euclid_vis_abs` IS NOT NULL
  AND `euclid_nisp_y_abs` IS NOT NULL
  AND `euclid_nisp_j_abs` IS NOT NULL
  AND `euclid_nisp_h_abs` IS NOT NULL

  -- required Euclid full observed fluxes
  AND `euclid_vis_el_model3_ext_odonnell_ext` IS NOT NULL
  AND `euclid_nisp_y_el_model3_ext_odonnell_ext` IS NOT NULL
  AND `euclid_nisp_j_el_model3_ext_odonnell_ext` IS NOT NULL
  AND `euclid_nisp_h_el_model3_ext_odonnell_ext` IS NOT NULL

  -- required Euclid errors / noisy realizations
  AND `euclid_vis_el_model3_ext_odonnell_ext_error` IS NOT NULL
  AND `euclid_nisp_y_el_model3_ext_odonnell_ext_error` IS NOT NULL
  AND `euclid_nisp_j_el_model3_ext_odonnell_ext_error` IS NOT NULL
  AND `euclid_nisp_h_el_model3_ext_odonnell_ext_error` IS NOT NULL

  AND `euclid_vis_el_model3_ext_odonnell_ext_error_realization` IS NOT NULL
  AND `euclid_nisp_y_el_model3_ext_odonnell_ext_error_realization` IS NOT NULL
  AND `euclid_nisp_j_el_model3_ext_odonnell_ext_error_realization` IS NOT NULL
  AND `euclid_nisp_h_el_model3_ext_odonnell_ext_error_realization` IS NOT NULL

  -- required redshift labels
  AND `phz_mode_1` IS NOT NULL
  AND `true_redshift_halo` IS NOT NULL
  AND `true_redshift_gal` IS NOT NULL
  AND `observed_redshift_gal` IS NOT NULL
  AND `zp` IS NOT NULL
  AND `phz_flags` IS NOT NULL
  AND `phz_median` IS NOT NULL
  AND `phz_min_70` IS NOT NULL
  AND `phz_max_70` IS NOT NULL
  AND `phz_min_90` IS NOT NULL
  AND `phz_max_90` IS NOT NULL
  AND `phz_min_95` IS NOT NULL
  AND `phz_max_95` IS NOT NULL
  AND `phz_mode_1_area` IS NOT NULL

  -- required physical labels
  AND `log_stellar_mass` IS NOT NULL
  AND `log_ml_r01` IS NOT NULL
  AND `metallicity` IS NOT NULL
  AND `log_sfr` IS NOT NULL

  -- required COSMOS dust information
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
