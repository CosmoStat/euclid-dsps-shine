"""Column naming conventions for prepared Diffsky datasets."""

DIFFSKY_LSST_BANDS = ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
DIFFSKY_ROMAN_BANDS = (
    "roman_F062",
    "roman_F087",
    "roman_F106",
    "roman_F129",
    "roman_F146",
    "roman_F158",
    "roman_F184",
    "roman_F213",
)
DIFFSKY_LSST_ROMAN_BANDS = DIFFSKY_LSST_BANDS + DIFFSKY_ROMAN_BANDS

HLTDS_TRUTH_COLUMNS = {
    "redshift_true": "redshift_true",
    "logsm_true": "logsm_obs",
    "logssfr_true": "logssfr_obs",
    "logmp_true": "logmp_obs",
    "logmp_host_true": "logmp_obs_host",
    "central_true": "central",
    "r50_disk_true": "r50_disk",
    "r50_bulge_true": "r50_bulge",
}

HLTDS_DIFFMAH_COLUMNS = (
    "early_index",
    "late_index",
    "logm0",
    "logmp0",
    "logtc",
    "t_peak",
)

HLTDS_DIFFSTAR_COLUMNS = (
    "lgmcrit",
    "lgy_at_mcrit",
    "indx_lo",
    "indx_hi",
    "lg_qt",
    "qlglgdt",
    "lg_drop",
    "lg_rejuv",
)

HLTDS_DUST_COLUMNS = ("av", "delta")
HLTDS_BURST_COLUMNS = ("lgfburst", "lgyr_peak", "lgyr_max")
