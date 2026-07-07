"""Shared model parameter contracts."""

from __future__ import annotations

POPCOSMOS_PARAMETER_NAMES = (
    "z_obs",
    "log10_stellar_mass",
    "dlog10_sfr_1",
    "dlog10_sfr_2",
    "dlog10_sfr_3",
    "dlog10_sfr_4",
    "dlog10_sfr_5",
    "dlog10_sfr_6",
    "log10_stellar_metallicity",
    "tau2",
    "dust_index_n",
    "tau1_over_tau2",
    "log10_gas_metallicity",
    "log10_gas_ionization",
    "ln_fagn",
    "ln_tauagn",
)


DIFFSTAR_REDUCED6_PARAMETER_NAMES = (
    "z_obs",
    "log10_stellar_mass",
    "diffstar_lgmcrit",
    "diffstar_lgy_at_mcrit",
    "diffstar_indx_lo",
    "diffstar_lg_qt",
    "diffstar_lg_drop",
    "diffstar_lg_rejuv",
    "log10_stellar_metallicity",
    "tau2",
    "dust_index_n",
    "tau1_over_tau2",
    "log10_gas_metallicity",
    "log10_gas_ionization",
    "ln_fagn",
    "ln_tauagn",
)


DIFFSKY_BASIC_PARAMETER_NAMES = (
    "z_obs",
    "log10_stellar_mass",
    "diffstar_lgmcrit",
    "diffstar_lgy_at_mcrit",
    "diffstar_indx_lo",
    "diffstar_indx_hi",
    "diffstar_lg_qt",
    "diffstar_qlglgdt",
    "diffstar_lg_drop",
    "diffstar_lg_rejuv",
    "diffmah_logm0",
    "diffmah_logtc",
    "diffmah_early_index",
    "diffmah_late_index",
    "diffmah_t_peak",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
)


DIFFSKY_TRUTH_BASIC_PARAMETER_NAMES = (
    "z_obs",
    "log10_stellar_mass",
    "log10_ssfr_at_obs",
    "dust_av",
    "dust_delta",
)


DIFFSTAR_FIXED_PARAMETER_DEFAULTS = {
    "diffstar_indx_hi": -1.0,
    "diffstar_qlglgdt": -0.50725,
}
