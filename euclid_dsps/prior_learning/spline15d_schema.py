"""Dependency-free names for the FENIKS spline-SFH 15D contract."""

N_SPLINE_NODES = 11
PHYSICAL_PARAMETER_NAMES = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
)
SFH_CONTRAST_NAMES = tuple(
    f"sfh_dlog_sfr_{index:02d}" for index in range(1, N_SPLINE_NODES)
)
SPLINE15D_PARAMETER_NAMES = PHYSICAL_PARAMETER_NAMES + SFH_CONTRAST_NAMES
