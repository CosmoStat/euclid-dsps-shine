"""Small helpers describing fit/truth semantics."""

from __future__ import annotations

from typing import Any

from .parameters import DIFFSTAR_REDUCED6_PARAMETER_NAMES, POPCOSMOS_PARAMETER_NAMES

DERIVED_PARAMETERS = {
    "t_obs_gyr",
    "formed_mass_msun",
    "log10_formed_mass_msun",
    "surviving_stellar_mass_msun",
    "log10_surviving_stellar_mass_msun",
    "sfr_at_obs_msun_per_yr",
    "log10_sfr_at_obs",
    *{f"sfr_bin_{index}" for index in range(1, 8)},
    *{f"lookback_bin_edge_{index}" for index in range(0, 8)},
}


def inferred_parameters(config: dict[str, Any]) -> list[str]:
    """Parameters optimized by MAP/MCMC."""
    return sorted((config.get("fit", {}).get("free_parameters") or {}).keys())


def active_parameters(config: dict[str, Any]) -> list[str]:
    """Parameters used by the forward model, whether fixed, injected, or free."""
    model = config.get("model", {}) or {}
    active = set((model.get("fixed_parameters") or {}).keys())
    active.update((model.get("parameter_columns") or {}).keys())
    active.update(inferred_parameters(config))
    active.add("z_obs")
    if model.get("sfh_model") == "popcosmos_bins":
        active = set(POPCOSMOS_PARAMETER_NAMES[:12])
        if model.get("nebular_model") == "gas_grid":
            active.update({"log10_gas_metallicity", "log10_gas_ionization"})
        if model.get("agn_model") == "template_grid":
            active.update({"ln_fagn", "ln_tauagn"})
        return sorted(active)
    if model.get("sfh_model") == "diffstar_reduced6":
        active = set(DIFFSTAR_REDUCED6_PARAMETER_NAMES[:12])
        active.update(
            {
                "diffstar_indx_hi",
                "diffstar_qlglgdt",
            }
        )
        if model.get("nebular_model") == "gas_grid":
            active.update({"log10_gas_metallicity", "log10_gas_ionization"})
        if model.get("agn_model") == "template_grid":
            active.update({"ln_fagn", "ln_tauagn"})
        return sorted(active)
    legacy_active = {
        "z_obs",
        "log10_sfr",
        "sfh_t_peak",
        "sfh_tau",
        "log10_metallicity",
        "metallicity_scatter",
        "dust_av",
        "dust_slope",
        "log10_formed_mass_msun",
        "cosmos_ebv_1",
        "cosmos_ebv_2",
        "cosmos_frac_1",
        "cosmos_frac_2",
        "cosmos_ext_curve_1",
        "cosmos_ext_curve_2",
    }
    active &= legacy_active
    if _uses_cosmos_proxy_dust(config):
        active.discard("dust_av")
        active.discard("dust_slope")
        active.update(
            {
                "cosmos_ebv_1",
                "cosmos_ebv_2",
                "cosmos_frac_1",
                "cosmos_frac_2",
                "cosmos_ext_curve_1",
                "cosmos_ext_curve_2",
            }
        )
    return sorted(active)


def inactive_parameters(config: dict[str, Any]) -> list[str]:
    """Configured or comparable parameters not active in the forward model."""
    model = config.get("model", {}) or {}
    known = set((model.get("fixed_parameters") or {}).keys())
    known.update((model.get("parameter_columns") or {}).keys())
    known.update((config.get("truth", {}).get("parameter_columns") or {}).keys())
    known.update(inferred_parameters(config))
    return sorted(known.difference(active_parameters(config)))


def is_forward_active(config: dict[str, Any], parameter: str) -> bool:
    return parameter in set(active_parameters(config))


def is_inferred(config: dict[str, Any], parameter: str) -> bool:
    return parameter in set(inferred_parameters(config))


def is_comparable_fit_parameter(config: dict[str, Any] | None, parameter: str) -> bool:
    """Return True when fit-vs-truth/proxy plots are scientifically meaningful."""
    if parameter == "dust_av":
        return False
    if parameter in {"z_obs", *DERIVED_PARAMETERS}:
        return True
    if config is None:
        return True
    return is_inferred(config, parameter)


def _uses_cosmos_proxy_dust(config: dict[str, Any]) -> bool:
    if config.get("dust_model") == "cosmos_proxy_fixed":
        return True
    cosmos = config.get("cosmos_sed", {}) or {}
    return bool(cosmos.get("use_cosmos_dust_in_dsps", False))
