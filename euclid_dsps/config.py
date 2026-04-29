"""Configuration loading and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Paths:
    catalog: Path
    ssp: Path


DEFAULT_MODEL_PARAMETERS = {
    "z_obs": 0.5,
    "log10_sfr": 0.0,
    "sfh_t_peak": 4.0,
    "sfh_tau": 0.6,
    "log10_metallicity": -2.0,
    "metallicity_scatter": 0.2,
    "dust_av": 0.2,
    "dust_slope": -0.7,
}

DEFAULT_REDSHIFT_CONFIG = {
    "column": None,
    "truth_column": None,
    "fixed_value": 0.5,
    "min": 1.0e-4,
    "max": 6.0,
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return normalize_config(config)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill lightweight defaults without hiding required paths."""
    config = dict(config)
    config.setdefault("selection", {})
    config.setdefault("redshift", {})
    config.setdefault("model", {})
    config.setdefault("fit", {})
    config.setdefault("eda", {})
    config.setdefault("truth", {})
    config.setdefault("extra_columns", [])

    raw_redshift = dict(config["redshift"] or {})
    redshift = dict(DEFAULT_REDSHIFT_CONFIG)
    redshift.update(raw_redshift)

    config["model"].setdefault("fixed_parameters", {})
    fixed = dict(DEFAULT_MODEL_PARAMETERS)
    fixed.update(config["model"]["fixed_parameters"] or {})
    if "fixed_value" in raw_redshift:
        fixed["z_obs"] = float(redshift["fixed_value"])
    else:
        redshift["fixed_value"] = float(fixed["z_obs"])
    config["redshift"] = redshift
    config["model"]["fixed_parameters"] = fixed
    config["model"].setdefault("parameter_columns", {})
    config["model"].setdefault("n_sfh_bins", 96)

    config["fit"].setdefault(
        "free_parameters",
        {
            "log10_sfr": {"initial": 0.0, "bounds": [-2.5, 3.0]},
            "dust_av": {"initial": 0.2, "bounds": [0.0, 2.5]},
            "log10_metallicity": {"initial": -2.0, "bounds": [-3.0, -1.0]},
        },
    )
    config["fit"].setdefault("method", "jax_adam")
    config["fit"].setdefault("maxiter", 80)
    config["fit"].setdefault("learning_rate", 0.1)
    config["fit"].setdefault("tolerance", 1.0e-5)
    config["fit"].setdefault("patience", 18)
    config["fit"]["population"] = dict(config["fit"].get("population") or {})
    config["fit"]["population"].setdefault("prior_weight", 1.0)
    config["fit"]["population"].setdefault("sigma_floor", 0.03)
    config["fit"]["population"].setdefault("hyper_mu_scale", 5.0)

    config["selection"].setdefault("index", None)
    config["selection"].setdefault("require_positive_flux", True)
    config["selection"].setdefault("sort_by_flux", None)

    config["truth"].setdefault("redshift_column", redshift.get("truth_column"))
    config["truth"].setdefault("parameter_columns", {})

    return config


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve paths relative to the current working directory or config dir."""
    p = Path(path)
    if p.is_absolute():
        return p
    if base_dir is None:
        return p.resolve()
    return (Path(base_dir) / p).resolve()
