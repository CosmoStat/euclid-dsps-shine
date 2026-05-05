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

SUPPORTED_PHOTOMETRY_UNITS = {"fnu_cgs", "abmag", "microjy", "ujy"}
SUPPORTED_FIT_METHODS = {"jax_adam", "jax_adam_vmap", "jax_bfgs"}
SUPPORTED_SAMPLERS = {"nuts", "hmc"}
SUPPORTED_CHAIN_METHODS = {"parallel", "sequential", "vectorized"}
SUPPORTED_TRUTH_TRANSFORMS = {None, "linear", "log10"}


class ConfigValidationError(ValueError):
    """Raised when a run configuration is internally inconsistent."""


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
    config.setdefault("sample", {})
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

    config["sample"] = dict(config["sample"] or {})
    config["sample"].setdefault("num_warmup", 100)
    config["sample"].setdefault("num_samples", 200)
    config["sample"].setdefault("num_chains", 1)
    config["sample"].setdefault("sampler", "nuts")
    config["sample"].setdefault("chain_method", "parallel")
    config["sample"].setdefault("target_accept_prob", 0.85)
    config["sample"].setdefault("max_tree_depth", 10)
    config["sample"].setdefault("num_steps", 8)
    config["sample"].setdefault("dense_mass", False)
    config["sample"].setdefault("jit_model_args", False)
    config["sample"].setdefault("seed", 42)
    config["sample"].setdefault("progress_bar", True)
    config["sample"].setdefault("init_from_map", True)
    config["sample"].setdefault("save_samples", True)
    config["sample"].setdefault("priors", {})

    config["selection"].setdefault("index", None)
    config["selection"].setdefault("require_positive_flux", True)
    config["selection"].setdefault("sort_by_flux", None)

    config["truth"].setdefault("redshift_column", redshift.get("truth_column"))
    config["truth"].setdefault("parameter_columns", {})

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate the normalized runtime configuration.

    Validation intentionally checks structure and scalar contracts only. It does
    not require local data files to exist, so CI can validate configs without
    shipping the private or large FS2 parquet files.
    """
    errors: list[str] = []
    _require_nonempty(config, "catalog_path", errors)
    _require_nonempty(config, "ssp_path", errors)
    _validate_bands(config.get("bands"), errors)
    _validate_redshift(config.get("redshift", {}), errors)
    _validate_model(config.get("model", {}), errors)
    _validate_fit(config.get("fit", {}), errors)
    _validate_sample(config.get("sample", {}), errors)
    _validate_truth(config.get("truth", {}), errors)
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise ConfigValidationError(f"Invalid configuration:\n{detail}")


def validate_catalog_columns(
    config: dict[str, Any], available_columns: set[str] | list[str] | tuple[str, ...]
) -> None:
    """Validate that every configured catalog column exists in a data source."""
    available = set(available_columns)
    missing = [
        column
        for column in _configured_catalog_columns(config)
        if column not in available
    ]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ConfigValidationError(f"Configured catalog columns are missing: {joined}")


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve paths relative to the current working directory or config dir."""
    p = Path(path)
    if p.is_absolute():
        return p
    if base_dir is None:
        return p.resolve()
    return (Path(base_dir) / p).resolve()


def _require_nonempty(config: dict[str, Any], key: str, errors: list[str]) -> None:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty path string")


def _validate_bands(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append("bands must be a non-empty list")
        return
    seen_names: set[str] = set()
    seen_columns: set[str] = set()
    for index, band in enumerate(value):
        if not isinstance(band, dict):
            errors.append(f"bands[{index}] must be a mapping")
            continue
        name = band.get("name")
        column = band.get("column")
        if not isinstance(name, str) or not name:
            errors.append(f"bands[{index}].name must be a non-empty string")
        elif name in seen_names:
            errors.append(f"bands[{index}].name duplicates {name!r}")
        else:
            seen_names.add(name)
        if not isinstance(column, str) or not column:
            errors.append(f"bands[{index}].column must be a non-empty string")
        elif column in seen_columns:
            errors.append(f"bands[{index}].column duplicates {column!r}")
        else:
            seen_columns.add(column)
        units = band.get("units", "fnu_cgs")
        if units not in SUPPORTED_PHOTOMETRY_UNITS:
            errors.append(
                f"bands[{index}].units must be one of {sorted(SUPPORTED_PHOTOMETRY_UNITS)}"
            )
        _positive_float(
            band.get("sigma_mag", 0.05), f"bands[{index}].sigma_mag", errors
        )
        filter_config = band.get("filter", {})
        if filter_config is not None and not isinstance(filter_config, dict):
            errors.append(f"bands[{index}].filter must be a mapping when provided")


def _validate_redshift(redshift: dict[str, Any], errors: list[str]) -> None:
    _optional_string(redshift.get("column"), "redshift.column", errors)
    _optional_string(redshift.get("truth_column"), "redshift.truth_column", errors)
    _finite_float(redshift.get("fixed_value"), "redshift.fixed_value", errors)
    z_min = _finite_float(redshift.get("min"), "redshift.min", errors)
    z_max = _finite_float(redshift.get("max"), "redshift.max", errors)
    if z_min is not None and z_max is not None and z_min >= z_max:
        errors.append("redshift.min must be smaller than redshift.max")


def _validate_model(model: dict[str, Any], errors: list[str]) -> None:
    n_sfh_bins = model.get("n_sfh_bins")
    if not isinstance(n_sfh_bins, int) or n_sfh_bins < 2:
        errors.append("model.n_sfh_bins must be an integer >= 2")
    fixed = model.get("fixed_parameters")
    if not isinstance(fixed, dict):
        errors.append("model.fixed_parameters must be a mapping")
    else:
        for name, value in fixed.items():
            _finite_float(value, f"model.fixed_parameters.{name}", errors)
    parameter_columns = model.get("parameter_columns", {})
    if not isinstance(parameter_columns, dict):
        errors.append("model.parameter_columns must be a mapping")
    else:
        for name, column in parameter_columns.items():
            if not isinstance(name, str) or not isinstance(column, str) or not column:
                errors.append("model.parameter_columns keys and values must be strings")


def _validate_fit(fit: dict[str, Any], errors: list[str]) -> None:
    method = str(fit.get("method", "jax_adam")).lower()
    if method not in SUPPORTED_FIT_METHODS:
        errors.append(f"fit.method must be one of {sorted(SUPPORTED_FIT_METHODS)}")
    _positive_int(fit.get("maxiter"), "fit.maxiter", errors)
    _positive_float(fit.get("learning_rate"), "fit.learning_rate", errors)
    _positive_float(fit.get("tolerance"), "fit.tolerance", errors)
    _positive_int(fit.get("patience"), "fit.patience", errors)
    free = fit.get("free_parameters")
    if not isinstance(free, dict) or not free:
        errors.append("fit.free_parameters must be a non-empty mapping")
        return
    for name, spec in free.items():
        if not isinstance(spec, dict):
            errors.append(f"fit.free_parameters.{name} must be a mapping")
            continue
        bounds = spec.get("bounds")
        if not isinstance(bounds, list | tuple) or len(bounds) != 2:
            errors.append(f"fit.free_parameters.{name}.bounds must contain [min, max]")
            continue
        lower = _finite_float(
            bounds[0], f"fit.free_parameters.{name}.bounds[0]", errors
        )
        upper = _finite_float(
            bounds[1], f"fit.free_parameters.{name}.bounds[1]", errors
        )
        if lower is not None and upper is not None and lower >= upper:
            errors.append(f"fit.free_parameters.{name}.bounds must be increasing")
        initial = spec.get("initial", 0.0)
        if initial != "from_base":
            _finite_float(initial, f"fit.free_parameters.{name}.initial", errors)


def _validate_sample(sample: dict[str, Any], errors: list[str]) -> None:
    sampler = sample.get("sampler")
    if sampler not in SUPPORTED_SAMPLERS:
        errors.append(f"sample.sampler must be one of {sorted(SUPPORTED_SAMPLERS)}")
    chain_method = sample.get("chain_method")
    if chain_method not in SUPPORTED_CHAIN_METHODS:
        errors.append(
            f"sample.chain_method must be one of {sorted(SUPPORTED_CHAIN_METHODS)}"
        )
    for key in (
        "num_warmup",
        "num_samples",
        "num_chains",
        "max_tree_depth",
        "num_steps",
    ):
        _positive_int(sample.get(key), f"sample.{key}", errors)
    target = _finite_float(
        sample.get("target_accept_prob"), "sample.target_accept_prob", errors
    )
    if target is not None and not 0.0 < target < 1.0:
        errors.append("sample.target_accept_prob must be between 0 and 1")
    _finite_float(sample.get("seed"), "sample.seed", errors)


def _validate_truth(truth: dict[str, Any], errors: list[str]) -> None:
    _optional_string(truth.get("redshift_column"), "truth.redshift_column", errors)
    specs = truth.get("parameter_columns", {})
    if not isinstance(specs, dict):
        errors.append("truth.parameter_columns must be a mapping")
        return
    for name, spec in specs.items():
        if isinstance(spec, str):
            continue
        if not isinstance(spec, dict):
            errors.append(f"truth.parameter_columns.{name} must be a string or mapping")
            continue
        _optional_string(
            spec.get("column"), f"truth.parameter_columns.{name}.column", errors
        )
        transform = spec.get("transform")
        if transform not in SUPPORTED_TRUTH_TRANSFORMS:
            errors.append(
                f"truth.parameter_columns.{name}.transform must be one of "
                f"{sorted(str(item) for item in SUPPORTED_TRUTH_TRANSFORMS)}"
            )
        _finite_float(
            spec.get("scale", 1.0), f"truth.parameter_columns.{name}.scale", errors
        )
        _finite_float(
            spec.get("offset", 0.0), f"truth.parameter_columns.{name}.offset", errors
        )


def _configured_catalog_columns(config: dict[str, Any]) -> set[str]:
    from .io import required_catalog_columns

    return set(required_catalog_columns(config))


def _optional_string(value: Any, label: str, errors: list[str]) -> None:
    if value is not None and not isinstance(value, str):
        errors.append(f"{label} must be a string or null")


def _finite_float(value: Any, label: str, errors: list[str]) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be numeric")
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        errors.append(f"{label} must be finite")
        return None
    return result


def _positive_float(value: Any, label: str, errors: list[str]) -> float | None:
    result = _finite_float(value, label, errors)
    if result is not None and result <= 0.0:
        errors.append(f"{label} must be > 0")
    return result


def _positive_int(value: Any, label: str, errors: list[str]) -> int | None:
    if not isinstance(value, int) or value <= 0:
        errors.append(f"{label} must be an integer > 0")
        return None
    return value
