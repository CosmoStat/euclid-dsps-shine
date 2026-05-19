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
    "sfh_burst_fraction": 0.0,
    "sfh_burst_time": 1.0,
    "sfh_burst_width": 0.12,
    "sfh_quench_time": 12.0,
    "sfh_quench_width": 0.5,
    "sfh_quench_depth": 0.0,
    "log10_metallicity": -2.0,
    "metallicity_scatter": 0.2,
    "dust_av": 0.2,
    "dust_slope": -0.7,
    "cosmos_ebv_1": 0.0,
    "cosmos_ebv_2": 0.0,
    "cosmos_frac_1": 0.5,
    "cosmos_frac_2": 0.5,
    "cosmos_ext_curve_1": 0.0,
    "cosmos_ext_curve_2": 0.0,
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
SUPPORTED_TRUTH_TRANSFORMS = {None, "linear", "log10", "log_stellar_mass_h2_to_msun"}
SUPPORTED_PRIOR_TYPES = {
    "uniform",
    "normal",
    "truncated_normal",
    "scaled_beta",
    "phz_interval",
}
SUPPORTED_FILTER_RESPONSE_KINDS = {"photon", "energy"}
SUPPORTED_COMPONENT_FRACTION_POLICIES = {"strict", "equal_if_missing"}
SUPPORTED_COSMOS_PHOTOMETRY_TARGET_SETS = {
    "continuum_internal_dust",
    "emission_lines_internal_dust",
    "emission_lines_internal_dust_mw",
    "noisy_observation",
}
SUPPORTED_REPORTING_LEVELS = {"full", "light"}
SUPPORTED_OUTPUT_FORMATS = {"csv", "parquet", "both"}

DEFAULT_RUNTIME_CONFIG = {
    "jax_platforms": "cpu",
    "disable_jax_plugin_autoload": True,
    "xla_python_client_preallocate": False,
    "require_gpu": False,
    "expected_gpu_name": None,
    "jax_compilation_cache_dir": None,
    "jax_persistent_cache_min_compile_time_secs": 1.0,
}

DEFAULT_COSMOS_SED_CONFIG = {
    "lephare_data_dir": "~/.cache/lephare/data",
    "template_subdir": "sed/GAL/COSMOS_SED",
    "template_list": "COSMOS_MOD.list",
    "expected_template_count": 31,
    "template_wave_unit": "angstrom",
    "template_flux_unit": "arbitrary_flambda",
    "value_added_data_dir": None,
    "catalog_h": None,
    "extinction_dir": "ext",
    "extinction": {
        "curves": {
            0: "none",
            1: "SMC_prevot",
            2: "SB_calzetti",
            3: "SB_calzetti_bump1",
            4: "SB_calzetti_bump2",
        }
    },
    "component_fraction_policy": "strict",
    "filter_response_kind": "photon",
    "comparison_wave_min_angstrom": 1000.0,
    "comparison_wave_max_angstrom": 30000.0,
    "sample_plot_count": 12,
    "observed_photometry_target_sets": ["continuum_internal_dust"],
    "normalization_bands": [],
    "use_cosmos_dust_in_dsps": False,
}


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
    config.setdefault("runtime", {})
    config.setdefault("reporting", {})
    config.setdefault("output", {})
    config.setdefault("extra_columns", [])
    config.setdefault("cosmos_sed", {})

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
    config["fit"].setdefault("prior_weight", 1.0)
    config["fit"].setdefault("fast_warmstart_only", False)
    config["fit"].setdefault("fast_grid_search", False)
    config["fit"].setdefault("redshift_grid_size", 5)
    config["fit"].setdefault("redshift_grid_width", 0.4)
    config["fit"].setdefault(
        "fast_grid_parameters",
        ["z_obs", "log10_metallicity", "sfh_t_peak", "sfh_tau"],
    )
    config["fit"].setdefault("fast_grid_prior_width", 1.0)
    config["fit"].setdefault("priors", {})
    config["fit"]["population"] = dict(config["fit"].get("population") or {})
    config["fit"]["population"].setdefault("prior_weight", 1.0)
    config["fit"]["population"].setdefault("sigma_floor", 0.03)
    config["fit"]["population"].setdefault("hyper_mu_scale", 5.0)
    config["fit"]["population"].setdefault("relations", {})

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

    runtime = dict(DEFAULT_RUNTIME_CONFIG)
    runtime.update(dict(config["runtime"] or {}))
    config["runtime"] = runtime

    config["reporting"] = dict(config["reporting"] or {})
    config["reporting"].setdefault("level", "full")
    config["output"] = dict(config["output"] or {})
    config["output"].setdefault("format", "both")
    config["output"].setdefault("verbose_benchmark", False)

    cosmos_sed = dict(DEFAULT_COSMOS_SED_CONFIG)
    raw_cosmos_sed = dict(config["cosmos_sed"] or {})
    raw_extinction = raw_cosmos_sed.pop("extinction", None)
    cosmos_sed.update(raw_cosmos_sed)
    extinction = dict(DEFAULT_COSMOS_SED_CONFIG["extinction"])
    if raw_extinction is not None:
        extinction.update(dict(raw_extinction or {}))
    cosmos_sed["extinction"] = extinction
    config["cosmos_sed"] = cosmos_sed

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
    _validate_runtime(config.get("runtime", {}), errors)
    _validate_reporting(config.get("reporting", {}), errors)
    _validate_output(config.get("output", {}), errors)
    _validate_cosmos_sed(config.get("cosmos_sed", {}), errors)
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
        _optional_string(
            band.get("error_column"), f"bands[{index}].error_column", errors
        )
        error_units = band.get("error_units", units)
        if error_units not in SUPPORTED_PHOTOMETRY_UNITS:
            errors.append(
                f"bands[{index}].error_units must be one of {sorted(SUPPORTED_PHOTOMETRY_UNITS)}"
            )
        if band.get("sigma_mag_floor") is not None:
            _positive_float(
                band.get("sigma_mag_floor"), f"bands[{index}].sigma_mag_floor", errors
            )
        if band.get("sigma_mag_ceiling") is not None:
            _positive_float(
                band.get("sigma_mag_ceiling"),
                f"bands[{index}].sigma_mag_ceiling",
                errors,
            )


def _validate_redshift(redshift: dict[str, Any], errors: list[str]) -> None:
    _optional_string(redshift.get("column"), "redshift.column", errors)
    _optional_string(redshift.get("truth_column"), "redshift.truth_column", errors)
    _finite_float(redshift.get("fixed_value"), "redshift.fixed_value", errors)
    z_min = _finite_float(redshift.get("min"), "redshift.min", errors)
    z_max = _finite_float(redshift.get("max"), "redshift.max", errors)
    if z_min is not None and z_max is not None and z_min >= z_max:
        errors.append("redshift.min must be smaller than redshift.max")
    interval = redshift.get("prior_interval")
    if interval is not None:
        _validate_redshift_interval(interval, "redshift.prior_interval", errors)
    intervals = redshift.get("prior_intervals")
    if intervals is not None:
        if not isinstance(intervals, list):
            errors.append("redshift.prior_intervals must be a list when provided")
        else:
            for index, item in enumerate(intervals):
                _validate_redshift_interval(
                    item, f"redshift.prior_intervals[{index}]", errors
                )


def _validate_redshift_interval(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be a mapping when provided")
        return
    _optional_string(value.get("min_column"), f"{label}.min_column", errors)
    _optional_string(value.get("max_column"), f"{label}.max_column", errors)
    probability = _finite_float(
        value.get("probability", 0.70), f"{label}.probability", errors
    )
    if probability is not None and not 0.0 < probability < 1.0:
        errors.append(f"{label}.probability must be in (0, 1)")


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
    _positive_float(fit.get("prior_weight", 1.0), "fit.prior_weight", errors)
    if not isinstance(fit.get("fast_warmstart_only", False), bool):
        errors.append("fit.fast_warmstart_only must be a boolean")
    if not isinstance(fit.get("fast_grid_search", False), bool):
        errors.append("fit.fast_grid_search must be a boolean")
    _positive_int(fit.get("redshift_grid_size", 5), "fit.redshift_grid_size", errors)
    _positive_float(
        fit.get("redshift_grid_width", 0.4), "fit.redshift_grid_width", errors
    )
    fast_grid_parameters = fit.get("fast_grid_parameters", [])
    if not isinstance(fast_grid_parameters, list) or not all(
        isinstance(name, str) and name for name in fast_grid_parameters
    ):
        errors.append("fit.fast_grid_parameters must be a list of parameter names")
    _positive_float(
        fit.get("fast_grid_prior_width", 1.0), "fit.fast_grid_prior_width", errors
    )
    _validate_population_config(fit.get("population", {}), errors)
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
    _validate_fit_priors(fit.get("priors", {}), free, errors)


def _validate_fit_priors(
    priors: Any, free_parameters: dict[str, Any], errors: list[str]
) -> None:
    if not isinstance(priors, dict):
        errors.append("fit.priors must be a mapping")
        return
    for name, spec in priors.items():
        if name not in free_parameters:
            errors.append(f"fit.priors.{name} must match a free parameter")
            continue
        if not isinstance(spec, dict):
            errors.append(f"fit.priors.{name} must be a mapping")
            continue
        prior_type = str(spec.get("type", "normal"))
        if prior_type not in SUPPORTED_PRIOR_TYPES:
            errors.append(
                f"fit.priors.{name}.type must be one of {sorted(SUPPORTED_PRIOR_TYPES)}"
            )
        if "loc" in spec and spec["loc"] != "from_base":
            _finite_float(spec["loc"], f"fit.priors.{name}.loc", errors)
        if "scale" in spec and spec["scale"] != "from_base":
            _positive_float(spec["scale"], f"fit.priors.{name}.scale", errors)
        if prior_type == "scaled_beta":
            _positive_float(spec.get("alpha", 1.0), f"fit.priors.{name}.alpha", errors)
            _positive_float(spec.get("beta", 1.0), f"fit.priors.{name}.beta", errors)
        if prior_type == "phz_interval":
            _positive_float(
                spec.get("tail_scale", 0.05), f"fit.priors.{name}.tail_scale", errors
            )
            _positive_float(
                spec.get("weight", 1.0), f"fit.priors.{name}.weight", errors
            )


def _validate_population_config(population: Any, errors: list[str]) -> None:
    if not isinstance(population, dict):
        errors.append("fit.population must be a mapping")
        return
    _positive_float(
        population.get("prior_weight", 1.0), "fit.population.prior_weight", errors
    )
    _positive_float(
        population.get("sigma_floor", 0.03), "fit.population.sigma_floor", errors
    )
    _positive_float(
        population.get("hyper_mu_scale", 5.0), "fit.population.hyper_mu_scale", errors
    )
    relations = population.get("relations", {})
    if not isinstance(relations, dict):
        errors.append("fit.population.relations must be a mapping")
        return
    for target, spec in relations.items():
        if not isinstance(target, str) or not target:
            errors.append("fit.population.relations keys must be parameter names")
            continue
        if not isinstance(spec, dict):
            errors.append(f"fit.population.relations.{target} must be a mapping")
            continue
        _optional_string(
            spec.get("predictor"),
            f"fit.population.relations.{target}.predictor",
            errors,
        )
        for key in (
            "pivot",
            "intercept_initial",
            "slope_initial",
            "sigma_initial",
            "slope_scale",
        ):
            if key in spec and spec[key] != "median":
                label = f"fit.population.relations.{target}.{key}"
                if key in {"sigma_initial", "slope_scale"}:
                    _positive_float(spec[key], label, errors)
                else:
                    _finite_float(spec[key], label, errors)


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
    _validate_sample_priors(sample.get("priors", {}), errors)


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
        if transform == "log_stellar_mass_h2_to_msun":
            _positive_float(spec.get("h"), f"truth.parameter_columns.{name}.h", errors)


def _validate_runtime(runtime: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(runtime, dict):
        errors.append("runtime must be a mapping")
        return
    platforms = runtime.get("jax_platforms")
    if not isinstance(platforms, str) or not platforms.strip():
        errors.append("runtime.jax_platforms must be a non-empty string")
    for key in (
        "disable_jax_plugin_autoload",
        "xla_python_client_preallocate",
        "require_gpu",
    ):
        if not isinstance(runtime.get(key), bool):
            errors.append(f"runtime.{key} must be a boolean")
    _optional_string(
        runtime.get("expected_gpu_name"), "runtime.expected_gpu_name", errors
    )
    _optional_string(
        runtime.get("jax_compilation_cache_dir"),
        "runtime.jax_compilation_cache_dir",
        errors,
    )
    cache_min = runtime.get("jax_persistent_cache_min_compile_time_secs")
    if cache_min is not None:
        _positive_float(
            cache_min, "runtime.jax_persistent_cache_min_compile_time_secs", errors
        )


def _validate_reporting(reporting: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(reporting, dict):
        errors.append("reporting must be a mapping")
        return
    if reporting.get("level") not in SUPPORTED_REPORTING_LEVELS:
        errors.append(
            f"reporting.level must be one of {sorted(SUPPORTED_REPORTING_LEVELS)}"
        )


def _validate_output(output: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(output, dict):
        errors.append("output must be a mapping")
        return
    if output.get("format") not in SUPPORTED_OUTPUT_FORMATS:
        errors.append(
            f"output.format must be one of {sorted(SUPPORTED_OUTPUT_FORMATS)}"
        )
    if not isinstance(output.get("verbose_benchmark"), bool):
        errors.append("output.verbose_benchmark must be a boolean")


def _validate_cosmos_sed(cosmos_sed: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(cosmos_sed, dict):
        errors.append("cosmos_sed must be a mapping")
        return
    for key in (
        "lephare_data_dir",
        "template_subdir",
        "template_list",
        "template_wave_unit",
        "template_flux_unit",
        "extinction_dir",
    ):
        if not isinstance(cosmos_sed.get(key), str) or not cosmos_sed.get(key):
            errors.append(f"cosmos_sed.{key} must be a non-empty string")
    _optional_string(
        cosmos_sed.get("value_added_data_dir"),
        "cosmos_sed.value_added_data_dir",
        errors,
    )
    if cosmos_sed.get("catalog_h") is not None:
        _positive_float(cosmos_sed.get("catalog_h"), "cosmos_sed.catalog_h", errors)
    expected = cosmos_sed.get("expected_template_count")
    if expected is not None:
        _positive_int(expected, "cosmos_sed.expected_template_count", errors)
    response = cosmos_sed.get("filter_response_kind")
    if response not in SUPPORTED_FILTER_RESPONSE_KINDS:
        errors.append(
            "cosmos_sed.filter_response_kind must be one of "
            f"{sorted(SUPPORTED_FILTER_RESPONSE_KINDS)}"
        )
    fraction_policy = cosmos_sed.get("component_fraction_policy")
    if fraction_policy not in SUPPORTED_COMPONENT_FRACTION_POLICIES:
        errors.append(
            "cosmos_sed.component_fraction_policy must be one of "
            f"{sorted(SUPPORTED_COMPONENT_FRACTION_POLICIES)}"
        )
    _finite_float(
        cosmos_sed.get("comparison_wave_min_angstrom"),
        "cosmos_sed.comparison_wave_min_angstrom",
        errors,
    )
    _finite_float(
        cosmos_sed.get("comparison_wave_max_angstrom"),
        "cosmos_sed.comparison_wave_max_angstrom",
        errors,
    )
    _positive_int(
        cosmos_sed.get("sample_plot_count"), "cosmos_sed.sample_plot_count", errors
    )
    target_sets = cosmos_sed.get("observed_photometry_target_sets", [])
    if not isinstance(target_sets, list):
        errors.append("cosmos_sed.observed_photometry_target_sets must be a list")
    elif not all(isinstance(item, str) and item for item in target_sets):
        errors.append(
            "cosmos_sed.observed_photometry_target_sets entries must be non-empty strings"
        )
    else:
        unknown = sorted(set(target_sets) - SUPPORTED_COSMOS_PHOTOMETRY_TARGET_SETS)
        if unknown:
            errors.append(
                "cosmos_sed.observed_photometry_target_sets contains unsupported "
                f"entries: {unknown}"
            )
    if not isinstance(cosmos_sed.get("use_cosmos_dust_in_dsps", False), bool):
        errors.append("cosmos_sed.use_cosmos_dust_in_dsps must be a boolean")
    extinction = cosmos_sed.get("extinction")
    if not isinstance(extinction, dict):
        errors.append("cosmos_sed.extinction must be a mapping")
    else:
        curves = extinction.get("curves")
        if not isinstance(curves, dict) or not curves:
            errors.append("cosmos_sed.extinction.curves must be a non-empty mapping")
        else:
            for code, curve in curves.items():
                try:
                    int(code)
                except (TypeError, ValueError):
                    errors.append("cosmos_sed.extinction.curves keys must be integers")
                if not isinstance(curve, str) or not curve:
                    errors.append(
                        f"cosmos_sed.extinction.curves.{code} must be a non-empty string"
                    )
    normalization_bands = cosmos_sed.get("normalization_bands", [])
    if normalization_bands is None:
        return
    if not isinstance(normalization_bands, list):
        errors.append("cosmos_sed.normalization_bands must be a list")
        return
    for index, item in enumerate(normalization_bands):
        if not isinstance(item, dict):
            errors.append(f"cosmos_sed.normalization_bands[{index}] must be a mapping")
            continue
        for key in ("band_name", "target_column"):
            if not isinstance(item.get(key), str) or not item.get(key):
                errors.append(
                    f"cosmos_sed.normalization_bands[{index}].{key} "
                    "must be a non-empty string"
                )


def _validate_sample_priors(priors: Any, errors: list[str]) -> None:
    if not isinstance(priors, dict):
        errors.append("sample.priors must be a mapping")
        return
    for name, spec in priors.items():
        if not isinstance(spec, dict):
            errors.append(f"sample.priors.{name} must be a mapping")
            continue
        prior_type = str(spec.get("type", "truncated_normal"))
        if prior_type not in SUPPORTED_PRIOR_TYPES:
            errors.append(
                f"sample.priors.{name}.type must be one of "
                f"{sorted(SUPPORTED_PRIOR_TYPES)}"
            )
        if "loc" in spec and spec["loc"] != "from_base":
            _finite_float(spec["loc"], f"sample.priors.{name}.loc", errors)
        if "scale" in spec and spec["scale"] != "from_base":
            _positive_float(spec["scale"], f"sample.priors.{name}.scale", errors)
        if prior_type == "scaled_beta":
            _positive_float(
                spec.get("alpha", 1.0), f"sample.priors.{name}.alpha", errors
            )
            _positive_float(spec.get("beta", 1.0), f"sample.priors.{name}.beta", errors)
        if prior_type == "phz_interval":
            _positive_float(
                spec.get("tail_scale", 0.05),
                f"sample.priors.{name}.tail_scale",
                errors,
            )
            _positive_float(
                spec.get("weight", 1.0), f"sample.priors.{name}.weight", errors
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
