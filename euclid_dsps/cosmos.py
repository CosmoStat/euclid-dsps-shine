"""COSMOS-template pseudo-ground-truth SED reconstruction.

The Flagship catalog does not contain wavelength-by-wavelength spectra. This
module reconstructs an approximate, template-level proxy from COSMOS template
IDs, E(B-V), extinction-curve codes, and optional component fractions. The
output is a diagnostic reference SED, not exact physical SPS truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .filters import FilterCurve
from .model import ModelResult

C_ANGSTROM_PER_S = 2.99792458e18
L_SUN_ERG_PER_S = 3.828e33
PC_CM = 3.0856775814913673e18
TEN_PC_CM = 10.0 * PC_CM
COSMOS_COMPONENT_COLUMNS = (
    "sed_cosmos_1",
    "sed_cosmos_2",
    "ebv_cosmos_1",
    "ebv_cosmos_2",
    "ext_curve_cosmos_1",
    "ext_curve_cosmos_2",
)
COSMOS_FRACTION_COLUMNS = ("frac_cosmos_1", "frac_cosmos_2")
EUCLID_BAND_NAMES = (
    "euclid_vis",
    "euclid_nisp_y",
    "euclid_nisp_j",
    "euclid_nisp_h",
)


class CosmosSedError(ValueError):
    """Raised when COSMOS-template SED reconstruction cannot continue."""


class MissingCosmosColumnsError(CosmosSedError):
    """Raised when catalog columns required by COSMOS SED reconstruction are absent."""


class MissingCosmosResourceError(FileNotFoundError):
    """Raised when LePhare template or extinction files are absent."""


@dataclass(frozen=True)
class CosmosTemplate:
    """One COSMOS template from ``COSMOS_MOD.list``."""

    template_id: int
    name: str
    path: Path
    wave_angstrom: np.ndarray
    flambda: np.ndarray


@dataclass(frozen=True)
class ExtinctionCurve:
    """One LePhare extinction curve ``k(lambda)``."""

    name: str
    path: Path
    wave_angstrom: np.ndarray
    k_lambda: np.ndarray


@dataclass(frozen=True)
class CosmosSedResources:
    """Loaded COSMOS templates and extinction-curve mapping."""

    templates: tuple[CosmosTemplate, ...]
    extinction_curves: dict[str, ExtinctionCurve]
    extinction_mapping: dict[int, str]
    template_list_path: Path
    extinction_dir: Path


@dataclass(frozen=True)
class CosmosSedResult:
    """Reconstructed and Euclid-absolute-flux-normalized proxy SED."""

    row_index: int
    wave_angstrom: np.ndarray
    flambda_unscaled: np.ndarray
    flambda_scaled: np.ndarray
    alpha: float
    synthetic_abs_fluxes_before: dict[str, float]
    synthetic_abs_fluxes_after: dict[str, float]
    catalog_abs_fluxes: dict[str, float]
    residuals_vs_catalog_abs: dict[str, float]
    relative_residuals_vs_catalog_abs: dict[str, float]
    diagnostics: dict[str, Any]


def resolve_lephare_data_dir(cosmos_config: dict[str, Any]) -> Path:
    """Resolve LePhare auxiliary-data root.

    Priority is explicit config, ``LEPHAREDIR``, then the LePhare default cache
    used by recent installations.
    """
    configured = cosmos_config.get("lephare_data_dir")
    value = configured or os.environ.get("LEPHAREDIR") or "~/.cache/lephare/data"
    return Path(str(value)).expanduser().resolve()


def load_cosmos_sed_resources(cosmos_config: dict[str, Any]) -> CosmosSedResources:
    """Load COSMOS templates and LePhare extinction curves."""
    templates = load_cosmos_templates(cosmos_config)
    extinction_mapping, extinction_curves, extinction_dir = load_extinction_curves(
        cosmos_config
    )
    template_list_path = _template_list_path(cosmos_config)
    return CosmosSedResources(
        templates=templates,
        extinction_curves=extinction_curves,
        extinction_mapping=extinction_mapping,
        template_list_path=template_list_path,
        extinction_dir=extinction_dir,
    )


def load_cosmos_templates(cosmos_config: dict[str, Any]) -> tuple[CosmosTemplate, ...]:
    """Load templates in exact ``COSMOS_MOD.list`` order.

    Catalog IDs ``0..N-1`` map directly to the line order in the list file.
    """
    list_path = _template_list_path(cosmos_config)
    if not list_path.exists():
        raise MissingCosmosResourceError(
            "COSMOS template list not found. "
            f"Searched: {list_path}. Configure cosmos_sed.lephare_data_dir, "
            "cosmos_sed.template_subdir, or cosmos_sed.template_list."
        )
    raw_entries = [
        line.split()[0]
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    templates = []
    for template_id, entry in enumerate(raw_entries):
        path = _resolve_template_entry(entry, list_path, cosmos_config)
        data = _load_two_column_file(path, f"COSMOS template {template_id}")
        wave = data[:, 0] * _wave_unit_factor(cosmos_config.get("template_wave_unit"))
        flambda = data[:, 1].astype(float)
        mask = np.isfinite(wave) & np.isfinite(flambda) & (wave > 0)
        order = np.argsort(wave[mask])
        templates.append(
            CosmosTemplate(
                template_id=template_id,
                name=Path(entry).name,
                path=path,
                wave_angstrom=wave[mask][order],
                flambda=np.clip(flambda[mask][order], 0.0, np.inf),
            )
        )

    expected = cosmos_config.get("expected_template_count")
    if expected is not None and len(templates) != int(expected):
        raise MissingCosmosResourceError(
            f"COSMOS template count mismatch: expected {int(expected)}, "
            f"loaded {len(templates)} from {list_path}."
        )
    return tuple(templates)


def load_extinction_curves(
    cosmos_config: dict[str, Any],
) -> tuple[dict[int, str], dict[str, ExtinctionCurve], Path]:
    """Load configured LePhare extinction curves from ``$LEPHAREDIR/ext``."""
    lephare_dir = resolve_lephare_data_dir(cosmos_config)
    extinction_dir = (
        lephare_dir / str(cosmos_config.get("extinction_dir", "ext"))
    ).resolve()
    mapping = _normalized_extinction_mapping(cosmos_config)
    curves: dict[str, ExtinctionCurve] = {}
    for curve_name in sorted(set(mapping.values())):
        if curve_name == "none":
            continue
        path = _resolve_extinction_file(extinction_dir, curve_name)
        if not path.exists():
            raise MissingCosmosResourceError(
                "LePhare extinction curve not found. "
                f"Curve: {curve_name}. Searched: {path}. "
                "Configure cosmos_sed.extinction.curves or "
                "cosmos_sed.extinction_dir."
            )
        data = _load_two_column_file(path, f"extinction curve {curve_name}")
        wave = data[:, 0]
        k_lambda = data[:, 1]
        mask = np.isfinite(wave) & np.isfinite(k_lambda) & (wave > 0)
        order = np.argsort(wave[mask])
        curves[curve_name] = ExtinctionCurve(
            name=curve_name,
            path=path,
            wave_angstrom=wave[mask][order],
            k_lambda=k_lambda[mask][order],
        )
    return mapping, curves, extinction_dir


def reconstruct_cosmos_proxy_sed(
    row: dict[str, Any] | pd.Series,
    row_index: int,
    resources: CosmosSedResources,
    filters: dict[str, FilterCurve],
    band_configs: list[dict[str, Any]],
    cosmos_config: dict[str, Any],
) -> CosmosSedResult:
    """Reconstruct, attenuate, combine, and normalize one COSMOS proxy SED."""
    row_dict = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    missing = [column for column in COSMOS_COMPONENT_COLUMNS if column not in row_dict]
    if missing:
        raise MissingCosmosColumnsError(
            "Missing COSMOS SED columns: " + ", ".join(sorted(missing))
        )

    template_1 = resources.templates[
        _template_id_from_row(row_dict, "sed_cosmos_1", len(resources.templates))
    ]
    template_2 = resources.templates[
        _template_id_from_row(row_dict, "sed_cosmos_2", len(resources.templates))
    ]
    wave = template_1.wave_angstrom
    comp_1 = apply_cosmos_extinction(
        wave,
        template_1.flambda,
        _row_float(row_dict, "ebv_cosmos_1"),
        _curve_code_from_row(row_dict, "ext_curve_cosmos_1"),
        resources,
    )
    comp_2_raw = np.interp(
        wave,
        template_2.wave_angstrom,
        template_2.flambda,
        left=0.0,
        right=0.0,
    )
    comp_2 = apply_cosmos_extinction(
        wave,
        comp_2_raw,
        _row_float(row_dict, "ebv_cosmos_2"),
        _curve_code_from_row(row_dict, "ext_curve_cosmos_2"),
        resources,
    )
    f1, f2, fraction_diagnostics = component_fractions(row_dict, cosmos_config)
    flambda_unscaled = f1 * comp_1 + f2 * comp_2
    norm = normalize_cosmos_sed(
        wave,
        flambda_unscaled,
        row_dict,
        filters,
        band_configs,
        cosmos_config,
    )
    diagnostics = {
        "row_index": int(row_index),
        "sed_cosmos_1": int(template_1.template_id),
        "sed_cosmos_2": int(template_2.template_id),
        "template_name_1": template_1.name,
        "template_name_2": template_2.name,
        "ext_curve_cosmos_1": _curve_code_from_row(row_dict, "ext_curve_cosmos_1"),
        "ext_curve_cosmos_2": _curve_code_from_row(row_dict, "ext_curve_cosmos_2"),
        "extinction_name_1": resources.extinction_mapping[
            _curve_code_from_row(row_dict, "ext_curve_cosmos_1")
        ],
        "extinction_name_2": resources.extinction_mapping[
            _curve_code_from_row(row_dict, "ext_curve_cosmos_2")
        ],
        "ebv_cosmos_1": _row_float(row_dict, "ebv_cosmos_1"),
        "ebv_cosmos_2": _row_float(row_dict, "ebv_cosmos_2"),
        "color_kind": _optional_row_float(row_dict, "color_kind"),
        "z_true": _optional_row_float(row_dict, "z_true"),
        "z_phz": _optional_row_float(row_dict, "z_phz"),
        "metallicity_true": _optional_row_float(row_dict, "metallicity_true"),
        "log_sfr_true": _optional_row_float(row_dict, "log_sfr_true"),
        "dust_ebv_true": _optional_row_float(row_dict, "dust_ebv_true"),
        **fraction_diagnostics,
    }
    return CosmosSedResult(
        row_index=int(row_index),
        wave_angstrom=wave,
        flambda_unscaled=flambda_unscaled,
        flambda_scaled=flambda_unscaled * norm["alpha"],
        alpha=norm["alpha"],
        synthetic_abs_fluxes_before=norm["synthetic_before"],
        synthetic_abs_fluxes_after=norm["synthetic_after"],
        catalog_abs_fluxes=norm["target"],
        residuals_vs_catalog_abs=norm["residual"],
        relative_residuals_vs_catalog_abs=norm["relative_residual"],
        diagnostics=diagnostics,
    )


def apply_cosmos_extinction(
    wave_angstrom: np.ndarray,
    flambda: np.ndarray,
    ebv: float,
    curve_code: int,
    resources: CosmosSedResources,
) -> np.ndarray:
    """Apply ``F_att = F * 10**(-0.4 E(B-V) k(lambda))``."""
    flux = np.asarray(flambda, dtype=float)
    if not np.isfinite(ebv) or ebv <= 0:
        return flux.copy()
    curve_name = resources.extinction_mapping.get(int(curve_code))
    if curve_name is None:
        raise CosmosSedError(
            f"Unsupported ext_curve_cosmos code {curve_code}; configured codes: "
            f"{sorted(resources.extinction_mapping)}"
        )
    if curve_name == "none":
        return flux.copy()
    curve = resources.extinction_curves.get(curve_name)
    if curve is None:
        raise CosmosSedError(f"Extinction curve {curve_name!r} was not loaded.")
    k_lambda = np.interp(
        wave_angstrom,
        curve.wave_angstrom,
        curve.k_lambda,
        left=curve.k_lambda[0],
        right=curve.k_lambda[-1],
    )
    return flux * 10.0 ** (-0.4 * float(ebv) * k_lambda)


def component_fractions(
    row: dict[str, Any], cosmos_config: dict[str, Any]
) -> tuple[float, float, dict[str, Any]]:
    """Return normalized component fractions with explicit missing-column policy."""
    policy = str(cosmos_config.get("component_fraction_policy", "strict"))
    missing = [column for column in COSMOS_FRACTION_COLUMNS if column not in row]
    if missing:
        if policy == "strict":
            raise MissingCosmosColumnsError(
                "Missing COSMOS SED fraction columns: " + ", ".join(missing)
            )
        if policy != "equal_if_missing":
            raise CosmosSedError(
                f"Unsupported component_fraction_policy for missing columns: {policy}"
            )
        return (
            0.5,
            0.5,
            {
                "frac_cosmos_1": np.nan,
                "frac_cosmos_2": np.nan,
                "frac_sum_raw": np.nan,
                "frac_cosmos_1_used": 0.5,
                "frac_cosmos_2_used": 0.5,
                "component_fraction_policy_used": "equal_if_missing",
                "missing_fraction_columns": ",".join(missing),
            },
        )

    f1 = _row_float(row, "frac_cosmos_1")
    f2 = _row_float(row, "frac_cosmos_2")
    if not np.isfinite(f1) or not np.isfinite(f2) or f1 < 0 or f2 < 0:
        raise CosmosSedError(
            "Invalid COSMOS component fractions: "
            f"frac_cosmos_1={f1}, frac_cosmos_2={f2}"
        )
    frac_sum = f1 + f2
    if not np.isfinite(frac_sum) or frac_sum <= 0:
        raise CosmosSedError(
            "COSMOS component fractions must have finite positive sum: "
            f"frac_sum={frac_sum}"
        )
    f1_used = f1 / frac_sum
    f2_used = f2 / frac_sum
    return (
        f1_used,
        f2_used,
        {
            "frac_cosmos_1": f1,
            "frac_cosmos_2": f2,
            "frac_sum_raw": frac_sum,
            "frac_cosmos_1_used": f1_used,
            "frac_cosmos_2_used": f2_used,
            "component_fraction_policy_used": (
                "normalized" if not np.isclose(frac_sum, 1.0) else "as_catalog"
            ),
            "missing_fraction_columns": "",
        },
    )


def synthetic_fnu_from_flambda(
    wave_angstrom: np.ndarray,
    flambda_cgs_per_angstrom: np.ndarray,
    filter_curve: FilterCurve,
    response_kind: str = "photon",
) -> float:
    """Integrate an ``F_lambda`` SED through a filter and return mean ``Fnu``.

    ``response_kind='photon'`` uses the common photon-counting AB convention:

    ``<Fnu> = integral(lambda F_lambda T dlambda) / integral(c/lambda T dlambda)``.

    ``response_kind='energy'`` uses an energy-response convention:

    ``<Fnu> = integral(F_lambda T dlambda) / integral(c/lambda^2 T dlambda)``.
    """
    wave = np.asarray(wave_angstrom, dtype=float)
    flambda = np.asarray(flambda_cgs_per_angstrom, dtype=float)
    filt_wave = np.asarray(filter_curve.wave, dtype=float)
    trans = np.asarray(filter_curve.transmission, dtype=float)
    response = response_kind.strip().lower()
    if response not in {"photon", "energy"}:
        raise CosmosSedError(f"Unsupported filter response kind: {response_kind}")
    interp = np.interp(filt_wave, wave, flambda, left=0.0, right=0.0)
    mask = (
        np.isfinite(filt_wave)
        & np.isfinite(trans)
        & np.isfinite(interp)
        & (filt_wave > 0)
        & (trans > 0)
    )
    if mask.sum() < 2:
        return float("nan")
    x = filt_wave[mask]
    t = trans[mask]
    y = interp[mask]
    if response == "photon":
        numerator = np.trapezoid(y * t * x, x)
        denominator = np.trapezoid(t * C_ANGSTROM_PER_S / x, x)
    else:
        numerator = np.trapezoid(y * t, x)
        denominator = np.trapezoid(t * C_ANGSTROM_PER_S / x**2, x)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def normalize_cosmos_sed(
    wave_angstrom: np.ndarray,
    flambda_unscaled: np.ndarray,
    row: dict[str, Any],
    filters: dict[str, FilterCurve],
    band_configs: list[dict[str, Any]],
    cosmos_config: dict[str, Any],
) -> dict[str, Any]:
    """Fit scalar normalization against Euclid ``*_abs`` flux densities."""
    response_kind = str(cosmos_config.get("filter_response_kind", "photon"))
    synthetic_before: dict[str, float] = {}
    target: dict[str, float] = {}
    model_values = []
    target_values = []
    for spec in normalization_band_specs(cosmos_config, band_configs):
        band_name = spec["band_name"]
        target_column = spec["target_column"]
        if band_name not in filters:
            raise MissingCosmosResourceError(
                f"Filter for normalization band {band_name!r} missing. "
                f"Configured filters: {sorted(filters)}"
            )
        model = synthetic_fnu_from_flambda(
            wave_angstrom,
            flambda_unscaled,
            filters[band_name],
            response_kind=response_kind,
        )
        value = _row_float(row, target_column)
        synthetic_before[band_name] = model
        target[band_name] = value
        if np.isfinite(model) and np.isfinite(value) and model > 0 and value > 0:
            model_values.append(model)
            target_values.append(value)
    if not model_values:
        raise CosmosSedError(
            "No finite positive model/catalog pairs for COSMOS SED normalization. "
            f"Target columns: {[s['target_column'] for s in normalization_band_specs(cosmos_config, band_configs)]}"
        )
    model_array = np.asarray(model_values, dtype=float)
    target_array = np.asarray(target_values, dtype=float)
    alpha = float(np.sum(model_array * target_array) / np.sum(model_array**2))
    synthetic_after = {
        band_name: value * alpha for band_name, value in synthetic_before.items()
    }
    residual = {
        band_name: synthetic_after[band_name] - target[band_name]
        for band_name in synthetic_after
    }
    relative = {
        band_name: (
            residual[band_name] / target[band_name]
            if np.isfinite(target[band_name]) and target[band_name] > 0
            else float("nan")
        )
        for band_name in synthetic_after
    }
    return {
        "alpha": alpha,
        "synthetic_before": synthetic_before,
        "synthetic_after": synthetic_after,
        "target": target,
        "residual": residual,
        "relative_residual": relative,
    }


def normalization_band_specs(
    cosmos_config: dict[str, Any], band_configs: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Return normalization band mapping to Euclid absolute-flux columns."""
    explicit = cosmos_config.get("normalization_bands")
    if explicit:
        return [
            {
                "band_name": str(item["band_name"]),
                "target_column": str(item["target_column"]),
            }
            for item in explicit
        ]
    specs = []
    for band in band_configs:
        name = str(band["name"])
        column = str(band["column"])
        if name in EUCLID_BAND_NAMES:
            specs.append({"band_name": name, "target_column": f"{column}_abs"})
    return specs


def cosmos_catalog_columns(
    config: dict[str, Any],
    available_columns: set[str] | None = None,
    include_optional: bool = True,
) -> list[str]:
    """Return columns needed for COSMOS SED reconstruction and diagnostics."""
    cosmos_config = config.get("cosmos_sed", {})
    columns = set(COSMOS_COMPONENT_COLUMNS)
    columns.update(
        spec["target_column"]
        for spec in normalization_band_specs(cosmos_config, config["bands"])
    )
    if str(cosmos_config.get("component_fraction_policy", "strict")) == "strict":
        columns.update(COSMOS_FRACTION_COLUMNS)
    elif include_optional:
        columns.update(COSMOS_FRACTION_COLUMNS)
    if include_optional:
        columns.update(
            [
                "color_kind",
                "z_true",
                "z_phz",
                "metallicity_true",
                "log_sfr_true",
                "dust_ebv_true",
            ]
        )
    if available_columns is not None:
        allowed = set(available_columns)
        columns = {column for column in columns if column in allowed}
    return sorted(columns)


def validate_cosmos_catalog(
    df: pd.DataFrame,
    config: dict[str, Any],
    available_columns: set[str] | None = None,
) -> dict[str, Any]:
    """Validate template IDs, extinction codes, fractions, and abs photometry."""
    available = set(df.columns) if available_columns is None else set(available_columns)
    cosmos_config = config.get("cosmos_sed", {})
    required = set(COSMOS_COMPONENT_COLUMNS)
    required.update(
        spec["target_column"]
        for spec in normalization_band_specs(cosmos_config, config["bands"])
    )
    if str(cosmos_config.get("component_fraction_policy", "strict")) == "strict":
        required.update(COSMOS_FRACTION_COLUMNS)
    missing_required = sorted(required - available)
    optional_missing = sorted(set(COSMOS_FRACTION_COLUMNS) - available)
    if missing_required:
        raise MissingCosmosColumnsError(
            "Missing required COSMOS SED columns: " + ", ".join(missing_required)
        )
    report: dict[str, Any] = {
        "n_rows_checked": int(len(df)),
        "missing_required_columns": missing_required,
        "missing_optional_fraction_columns": optional_missing,
        "component_fraction_policy": cosmos_config.get(
            "component_fraction_policy", "strict"
        ),
    }
    for column in ("sed_cosmos_1", "sed_cosmos_2"):
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce")
            invalid = values.isna() | (values < 0) | (values > 30) | (values % 1 != 0)
            report[f"{column}_invalid_count"] = int(invalid.sum())
            report[f"{column}_unique_values"] = [
                int(value) for value in sorted(values.dropna().unique().tolist())
            ]
    for column in ("ext_curve_cosmos_1", "ext_curve_cosmos_2"):
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce")
            invalid = values.isna() | (values < 0) | (values > 4) | (values % 1 != 0)
            report[f"{column}_invalid_count"] = int(invalid.sum())
            report[f"{column}_unique_values"] = [
                int(value) for value in sorted(values.dropna().unique().tolist())
            ]
    if set(COSMOS_FRACTION_COLUMNS).issubset(df.columns):
        frac = df[list(COSMOS_FRACTION_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        frac_sum = frac["frac_cosmos_1"] + frac["frac_cosmos_2"]
        report.update(
            {
                "frac_cosmos_1_summary": _series_summary(frac["frac_cosmos_1"]),
                "frac_cosmos_2_summary": _series_summary(frac["frac_cosmos_2"]),
                "frac_sum_summary": _series_summary(frac_sum),
                "frac_sum_not_close_to_one_count": int(
                    (~np.isclose(frac_sum, 1.0, rtol=1.0e-4, atol=1.0e-6)).sum()
                ),
            }
        )
    return report


def cosmos_diagnostic_row(result: CosmosSedResult) -> dict[str, Any]:
    """Flatten one reconstructed SED normalization result for CSV output."""
    row = dict(result.diagnostics)
    row["alpha"] = result.alpha
    for band, value in result.catalog_abs_fluxes.items():
        suffix = _band_suffix(band)
        row[f"catalog_abs_flux_{suffix}"] = value
        row[f"synthetic_abs_flux_before_{suffix}"] = result.synthetic_abs_fluxes_before[
            band
        ]
        row[f"synthetic_abs_flux_after_{suffix}"] = result.synthetic_abs_fluxes_after[
            band
        ]
        row[f"abs_flux_residual_{suffix}"] = result.residuals_vs_catalog_abs[band]
        row[f"abs_flux_relative_residual_{suffix}"] = (
            result.relative_residuals_vs_catalog_abs[band]
        )
    return row


def cosmos_abs_flux_rows(result: CosmosSedResult) -> list[dict[str, Any]]:
    """Return one row per Euclid absolute-flux comparison."""
    rows = []
    for band in result.catalog_abs_fluxes:
        rows.append(
            {
                "row_index": result.row_index,
                "band": band,
                "catalog_abs_flux_fnu_cgs": result.catalog_abs_fluxes[band],
                "synthetic_abs_flux_before_scaling_fnu_cgs": result.synthetic_abs_fluxes_before[
                    band
                ],
                "synthetic_abs_flux_after_scaling_fnu_cgs": result.synthetic_abs_fluxes_after[
                    band
                ],
                "residual_fnu_cgs": result.residuals_vs_catalog_abs[band],
                "relative_residual": result.relative_residuals_vs_catalog_abs[band],
                **{
                    key: result.diagnostics.get(key)
                    for key in (
                        "color_kind",
                        "z_true",
                        "z_phz",
                        "sed_cosmos_1",
                        "sed_cosmos_2",
                    )
                },
            }
        )
    return rows


def cosmos_sed_long_rows(result: CosmosSedResult) -> pd.DataFrame:
    """Return reproducible long-form sampled SED table for parquet output."""
    return pd.DataFrame(
        {
            "row_index": result.row_index,
            "wave_angstrom": result.wave_angstrom,
            "flambda_rest_proxy_unscaled": result.flambda_unscaled,
            "flambda_rest_proxy_scaled": result.flambda_scaled,
            "fnu_10pc_rest_proxy_scaled": flambda_to_fnu(
                result.wave_angstrom, result.flambda_scaled
            ),
            "lnu_lsun_per_hz_rest_proxy_scaled": flambda_10pc_to_lnu_lsun(
                result.wave_angstrom, result.flambda_scaled
            ),
        }
    )


def flambda_to_fnu(wave_angstrom: np.ndarray, flambda: np.ndarray) -> np.ndarray:
    """Convert ``F_lambda`` per Angstrom to ``Fnu`` at the same wavelength."""
    wave = np.asarray(wave_angstrom, dtype=float)
    return np.asarray(flambda, dtype=float) * wave**2 / C_ANGSTROM_PER_S


def fnu_to_flambda(wave_angstrom: np.ndarray, fnu: np.ndarray | float) -> np.ndarray:
    """Convert ``Fnu`` to ``F_lambda`` per Angstrom."""
    wave = np.asarray(wave_angstrom, dtype=float)
    return np.asarray(fnu, dtype=float) * C_ANGSTROM_PER_S / wave**2


def flambda_10pc_to_lnu_lsun(
    wave_angstrom: np.ndarray, flambda: np.ndarray
) -> np.ndarray:
    """Convert rest-frame 10 pc ``F_lambda`` to ``Lnu`` in ``Lsun/Hz``."""
    return (
        4.0
        * np.pi
        * TEN_PC_CM**2
        * flambda_to_fnu(wave_angstrom, flambda)
        / L_SUN_ERG_PER_S
    )


def lnu_lsun_to_flambda_10pc(
    wave_angstrom: np.ndarray, lnu_lsun_per_hz: np.ndarray
) -> np.ndarray:
    """Convert ``Lnu`` in ``Lsun/Hz`` to rest-frame 10 pc ``F_lambda``."""
    fnu = (
        np.asarray(lnu_lsun_per_hz, dtype=float)
        * L_SUN_ERG_PER_S
        / (4.0 * np.pi * TEN_PC_CM**2)
    )
    return fnu_to_flambda(wave_angstrom, fnu)


def compare_cosmos_to_dsps_rest_sed(
    cosmos_result: CosmosSedResult,
    dsps_result: ModelResult,
    filters: dict[str, FilterCurve],
    cosmos_config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare COSMOS proxy rest SED shape against DSPS rest SED."""
    wave_min = float(cosmos_config.get("comparison_wave_min_angstrom", 1000.0))
    wave_max = float(cosmos_config.get("comparison_wave_max_angstrom", 30_000.0))
    wave = cosmos_result.wave_angstrom
    cosmos_lnu = flambda_10pc_to_lnu_lsun(wave, cosmos_result.flambda_scaled)
    dsps_lnu = np.interp(wave, dsps_result.wave, dsps_result.dusted_rest_sed)
    mask = (
        np.isfinite(wave)
        & np.isfinite(cosmos_lnu)
        & np.isfinite(dsps_lnu)
        & (wave >= wave_min)
        & (wave <= wave_max)
        & (cosmos_lnu > 0)
        & (dsps_lnu > 0)
    )
    if mask.sum() < 2:
        metrics = {
            "row_index": cosmos_result.row_index,
            "n_common_sed_points": int(mask.sum()),
            "dsps_scale_to_cosmos": float("nan"),
            "rms_log_sed_residual": float("nan"),
            "median_abs_log_sed_residual": float("nan"),
            **_diagnostic_group_values(cosmos_result),
        }
        return metrics, pd.DataFrame()
    scale = float(
        np.sum(dsps_lnu[mask] * cosmos_lnu[mask]) / np.sum(dsps_lnu[mask] ** 2)
    )
    dsps_scaled = dsps_lnu * scale
    log_residual = np.full_like(wave, np.nan, dtype=float)
    relative = np.full_like(wave, np.nan, dtype=float)
    good = mask & (dsps_scaled > 0)
    log_residual[good] = np.log10(dsps_scaled[good]) - np.log10(cosmos_lnu[good])
    relative[good] = (dsps_scaled[good] - cosmos_lnu[good]) / cosmos_lnu[good]
    residual_values = log_residual[good]
    metrics = {
        "row_index": cosmos_result.row_index,
        "n_common_sed_points": int(good.sum()),
        "dsps_scale_to_cosmos": scale,
        "rms_log_sed_residual": float(np.sqrt(np.mean(residual_values**2))),
        "median_abs_log_sed_residual": float(np.median(np.abs(residual_values))),
        **_diagnostic_group_values(cosmos_result),
    }
    metrics.update(_shape_diagnostic_metrics(wave, cosmos_lnu, dsps_scaled))
    metrics.update(
        rest_sed_color_residuals(
            wave,
            cosmos_result.flambda_scaled,
            dsps_result,
            scale,
            filters,
            cosmos_config,
        )
    )
    comparison = pd.DataFrame(
        {
            "row_index": cosmos_result.row_index,
            "wave_angstrom": wave[good],
            "cosmos_proxy_lnu_lsun_per_hz": cosmos_lnu[good],
            "dsps_dusted_lnu_lsun_per_hz": dsps_lnu[good],
            "dsps_scaled_lnu_lsun_per_hz": dsps_scaled[good],
            "log10_dsps_scaled_minus_cosmos": log_residual[good],
            "relative_residual_dsps_scaled_minus_cosmos": relative[good],
        }
    )
    return metrics, comparison


def rest_sed_color_residuals(
    wave: np.ndarray,
    cosmos_flambda: np.ndarray,
    dsps_result: ModelResult,
    dsps_scale: float,
    filters: dict[str, FilterCurve],
    cosmos_config: dict[str, Any],
) -> dict[str, float]:
    """Compute synthetic Euclid rest-flux color residuals."""
    response_kind = str(cosmos_config.get("filter_response_kind", "photon"))
    dsps_flambda = lnu_lsun_to_flambda_10pc(
        dsps_result.wave, dsps_result.dusted_rest_sed * dsps_scale
    )
    flux_cosmos = {}
    flux_dsps = {}
    for band in EUCLID_BAND_NAMES:
        if band not in filters:
            continue
        flux_cosmos[band] = synthetic_fnu_from_flambda(
            wave, cosmos_flambda, filters[band], response_kind
        )
        flux_dsps[band] = synthetic_fnu_from_flambda(
            dsps_result.wave, dsps_flambda, filters[band], response_kind
        )
    residuals = {}
    pairs = [
        ("vis_y", "euclid_vis", "euclid_nisp_y"),
        ("y_j", "euclid_nisp_y", "euclid_nisp_j"),
        ("j_h", "euclid_nisp_j", "euclid_nisp_h"),
    ]
    for label, blue, red in pairs:
        if blue in flux_cosmos and red in flux_cosmos:
            residuals[f"rest_color_residual_{label}_mag"] = _color_mag(
                flux_dsps[blue], flux_dsps[red]
            ) - _color_mag(flux_cosmos[blue], flux_cosmos[red])
    return residuals


def observed_photometry_target_rows(
    row: dict[str, Any],
    row_index: int,
    dsps_result: ModelResult,
    band_configs: list[dict[str, Any]],
    target_set_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compare DSPS observed photometry to configured Euclid catalog target sets."""
    rows = []
    for target_set in photometry_target_sets(band_configs, target_set_names):
        for item in target_set["bands"]:
            band_name = item["band_name"]
            if band_name not in dsps_result.photometry:
                continue
            model_flux = float(dsps_result.photometry[band_name]["model_flux_fnu_cgs"])
            observed = _row_float(row, item["target_column"])
            error = _row_float(row, item.get("error_column"))
            residual = model_flux - observed
            relative = (
                residual / observed
                if np.isfinite(observed) and observed > 0
                else float("nan")
            )
            chi = residual / error if np.isfinite(error) and error > 0 else float("nan")
            rows.append(
                {
                    "row_index": int(row_index),
                    "target_set": target_set["name"],
                    "band": band_name,
                    "target_column": item["target_column"],
                    "error_column": item.get("error_column", ""),
                    "model_flux_fnu_cgs": model_flux,
                    "observed_flux_fnu_cgs": observed,
                    "flux_error_fnu_cgs": error,
                    "flux_residual_model_minus_observed": residual,
                    "relative_flux_residual": relative,
                    "chi": chi,
                    **_row_group_values(row),
                }
            )
    return rows


def observed_photometry_chi2_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize branch-2 chi-square by row and target set."""
    if frame.empty or "chi" not in frame:
        return pd.DataFrame()
    work = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["chi"])
    if work.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["row_index", "target_set"]
    for keys, group in work.groupby(group_cols):
        row_index, target_set = keys
        chi = group["chi"].to_numpy(dtype=float)
        rows.append(
            {
                "row_index": int(row_index),
                "target_set": target_set,
                "n_bands_with_error": int(len(chi)),
                "chi2": float(np.sum(chi**2)),
                "reduced_chi2": float(np.sum(chi**2) / max(len(chi), 1)),
                **{
                    key: group[key].iloc[0]
                    for key in (
                        "color_kind",
                        "z_true",
                        "z_phz",
                        "metallicity_true",
                        "log_sfr_true",
                        "dust_ebv_true",
                    )
                    if key in group
                },
            }
        )
    return pd.DataFrame(rows)


def photometry_target_sets(
    band_configs: list[dict[str, Any]], names: list[str] | None = None
) -> list[dict[str, Any]]:
    """Build branch-2 target column sets for configured photometric bands."""
    all_bands = [(str(band["name"]), str(band["column"])) for band in band_configs]
    euclid_bands = [
        (str(band["name"]), str(band["column"]))
        for band in band_configs
        if str(band["name"]) in EUCLID_BAND_NAMES
    ]
    specs = [
        ("continuum_internal_dust", "{base}", None, all_bands),
        ("emission_lines_internal_dust", "{base}_el_model3_ext", None, euclid_bands),
        (
            "emission_lines_internal_dust_mw",
            "{base}_el_model3_ext_odonnell_ext",
            None,
            euclid_bands,
        ),
        (
            "noisy_observation",
            "{base}_el_model3_ext_odonnell_ext_error_realization",
            "{base}_el_model3_ext_odonnell_ext_error",
            euclid_bands,
        ),
    ]
    requested = set(names or [])
    target_sets = []
    for name, target_pattern, error_pattern, bands in specs:
        if requested and name not in requested:
            continue
        target_sets.append(
            {
                "name": name,
                "bands": [
                    {
                        "band_name": band_name,
                        "target_column": target_pattern.format(base=column),
                        "error_column": (
                            error_pattern.format(base=column) if error_pattern else ""
                        ),
                    }
                    for band_name, column in bands
                ],
            }
        )
    return target_sets


def grouped_metric_summary(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Group metrics by color kind and broad redshift bins."""
    if frame.empty or value_column not in frame:
        return pd.DataFrame()
    work = frame.copy()
    if "z_true" in work:
        work["z_bin"] = pd.cut(
            pd.to_numeric(work["z_true"], errors="coerce"),
            bins=[0.0, 0.5, 1.0, 1.5, 2.5, 4.5, np.inf],
            include_lowest=True,
        ).astype(str)
    group_cols = [
        column for column in ("color_kind", "z_bin") if column in work.columns
    ]
    if not group_cols:
        return pd.DataFrame()
    values = pd.to_numeric(work[value_column], errors="coerce")
    work[value_column] = values
    grouped = work.dropna(subset=[value_column]).groupby(group_cols, dropna=False)[
        value_column
    ]
    return grouped.agg(["count", "mean", "median", "std"]).reset_index()


def _template_list_path(cosmos_config: dict[str, Any]) -> Path:
    lephare_dir = resolve_lephare_data_dir(cosmos_config)
    template_subdir = str(cosmos_config.get("template_subdir", "sed/GAL/COSMOS_SED"))
    template_list = str(cosmos_config.get("template_list", "COSMOS_MOD.list"))
    return (lephare_dir / template_subdir / template_list).resolve()


def _resolve_template_entry(
    entry: str, list_path: Path, cosmos_config: dict[str, Any]
) -> Path:
    lephare_dir = resolve_lephare_data_dir(cosmos_config)
    template_dir = list_path.parent
    candidates = [
        template_dir / entry,
        template_dir / Path(entry).name,
        template_dir.parent / entry,
        lephare_dir / entry,
        lephare_dir / "sed" / "GAL" / entry,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise MissingCosmosResourceError(
        f"COSMOS template file not found for list entry {entry!r}. Searched: {searched}"
    )


def _resolve_extinction_file(extinction_dir: Path, curve_name: str) -> Path:
    name = curve_name if curve_name.endswith(".dat") else f"{curve_name}.dat"
    return (extinction_dir / name).resolve()


def _normalized_extinction_mapping(cosmos_config: dict[str, Any]) -> dict[int, str]:
    raw = cosmos_config.get("extinction", {}).get("curves", {})
    mapping = {int(code): str(name) for code, name in raw.items()}
    if not mapping:
        mapping = {
            0: "none",
            1: "SMC_prevot",
            2: "SB_calzetti",
            3: "SB_calzetti_bump1",
            4: "SB_calzetti_bump2",
        }
    return {code: _normalized_curve_name(name) for code, name in mapping.items()}


def _normalized_curve_name(name: str) -> str:
    normalized = name.strip()
    if normalized.lower() in {"", "none", "null"}:
        return "none"
    return normalized[:-4] if normalized.endswith(".dat") else normalized


def _load_two_column_file(path: Path, label: str) -> np.ndarray:
    try:
        data = np.loadtxt(path)
    except OSError as exc:
        raise MissingCosmosResourceError(f"Cannot read {label}: {path}") from exc
    if data.ndim != 2 or data.shape[1] < 2:
        raise CosmosSedError(f"{label} must be a two-column ASCII file: {path}")
    return np.asarray(data[:, :2], dtype=float)


def _wave_unit_factor(unit: Any) -> float:
    normalized = str(unit or "angstrom").strip().lower()
    if normalized in {"angstrom", "ang", "aa", "a"}:
        return 1.0
    if normalized in {"nm", "nanometer", "nanometers"}:
        return 10.0
    if normalized in {"micron", "microns", "um"}:
        return 10_000.0
    raise CosmosSedError(f"Unsupported COSMOS template wavelength unit: {unit}")


def _template_id_from_row(row: dict[str, Any], column: str, n_templates: int) -> int:
    value = _row_float(row, column)
    if not np.isfinite(value) or value < 0 or value > n_templates - 1:
        raise CosmosSedError(
            f"{column} must be finite and in [0, {n_templates - 1}], got {value}"
        )
    rounded = int(round(value))
    if not np.isclose(value, rounded):
        raise CosmosSedError(f"{column} must be an integer template ID, got {value}")
    return rounded


def _curve_code_from_row(row: dict[str, Any], column: str) -> int:
    value = _row_float(row, column)
    if not np.isfinite(value):
        raise CosmosSedError(f"{column} must be finite, got {value}")
    rounded = int(round(value))
    if not np.isclose(value, rounded):
        raise CosmosSedError(f"{column} must be an integer code, got {value}")
    return rounded


def _row_float(row: dict[str, Any], column: str | None) -> float:
    if not column:
        return float("nan")
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _optional_row_float(row: dict[str, Any], column: str) -> float | None:
    value = _row_float(row, column)
    return float(value) if np.isfinite(value) else None


def _series_summary(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "p16": float(values.quantile(0.16)),
        "median": float(values.median()),
        "p84": float(values.quantile(0.84)),
        "max": float(values.max()),
    }


def _band_suffix(band_name: str) -> str:
    return (
        band_name.replace("euclid_", "").replace("nisp_", "").replace("-", "_").upper()
    )


def _color_mag(blue_flux: float, red_flux: float) -> float:
    if (
        not np.isfinite(blue_flux)
        or not np.isfinite(red_flux)
        or blue_flux <= 0
        or red_flux <= 0
    ):
        return float("nan")
    return float(-2.5 * np.log10(blue_flux / red_flux))


def _shape_diagnostic_metrics(
    wave: np.ndarray, cosmos_lnu: np.ndarray, dsps_lnu: np.ndarray
) -> dict[str, float]:
    metrics = {}
    intervals = {
        "uv_1500_2800": (1500.0, 2800.0),
        "optical_4000_8000": (4000.0, 8000.0),
        "nir_9000_18000": (9000.0, 18_000.0),
    }
    for name, (lo, hi) in intervals.items():
        cosmos_slope = _log_slope(wave, cosmos_lnu, lo, hi)
        dsps_slope = _log_slope(wave, dsps_lnu, lo, hi)
        metrics[f"cosmos_log_slope_{name}"] = cosmos_slope
        metrics[f"dsps_log_slope_{name}"] = dsps_slope
        metrics[f"log_slope_residual_{name}"] = dsps_slope - cosmos_slope
    cosmos_d4000 = _break_ratio(wave, cosmos_lnu, (4000.0, 4100.0), (3850.0, 3950.0))
    dsps_d4000 = _break_ratio(wave, dsps_lnu, (4000.0, 4100.0), (3850.0, 3950.0))
    metrics["cosmos_d4000_proxy"] = cosmos_d4000
    metrics["dsps_d4000_proxy"] = dsps_d4000
    metrics["d4000_proxy_residual"] = dsps_d4000 - cosmos_d4000
    return metrics


def _log_slope(
    wave: np.ndarray, values: np.ndarray, wave_min: float, wave_max: float
) -> float:
    mask = (
        np.isfinite(wave)
        & np.isfinite(values)
        & (wave >= wave_min)
        & (wave <= wave_max)
        & (wave > 0)
        & (values > 0)
    )
    if mask.sum() < 5:
        return float("nan")
    slope, _ = np.polyfit(np.log10(wave[mask]), np.log10(values[mask]), 1)
    return float(slope)


def _break_ratio(
    wave: np.ndarray,
    values: np.ndarray,
    numerator_interval: tuple[float, float],
    denominator_interval: tuple[float, float],
) -> float:
    numerator = _interval_median(wave, values, numerator_interval)
    denominator = _interval_median(wave, values, denominator_interval)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _interval_median(
    wave: np.ndarray, values: np.ndarray, interval: tuple[float, float]
) -> float:
    lo, hi = interval
    mask = (
        np.isfinite(wave)
        & np.isfinite(values)
        & (wave >= lo)
        & (wave <= hi)
        & (values > 0)
    )
    if not mask.any():
        return float("nan")
    return float(np.median(values[mask]))


def _diagnostic_group_values(result: CosmosSedResult) -> dict[str, Any]:
    return {
        key: result.diagnostics.get(key)
        for key in (
            "color_kind",
            "z_true",
            "z_phz",
            "metallicity_true",
            "log_sfr_true",
            "dust_ebv_true",
        )
        if key in result.diagnostics
    }


def _row_group_values(row: dict[str, Any]) -> dict[str, Any]:
    return {
        column: _optional_row_float(row, column)
        for column in (
            "color_kind",
            "z_true",
            "z_phz",
            "metallicity_true",
            "log_sfr_true",
            "dust_ebv_true",
        )
        if column in row
    }
