"""End-to-end workflows used by the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .filters import load_filters
from .fit import fit_galaxy_batch_adam, fit_one_galaxy, fit_population_batch_adam
from .io import (
    abmag_to_flux_fnu_cgs,
    build_observation,
    ensure_dir,
    flux_fnu_cgs_to_abmag,
    iter_catalog_batches,
    microjy_to_abmag,
    microjy_to_flux_fnu_cgs,
    read_catalog,
    required_catalog_columns,
    write_json,
)
from .model import load_context, parameters_for_row, predict_batch_mags, run_dsps_model
from .reports import write_batch_outputs, write_eda_outputs, write_fit_outputs, write_run_outputs
from .selection import select_galaxy_row

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional runtime dependency fallback
    tqdm = None


def run_eda(config: dict[str, Any], out_dir: str | Path) -> None:
    columns = required_catalog_columns(config)
    df = read_catalog(config["catalog_path"], columns=columns, nrows=config["eda"].get("nrows"))
    write_eda_outputs(df, config["bands"], out_dir, redshift_config=config.get("redshift"))


def prepare_one(config: dict[str, Any]):
    columns = required_catalog_columns(config)
    df = read_catalog(config["catalog_path"], columns=columns, nrows=config["selection"].get("read_nrows"))
    band_columns = [band["column"] for band in config["bands"]]
    row_index, row = select_galaxy_row(
        df,
        band_columns=band_columns,
        index=config["selection"].get("index"),
        require_positive_flux=bool(config["selection"].get("require_positive_flux", True)),
        sort_by_flux=config["selection"].get("sort_by_flux"),
    )
    observation = build_observation(row_index, row, config["bands"])
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
    )
    params = parameters_for_row(
        config["model"]["fixed_parameters"],
        config["model"].get("parameter_columns", {}),
        row.to_dict(),
        config.get("redshift", {}),
    )
    return context, observation, params


def run_one(config: dict[str, Any], out_dir: str | Path) -> pd.DataFrame:
    out = ensure_dir(out_dir)
    context, observation, params = prepare_one(config)
    result = run_dsps_model(context, params)
    comparison = write_run_outputs(observation, result, out)
    write_json(
        out / "run_summary.json",
        {
            "row_index": observation.row_index,
            "n_bands": len(observation.bands),
            **_row_context(observation.row, params, config),
        },
    )
    return comparison


def fit_one(config: dict[str, Any], out_dir: str | Path) -> None:
    out = ensure_dir(out_dir)
    context, observation, params = prepare_one(config)
    fit_result = fit_one_galaxy(context, observation, params, config["fit"])
    write_run_outputs(observation, fit_result.model_result, out)
    write_fit_outputs(fit_result, out)


def run_batch(config: dict[str, Any], out_dir: str | Path, limit: int | None = None, batch_size: int = 10_000) -> None:
    """Run the same configured DSPS model over many catalog rows.

    This is intentionally conservative: it writes a flat comparison table and
    supports per-row physical parameters through `model.parameter_columns`.
    """
    out = ensure_dir(out_dir)
    columns = required_catalog_columns(config)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
    )

    rows = []
    total = _progress_total(config["catalog_path"], limit)
    with _make_progress_bar(total=total, desc="run-batch", unit="galaxy") as progress:
        for batch in iter_catalog_batches(config["catalog_path"], columns=columns, batch_size=batch_size, limit=limit):
            rows.extend(_forward_dataframe_batch(context, batch, config))
            _update_progress(progress, row_index=int(batch.index[-1]), amount=len(batch))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "batch_photometry_comparison.csv", index=False)
    write_batch_outputs(comparison, out, label="batch")
    write_json(out / "batch_run_config.json", {"rows_written": len(rows), "limit": limit, "batch_size": batch_size})


def fit_batch(config: dict[str, Any], out_dir: str | Path, limit: int | None = 25, batch_size: int = 1000) -> None:
    """Fit the configured free parameters for many rows.

    The default path optimizes each parquet chunk with one JAX-vmapped Adam run.
    """
    out = ensure_dir(out_dir)
    columns = required_catalog_columns(config)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
    )

    comparison_rows = []
    fit_rows = []
    trace_rows = []
    total = _progress_total(config["catalog_path"], limit)
    with _make_progress_bar(total=total, desc="fit-batch", unit="galaxy") as progress:
        for batch in iter_catalog_batches(config["catalog_path"], columns=columns, batch_size=batch_size, limit=limit):
            batch_result = _fit_dataframe_batch(context, batch, config)
            fit_rows.extend(batch_result["fit_rows"])
            comparison_rows.extend(batch_result["comparison_rows"])
            trace_rows.extend(batch_result["trace_rows"])
            _update_progress(progress, row_index=int(batch.index[-1]), amount=len(batch))

    fits = pd.DataFrame(fit_rows)
    comparison = pd.DataFrame(comparison_rows)
    fits.to_csv(out / "batch_fit_results.csv", index=False)
    comparison.to_csv(out / "batch_fit_photometry_comparison.csv", index=False)
    if trace_rows:
        pd.DataFrame(trace_rows).to_csv(out / "batch_fit_trace.csv", index=False)
    write_batch_outputs(comparison, out, label="batch_fit")
    write_json(out / "batch_fit_run_config.json", {"rows_written": len(comparison_rows), "limit": limit, "batch_size": batch_size})


def fit_population(config: dict[str, Any], out_dir: str | Path, limit: int | None = 25, batch_size: int = 256) -> None:
    """Fit chunked hierarchical population MAP models with JAX-vmapped Adam."""
    out = ensure_dir(out_dir)
    columns = required_catalog_columns(config)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
    )

    comparison_rows = []
    fit_rows = []
    hyper_rows = []
    trace_rows = []
    total = _progress_total(config["catalog_path"], limit)
    with _make_progress_bar(total=total, desc="fit-population", unit="galaxy") as progress:
        chunk_index = 0
        for batch in iter_catalog_batches(config["catalog_path"], columns=columns, batch_size=batch_size, limit=limit):
            batch_result = _fit_dataframe_batch(context, batch, config, population=True, chunk_index=chunk_index)
            fit_rows.extend(batch_result["fit_rows"])
            comparison_rows.extend(batch_result["comparison_rows"])
            hyper_rows.extend(batch_result["hyper_rows"])
            trace_rows.extend(batch_result["trace_rows"])
            _update_progress(progress, row_index=int(batch.index[-1]), amount=len(batch))
            chunk_index += 1

    fits = pd.DataFrame(fit_rows)
    comparison = pd.DataFrame(comparison_rows)
    fits.to_csv(out / "population_fit_results.csv", index=False)
    comparison.to_csv(out / "population_fit_photometry_comparison.csv", index=False)
    if hyper_rows:
        pd.DataFrame(hyper_rows).to_csv(out / "population_hyperparameters.csv", index=False)
    if trace_rows:
        pd.DataFrame(trace_rows).to_csv(out / "population_fit_trace.csv", index=False)
    write_batch_outputs(comparison, out, label="population_fit")
    write_json(out / "population_fit_run_config.json", {"rows_written": len(comparison_rows), "limit": limit, "batch_size": batch_size})


def _comparison_for_batch(observation, result, params, config):
    from .model import comparison_rows

    context_values = _row_context(observation.row, params, config)
    for row in comparison_rows(observation, result):
        row["row_index"] = observation.row_index
        row.update(context_values)
        yield row


def _row_context(row: dict[str, Any], params: dict[str, float], config: dict[str, Any]) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    values["z_obs"] = float(params["z_obs"])
    redshift = config.get("redshift", {})
    truth_col = redshift.get("truth_column") or config.get("truth", {}).get("redshift_column")
    if truth_col and truth_col in row and pd.notna(row[truth_col]):
        values["redshift_truth"] = float(row[truth_col])
        values["delta_z_obs_minus_truth"] = values["z_obs"] - values["redshift_truth"]
    for key, value in params.items():
        values[f"param_{key}"] = float(value)
    for truth_name, column in (config.get("truth", {}).get("parameter_columns") or {}).items():
        if column in row and pd.notna(row[column]):
            values[f"truth_{truth_name}"] = float(row[column])
            param_key = f"param_{truth_name}"
            if param_key in values:
                values[f"delta_{truth_name}"] = values[param_key] - values[f"truth_{truth_name}"]
    for column in config.get("extra_columns", []):
        if column in row and pd.notna(row[column]):
            values[f"catalog_{column}"] = float(row[column])
    return values


def _fit_dataframe_batch(context, batch: pd.DataFrame, config: dict[str, Any], population: bool = False, chunk_index: int = 0) -> dict[str, list[dict[str, Any]]]:
    observed_mag, observed_flux, sigma_mag = _photometry_arrays(batch, config["bands"])
    base_rows = [
        parameters_for_row(
            config["model"]["fixed_parameters"],
            config["model"].get("parameter_columns", {}),
            row.to_dict(),
            config.get("redshift", {}),
        )
        for _, row in batch.iterrows()
    ]
    if population:
        pop_result = fit_population_batch_adam(context, base_rows, observed_mag, sigma_mag, config["fit"])
        fit_result = pop_result.batch
        hyper_rows = [
            {
                "chunk_index": chunk_index,
                "n_galaxies": len(batch),
                "parameter": name,
                "population_mu": pop_result.hyper_mu[name],
                "population_sigma": pop_result.hyper_sigma[name],
                "loss": pop_result.loss,
                "device": fit_result.device,
            }
            for name in fit_result.free_parameter_names
        ]
    else:
        fit_result = fit_galaxy_batch_adam(context, base_rows, observed_mag, sigma_mag, config["fit"])
        hyper_rows = []

    fit_rows = []
    comparison_rows = []
    band_names = [band["name"] for band in config["bands"]]
    filter_curves = [context.filters[name] for name in band_names]
    param_matrix = fit_result.best_parameter_matrix
    for local_index, (row_index, row) in enumerate(batch.iterrows()):
        params = {
            name: float(param_matrix[local_index, param_index])
            for param_index, name in enumerate(fit_result.parameter_names)
        }
        context_values = _row_context(row.to_dict(), params, config)
        n_bands = len(config["bands"])
        fit_rows.append(
            {
                "row_index": int(row_index),
                "success": bool(fit_result.success[local_index]),
                "message": fit_result.message,
                "chi2": float(fit_result.chi2[local_index]),
                "reduced_chi2": float(fit_result.chi2[local_index]) / max(n_bands, 1),
                "gradient_norm": float(fit_result.gradient_norm[local_index]),
                "n_bands": n_bands,
                "device": fit_result.device,
                **{f"fit_{key}": value for key, value in params.items()},
                **context_values,
            }
        )
        for band_index, band in enumerate(config["bands"]):
            model_mag = float(fit_result.model_mags[local_index, band_index])
            obs_mag = float(observed_mag[local_index, band_index])
            obs_flux = float(observed_flux[local_index, band_index])
            sigma = float(sigma_mag[local_index, band_index])
            model_flux = abmag_to_flux_fnu_cgs(model_mag)
            flux_ratio = model_flux / obs_flux if obs_flux > 0 else float("nan")
            residual = obs_mag - model_mag
            comparison_rows.append(
                {
                    "row_index": int(row_index),
                    "band": band["name"],
                    "column": band["column"],
                    "effective_wavelength_angstrom": filter_curves[band_index].effective_wavelength,
                    "observed_flux_fnu_cgs": obs_flux,
                    "observed_mag_ab": obs_mag,
                    "sigma_mag": sigma,
                    "model_flux_fnu_cgs": model_flux,
                    "model_mag_ab": model_mag,
                    "residual_mag_observed_minus_model": residual,
                    "residual_mag_model_minus_observed": -residual,
                    "flux_ratio_model_over_observed": flux_ratio,
                    "fractional_flux_residual_model_minus_observed": flux_ratio - 1.0,
                    "chi": residual / sigma if sigma > 0 else float("nan"),
                    "filter_source": filter_curves[band_index].source,
                    **context_values,
                }
            )

    trace_rows = [
        {
            "chunk_index": chunk_index,
            **entry,
        }
        for entry in fit_result.trace
    ]
    return {
        "fit_rows": fit_rows,
        "comparison_rows": comparison_rows,
        "hyper_rows": hyper_rows,
        "trace_rows": trace_rows,
    }


def _forward_dataframe_batch(context, batch: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    observed_mag, observed_flux, sigma_mag = _photometry_arrays(batch, config["bands"])
    base_rows = [
        parameters_for_row(
            config["model"]["fixed_parameters"],
            config["model"].get("parameter_columns", {}),
            row.to_dict(),
            config.get("redshift", {}),
        )
        for _, row in batch.iterrows()
    ]
    parameter_names = list(base_rows[0])
    parameter_matrix = pd.DataFrame(base_rows, columns=parameter_names).to_numpy(dtype=float)
    model_mags = predict_batch_mags(context, parameter_names, parameter_matrix)
    comparison_rows = []
    band_names = [band["name"] for band in config["bands"]]
    filter_curves = [context.filters[name] for name in band_names]
    for local_index, (row_index, row) in enumerate(batch.iterrows()):
        params = {
            name: float(parameter_matrix[local_index, param_index])
            for param_index, name in enumerate(parameter_names)
        }
        context_values = _row_context(row.to_dict(), params, config)
        for band_index, band in enumerate(config["bands"]):
            model_mag = float(model_mags[local_index, band_index])
            obs_mag = float(observed_mag[local_index, band_index])
            obs_flux = float(observed_flux[local_index, band_index])
            sigma = float(sigma_mag[local_index, band_index])
            model_flux = abmag_to_flux_fnu_cgs(model_mag)
            flux_ratio = model_flux / obs_flux if obs_flux > 0 else float("nan")
            residual = obs_mag - model_mag
            comparison_rows.append(
                {
                    "row_index": int(row_index),
                    "band": band["name"],
                    "column": band["column"],
                    "effective_wavelength_angstrom": filter_curves[band_index].effective_wavelength,
                    "observed_flux_fnu_cgs": obs_flux,
                    "observed_mag_ab": obs_mag,
                    "sigma_mag": sigma,
                    "model_flux_fnu_cgs": model_flux,
                    "model_mag_ab": model_mag,
                    "residual_mag_observed_minus_model": residual,
                    "residual_mag_model_minus_observed": -residual,
                    "flux_ratio_model_over_observed": flux_ratio,
                    "fractional_flux_residual_model_minus_observed": flux_ratio - 1.0,
                    "chi": residual / sigma if sigma > 0 else float("nan"),
                    "filter_source": filter_curves[band_index].source,
                    **context_values,
                }
            )
    return comparison_rows


def _photometry_arrays(batch: pd.DataFrame, band_configs: list[dict[str, Any]]) -> tuple[Any, Any, Any]:
    mag_columns = []
    flux_columns = []
    sigma_columns = []
    for band in band_configs:
        values = batch[band["column"]].astype(float).to_numpy()
        units = band.get("units", "fnu_cgs")
        if units == "fnu_cgs":
            flux = values
            mag = [flux_fnu_cgs_to_abmag(value) for value in values]
        elif units == "abmag":
            mag = values
            flux = [abmag_to_flux_fnu_cgs(value) for value in values]
        elif units in {"microjy", "ujy"}:
            mag = [microjy_to_abmag(value) for value in values]
            flux = [microjy_to_flux_fnu_cgs(value) for value in values]
        else:
            raise ValueError(f"Unsupported photometry units for {band['name']}: {units}")
        mag_columns.append(mag)
        flux_columns.append(flux)
        sigma_columns.append([float(band.get("sigma_mag", 0.05))] * len(batch))
    return (
        pd.DataFrame(mag_columns).transpose().to_numpy(dtype=float),
        pd.DataFrame(flux_columns).transpose().to_numpy(dtype=float),
        pd.DataFrame(sigma_columns).transpose().to_numpy(dtype=float),
    )


def _progress_total(catalog_path: str | Path, limit: int | None) -> int | None:
    if limit is not None:
        return int(limit)
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(catalog_path).metadata.num_rows)
    except Exception:
        return None


class _NullProgress:
    def update(self, _: int = 1) -> None:
        return None

    def set_postfix_str(self, _: str, refresh: bool = False) -> None:
        return None

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _make_progress_bar(total: int | None, desc: str, unit: str) -> Any:
    if tqdm is None:
        return _NullProgress()
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, mininterval=0.2, smoothing=0.05)


def _update_progress(progress: Any, row_index: int, amount: int = 1) -> None:
    progress.update(amount)
    progress.set_postfix_str(f"row={row_index}", refresh=False)
