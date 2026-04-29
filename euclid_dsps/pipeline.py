"""End-to-end workflows used by the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .filters import load_filters
from .fit import fit_one_galaxy
from .io import (
    build_observation,
    ensure_dir,
    iter_catalog_batches,
    read_catalog,
    required_catalog_columns,
    write_json,
)
from .model import load_context, parameters_for_row, run_dsps_model
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
            for row_index, row in batch.iterrows():
                try:
                    observation = build_observation(int(row_index), row, config["bands"])
                    params = parameters_for_row(
                        config["model"]["fixed_parameters"],
                        config["model"].get("parameter_columns", {}),
                        row.to_dict(),
                        config.get("redshift", {}),
                    )
                    result = run_dsps_model(context, params)
                    for comp in _comparison_for_batch(observation, result, params, config):
                        rows.append(comp)
                except Exception as exc:
                    rows.append({"row_index": int(row_index), "error": str(exc)})
                _update_progress(progress, row_index=int(row_index))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "batch_photometry_comparison.csv", index=False)
    write_batch_outputs(comparison, out, label="batch")
    write_json(out / "batch_run_config.json", {"rows_written": len(rows), "limit": limit, "batch_size": batch_size})


def fit_batch(config: dict[str, Any], out_dir: str | Path, limit: int | None = 25, batch_size: int = 1000) -> None:
    """Fit the configured free parameters for many rows.

    Keep `limit` small until the model and filters are final; this performs an
    optimizer run per galaxy.
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
    total = _progress_total(config["catalog_path"], limit)
    with _make_progress_bar(total=total, desc="fit-batch", unit="galaxy") as progress:
        for batch in iter_catalog_batches(config["catalog_path"], columns=columns, batch_size=batch_size, limit=limit):
            for row_index, row in batch.iterrows():
                try:
                    observation = build_observation(int(row_index), row, config["bands"])
                    params = parameters_for_row(
                        config["model"]["fixed_parameters"],
                        config["model"].get("parameter_columns", {}),
                        row.to_dict(),
                        config.get("redshift", {}),
                    )
                    fit_result = fit_one_galaxy(context, observation, params, config["fit"])
                    context_values = _row_context(row.to_dict(), fit_result.best_parameters, config)
                    fit_rows.append(
                        {
                            "row_index": int(row_index),
                            "success": fit_result.success,
                            "message": fit_result.message,
                            "chi2": fit_result.chi2,
                            "reduced_chi2": fit_result.chi2 / max(fit_result.n_bands, 1),
                            "n_bands": fit_result.n_bands,
                            **{f"fit_{key}": value for key, value in fit_result.best_parameters.items()},
                            **context_values,
                        }
                    )
                    for comp in _comparison_for_batch(observation, fit_result.model_result, fit_result.best_parameters, config):
                        comparison_rows.append(comp)
                except Exception as exc:
                    fit_rows.append({"row_index": int(row_index), "success": False, "message": str(exc)})
                    comparison_rows.append({"row_index": int(row_index), "error": str(exc)})
                _update_progress(progress, row_index=int(row_index))

    fits = pd.DataFrame(fit_rows)
    comparison = pd.DataFrame(comparison_rows)
    fits.to_csv(out / "batch_fit_results.csv", index=False)
    comparison.to_csv(out / "batch_fit_photometry_comparison.csv", index=False)
    write_batch_outputs(comparison, out, label="batch_fit")
    write_json(out / "batch_fit_run_config.json", {"rows_written": len(comparison_rows), "limit": limit, "batch_size": batch_size})


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


def _update_progress(progress: Any, row_index: int) -> None:
    progress.update(1)
    progress.set_postfix_str(f"row={row_index}", refresh=False)
