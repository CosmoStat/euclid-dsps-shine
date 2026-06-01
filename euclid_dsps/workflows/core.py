"""End-to-end workflows used by the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..filters import load_filters
from ..fit import fit_galaxy_batch_adam, fit_one_galaxy, fit_population_batch_adam
from ..io import (
    abmag_to_flux_fnu_cgs,
    build_observation,
    ensure_dir,
    flux_error_to_sigma_mag,
    flux_fnu_cgs_to_abmag,
    iter_catalog_batches,
    load_row_indices,
    microjy_to_abmag,
    microjy_to_flux_fnu_cgs,
    read_catalog,
    required_catalog_columns,
    truth_column_from_spec,
    truth_value_from_spec,
    write_dataframe_outputs,
    write_json,
)
from ..model import (
    BatchSedResult,
    ModelResult,
    load_context,
    parameters_for_row,
    predict_batch_derived,
    predict_batch_mags,
    predict_batch_seds,
    run_dsps_model,
)
from ..nebular import write_nebular_diagnostic_outputs
from ..performance import PerformanceRecorder, write_performance_outputs
from ..photometry import magerr_to_fluxerr_fnu_cgs
from ..reporting import (
    write_batch_outputs,
    write_eda_outputs,
    write_fit_diagnostic_outputs,
    write_fit_outputs,
    write_mcmc_outputs,
    write_population_corner_outputs,
    write_run_outputs,
    write_sed_diagnostic_outputs,
    write_trace_truth_outputs,
    write_workflow_comparison,
)
from ..selection import select_galaxy_row
from ..semantics import is_forward_active, is_inferred

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional runtime dependency fallback
    tqdm = None


def run_eda(config: dict[str, Any], out_dir: str | Path) -> None:
    columns = required_catalog_columns(config)
    import pyarrow.parquet as pq

    available_columns = pq.ParquetFile(config["catalog_path"]).schema.names
    for p in ["metallicity_true", "sfr_true", "log_sfr_true", "dust_ebv_true"]:
        if p in available_columns and p not in columns:
            columns.append(p)

    df = read_catalog(
        config["catalog_path"], columns=columns, nrows=config["eda"].get("nrows")
    )
    write_eda_outputs(
        df, config["bands"], out_dir, redshift_config=config.get("redshift")
    )


def prepare_one(config: dict[str, Any]):
    columns = required_catalog_columns(config)
    df = read_catalog(
        config["catalog_path"],
        columns=columns,
        nrows=config["selection"].get("read_nrows"),
    )
    band_columns = [band["column"] for band in config["bands"]]
    row_index, row = select_galaxy_row(
        df,
        band_columns=band_columns,
        index=config["selection"].get("index"),
        require_positive_flux=bool(
            config["selection"].get("require_positive_flux", True)
        ),
        nondetection_policy=config["selection"].get("nondetection_policy"),
        sort_by_flux=config["selection"].get("sort_by_flux"),
    )
    observation = build_observation(row_index, row, config["bands"])
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
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
    ground_truth_sed, ground_truth_status = _ground_truth_sed_for_row(
        observation.row, observation.row_index, context.filters, config
    )
    comparison = write_run_outputs(
        observation,
        result,
        out,
        ground_truth_sed=ground_truth_sed,
        include_filters=_plot_filters(config),
    )
    write_json(
        out / "run_summary.json",
        {
            "row_index": observation.row_index,
            "n_bands": len(observation.bands),
            "ground_truth_sed_status": ground_truth_status,
            **_row_context(observation.row, params, config),
        },
    )
    write_json(out / "normalized_config.json", config)
    return comparison


def fit_one(config: dict[str, Any], out_dir: str | Path) -> None:
    out = ensure_dir(out_dir)
    context, observation, params = prepare_one(config)
    fit_result = fit_one_galaxy(context, observation, params, config["fit"])
    ground_truth_sed, ground_truth_status = _ground_truth_sed_for_row(
        observation.row, observation.row_index, context.filters, config
    )
    write_run_outputs(
        observation,
        fit_result.model_result,
        out,
        ground_truth_sed=ground_truth_sed,
        include_filters=_plot_filters(config),
    )
    write_fit_outputs(fit_result, out)
    write_json(
        out / "sed_diagnostic_summary.json",
        {"ground_truth_sed_status": ground_truth_status},
    )
    write_json(out / "normalized_config.json", config)


def sample_one(config: dict[str, Any], out_dir: str | Path) -> None:
    from ..mcmc import sample_one_galaxy

    out = ensure_dir(out_dir)
    context, observation, params = prepare_one(config)
    map_result = _map_start(context, observation, params, config)
    initial_params = map_result.best_parameters if map_result is not None else None
    mcmc_result = sample_one_galaxy(
        context,
        observation,
        params,
        config["fit"],
        config["sample"],
        initial_params=initial_params,
    )
    context_values = _row_context(observation.row, params, config)
    write_json(
        out / "sampled_galaxy.json",
        {
            "row_index": observation.row_index,
            "n_bands": len(observation.bands),
            "hmc_initialized_from_map": map_result is not None,
            "map_chi2": float(map_result.chi2) if map_result is not None else None,
            **context_values,
        },
    )
    write_mcmc_outputs(
        mcmc_result,
        out,
        truth_values=_posterior_truth_values(context_values),
    )
    write_json(out / "normalized_config.json", config)


def sample_batch(
    config: dict[str, Any],
    out_dir: str | Path,
    limit: int | None = 5,
    batch_size: int = 1,
    row_indices_file: str | None = None,
) -> None:
    """Sample independent galaxy posteriors with NumPyro NUTS.

    This intentionally runs one galaxy at a time; HMC/NUTS is for posterior
    density checks on small subsets, while Adam/MAP remains the full-catalog path.
    """
    from ..mcmc import sample_one_galaxy

    if limit is None and not row_indices_file:
        raise ValueError(
            "Bayesian batch sampling requires --limit; full-catalog HMC is not practical here."
        )

    out = ensure_dir(out_dir)
    row_indices = _load_row_indices_set(row_indices_file)
    columns = required_catalog_columns(config)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )

    summary_rows = []
    predictive_rows = []
    diagnostic_rows = []
    sample_rows = []
    save_samples = bool(config["sample"].get("save_samples", True))

    total = _progress_total(config["catalog_path"], limit, row_indices)
    with _make_progress_bar(
        total=total, desc="sample-batch", unit="galaxy"
    ) as progress:
        for batch in iter_catalog_batches(
            config["catalog_path"],
            columns=columns,
            batch_size=batch_size,
            limit=limit,
            row_indices=row_indices,
        ):
            for row_index, row in batch.iterrows():
                observation = build_observation(int(row_index), row, config["bands"])
                params = parameters_for_row(
                    config["model"]["fixed_parameters"],
                    config["model"].get("parameter_columns", {}),
                    row.to_dict(),
                    config.get("redshift", {}),
                )
                map_result = _map_start(context, observation, params, config)
                initial_params = (
                    map_result.best_parameters if map_result is not None else None
                )
                result = sample_one_galaxy(
                    context,
                    observation,
                    params,
                    config["fit"],
                    config["sample"],
                    initial_params=initial_params,
                )
                context_values = _row_context(row.to_dict(), params, config)
                if map_result is not None:
                    context_values = {
                        **context_values,
                        "hmc_initialized_from_map": True,
                        "map_chi2": float(map_result.chi2),
                    }
                else:
                    context_values = {
                        **context_values,
                        "hmc_initialized_from_map": False,
                        "map_chi2": None,
                    }
                for item in result.summary:
                    summary_rows.append(
                        {"row_index": int(row_index), **item, **context_values}
                    )
                predictive_rows.extend(
                    _posterior_predictive_rows(int(row_index), result, context_values)
                )
                diagnostic_rows.append(
                    {
                        "row_index": int(row_index),
                        **result.diagnostics,
                        **context_values,
                    }
                )
                if save_samples:
                    sample_rows.extend(_posterior_sample_rows(int(row_index), result))
                _update_progress(progress, row_index=int(row_index))

    pd.DataFrame(summary_rows).to_csv(out / "batch_posterior_summary.csv", index=False)
    pd.DataFrame(predictive_rows).to_csv(
        out / "batch_posterior_predictive.csv", index=False
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        out / "batch_mcmc_diagnostics.csv", index=False
    )
    if sample_rows:
        pd.DataFrame(sample_rows).to_csv(
            out / "batch_posterior_samples.csv", index=False
        )
    write_json(out / "normalized_config.json", config)
    from ..reporting import write_mcmc_batch_outputs

    write_mcmc_batch_outputs(
        pd.DataFrame(summary_rows),
        pd.DataFrame(predictive_rows),
        pd.DataFrame(diagnostic_rows),
        out,
    )


def fit_workflow(
    config: dict[str, Any],
    out_dir: str | Path,
    limit: int | None = 1000,
    batch_size: int = 64,
    hmc_n: int = 20,
    hmc_batch_size: int = 1,
    population_batch_size: int | None = None,
    hmc_select: str = "stratified",
    seed: int = 42,
) -> None:
    """Run MAP batch, HMC subset, population MAP, and comparison reports."""
    out = ensure_dir(out_dir)
    map_out = ensure_dir(out / "map")
    hmc_out = ensure_dir(out / "hmc")
    population_out = ensure_dir(out / "population")
    comparison_out = ensure_dir(out / "comparison")

    fit_batch(config, map_out, limit=limit, batch_size=batch_size)
    map_fits = _read_table(map_out / "batch_fit_results.csv")
    selected_rows = _select_hmc_row_indices(
        map_fits, n=hmc_n, method=hmc_select, seed=seed
    )
    hmc_rows_path = out / "hmc_row_indices.txt"
    hmc_rows_path.write_text(
        "\n".join(str(row) for row in selected_rows) + "\n", encoding="utf-8"
    )

    if selected_rows:
        sample_batch(
            config,
            hmc_out,
            limit=None,
            batch_size=hmc_batch_size,
            row_indices_file=str(hmc_rows_path),
        )

    fit_population(
        config,
        population_out,
        limit=limit,
        batch_size=population_batch_size or batch_size,
        map_init_file=str(map_out / "batch_fit_results.csv"),
    )
    population_fits = _read_table(population_out / "population_fit_results.csv")
    hmc_summary = _read_optional_csv(hmc_out / "batch_posterior_summary.csv")
    hmc_diagnostics = _read_optional_csv(hmc_out / "batch_mcmc_diagnostics.csv")
    hmc_samples = _read_optional_csv(hmc_out / "batch_posterior_samples.csv")
    write_workflow_comparison(
        map_fits=map_fits,
        population_fits=population_fits,
        hmc_summary=hmc_summary,
        hmc_diagnostics=hmc_diagnostics,
        hmc_samples=hmc_samples,
        free_parameters=list(config["fit"]["free_parameters"]),
        out_dir=comparison_out,
    )
    write_json(
        out / "fit_workflow_summary.json",
        {
            "limit": limit,
            "batch_size": batch_size,
            "hmc_n": len(selected_rows),
            "hmc_selection": hmc_select,
            "hmc_row_indices_file": str(hmc_rows_path),
            "map_out": str(map_out),
            "hmc_out": str(hmc_out),
            "population_out": str(population_out),
            "comparison_out": str(comparison_out),
        },
    )


def _select_hmc_row_indices(
    map_fits: pd.DataFrame, n: int, method: str, seed: int
) -> list[int]:
    if n <= 0 or map_fits.empty:
        return []
    work = map_fits.drop_duplicates("row_index").copy()
    work = work[pd.to_numeric(work.get("reduced_chi2"), errors="coerce").notna()]
    if work.empty:
        return []
    n = min(int(n), len(work))
    method = method.lower()
    if method == "random":
        selected = work.sample(n=n, random_state=seed)
    elif method == "best":
        selected = work.nsmallest(n, "reduced_chi2")
    elif method == "worst":
        selected = work.nlargest(n, "reduced_chi2")
    elif method == "stratified":
        sorted_work = work.sort_values("reduced_chi2").reset_index(drop=True)
        positions = np.rint(np.linspace(0, len(sorted_work) - 1, n)).astype(int)
        selected = sorted_work.iloc[np.unique(positions)]
        if len(selected) < n:
            missing = sorted_work[~sorted_work["row_index"].isin(selected["row_index"])]
            selected = pd.concat(
                [selected, missing.head(n - len(selected))], ignore_index=True
            )
    else:
        raise ValueError(f"Unsupported HMC selection method: {method}")
    return sorted(int(row) for row in selected["row_index"].tolist())


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.DataFrame()


def _read_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        return pd.read_parquet(parquet)
    raise FileNotFoundError(path)


def _reporting_level(config: dict[str, Any]) -> str:
    return str((config.get("reporting", {}) or {}).get("level", "full")).lower()


def _verbose_benchmark(config: dict[str, Any]) -> bool:
    return bool((config.get("output", {}) or {}).get("verbose_benchmark", False))


def report_workflow(config: dict[str, Any], run_dir: str | Path) -> None:
    """Regenerate workflow comparison reports from existing workflow outputs."""
    root = Path(run_dir)
    map_fits = _read_table(root / "map" / "batch_fit_results.csv")
    population_fits = _read_table(root / "population" / "population_fit_results.csv")
    hmc_summary = _read_optional_csv(root / "hmc" / "batch_posterior_summary.csv")
    hmc_diagnostics = _read_optional_csv(root / "hmc" / "batch_mcmc_diagnostics.csv")
    hmc_samples = _read_optional_csv(root / "hmc" / "batch_posterior_samples.csv")
    write_workflow_comparison(
        map_fits=map_fits,
        population_fits=population_fits,
        hmc_summary=hmc_summary,
        hmc_diagnostics=hmc_diagnostics,
        hmc_samples=hmc_samples,
        free_parameters=list(config["fit"]["free_parameters"]),
        out_dir=root / "comparison",
    )


def _map_start(context, observation, params: dict[str, float], config: dict[str, Any]):
    if not bool(config.get("sample", {}).get("init_from_map", True)):
        return None
    return fit_one_galaxy(context, observation, params, config["fit"])


def run_batch(
    config: dict[str, Any],
    out_dir: str | Path,
    limit: int | None = None,
    batch_size: int = 10_000,
    row_indices_file: str | None = None,
) -> None:
    """Run the same configured DSPS model over many catalog rows.

    This is intentionally conservative: it writes a flat comparison table and
    supports per-row physical parameters through `model.parameter_columns`.
    """
    out = ensure_dir(out_dir)
    perf = PerformanceRecorder(_verbose_benchmark(config))
    reporting_level = _reporting_level(config)
    row_indices = _load_row_indices_set(row_indices_file)
    columns = required_catalog_columns(config)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    perf.mark("load_context", n_bands=len(config["bands"]))

    rows = []
    sed_manifest_rows = []
    sed_samples_written = 0
    sed_sample_limit = _sed_sample_limit(config)
    total = _progress_total(config["catalog_path"], limit, row_indices)
    chunk_index = 0
    with _make_progress_bar(total=total, desc="check-batch", unit="galaxy") as progress:
        for batch in iter_catalog_batches(
            config["catalog_path"],
            columns=columns,
            batch_size=batch_size,
            limit=limit,
            row_indices=row_indices,
        ):
            perf.mark("read_chunk", chunk_index=chunk_index, n_rows=len(batch))
            rows.extend(_forward_dataframe_batch(context, batch, config))
            perf.mark("forward_chunk", chunk_index=chunk_index, n_rows=len(batch))
            if sed_samples_written < sed_sample_limit:
                base_rows = [
                    parameters_for_row(
                        config["model"]["fixed_parameters"],
                        config["model"].get("parameter_columns", {}),
                        row.to_dict(),
                        config.get("redshift", {}),
                    )
                    for _, row in batch.iterrows()
                ]
                new_rows = _write_sed_samples(
                    context,
                    batch,
                    base_rows,
                    config,
                    out,
                    mode="forward",
                    chunk_index=chunk_index,
                    remaining=sed_sample_limit - sed_samples_written,
                )
                sed_manifest_rows.extend(new_rows)
                sed_samples_written += len(new_rows)
                perf.mark(
                    "write_sed_samples",
                    chunk_index=chunk_index,
                    n_rows=len(new_rows),
                )
            _update_progress(
                progress, row_index=int(batch.index[-1]), amount=len(batch)
            )
            chunk_index += 1

    comparison = pd.DataFrame(rows)
    write_dataframe_outputs(comparison, out, "batch_photometry_comparison", config)
    perf.mark("write_primary_outputs", n_rows=len(comparison))
    write_batch_outputs(
        comparison, out, label="batch", reporting_level=reporting_level, config=config
    )
    if sed_manifest_rows:
        pd.DataFrame(sed_manifest_rows).to_csv(
            out / "sed_diagnostics_manifest.csv", index=False
        )
    perf.mark("write_reports")
    write_json(out / "normalized_config.json", config)
    write_json(
        out / "batch_run_config.json",
        {
            "rows_written": len(rows),
            "limit": limit,
            "batch_size": batch_size,
            "row_indices_file": row_indices_file,
        },
    )
    write_performance_outputs(perf.rows, out, "batch")


def fit_batch(
    config: dict[str, Any],
    out_dir: str | Path,
    limit: int | None = 25,
    batch_size: int = 1000,
    row_indices_file: str | None = None,
) -> None:
    """Fit the configured free parameters for many rows.

    The default path optimizes each parquet chunk with one JAX-vmapped Adam run.
    """
    out = ensure_dir(out_dir)
    perf = PerformanceRecorder(_verbose_benchmark(config))
    reporting_level = _reporting_level(config)
    row_indices = _load_row_indices_set(row_indices_file)
    columns = required_catalog_columns(config)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    perf.mark("load_context", n_bands=len(config["bands"]))

    comparison_rows = []
    fit_rows = []
    trace_rows = []
    sed_sample_limit = _sed_sample_limit(config)
    total = _progress_total(config["catalog_path"], limit, row_indices)
    chunk_index = 0
    with _make_progress_bar(total=total, desc="fit", unit="galaxy") as progress:
        for batch in iter_catalog_batches(
            config["catalog_path"],
            columns=columns,
            batch_size=batch_size,
            limit=limit,
            row_indices=row_indices,
        ):
            perf.mark("read_chunk", chunk_index=chunk_index, n_rows=len(batch))
            fit_batch_frame, real_row_indices = _pad_fit_batch_to_static_size(
                batch, batch_size
            )
            if real_row_indices is not None:
                perf.mark(
                    "pad_final_chunk",
                    chunk_index=chunk_index,
                    n_rows=len(batch),
                    padded_n_rows=len(fit_batch_frame),
                )
            batch_result = _fit_dataframe_batch(
                context,
                fit_batch_frame,
                config,
                chunk_index=chunk_index,
                perf=perf,
            )
            if real_row_indices is not None:
                batch_result = _filter_padded_fit_batch_result(
                    batch_result, real_row_indices
                )
            fit_rows.extend(batch_result["fit_rows"])
            comparison_rows.extend(batch_result["comparison_rows"])
            trace_rows.extend(batch_result["trace_rows"])
            _write_fit_chunk_checkpoint(out, batch_result, config, chunk_index)
            perf.mark("fit_chunk", chunk_index=chunk_index, n_rows=len(batch))
            _update_progress(
                progress, row_index=int(batch.index[-1]), amount=len(batch)
            )
            chunk_index += 1

    fits = pd.DataFrame(fit_rows)
    comparison = pd.DataFrame(comparison_rows)
    write_dataframe_outputs(fits, out, "batch_fit_results", config)
    write_dataframe_outputs(comparison, out, "batch_fit_photometry_comparison", config)
    perf.mark("write_primary_outputs", n_rows=len(fits))
    sed_manifest_rows = _write_worst_fit_sed_samples(
        context,
        fits,
        comparison,
        config,
        out,
        limit=sed_sample_limit,
    )
    if sed_manifest_rows:
        perf.mark("write_sed_samples", n_rows=len(sed_manifest_rows))
    if trace_rows:
        trace = pd.DataFrame(trace_rows)
        write_dataframe_outputs(trace, out, "batch_fit_trace", config)
        write_trace_truth_outputs(
            trace, out, label="batch_fit", make_plots=reporting_level == "full"
        )
    write_batch_outputs(
        comparison,
        out,
        label="batch_fit",
        reporting_level=reporting_level,
        config=config,
    )
    if sed_manifest_rows:
        pd.DataFrame(sed_manifest_rows).to_csv(
            out / "sed_diagnostics_manifest.csv", index=False
        )
    write_fit_diagnostic_outputs(fits, comparison, config, out, label="batch_fit")
    write_nebular_diagnostic_outputs(
        context,
        fits,
        out,
        label="batch_fit",
        make_plots=reporting_level == "full",
    )
    perf.mark("write_reports")
    write_json(out / "normalized_config.json", config)
    write_json(
        out / "batch_fit_run_config.json",
        {
            "rows_written": len(comparison_rows),
            "limit": limit,
            "batch_size": batch_size,
            "row_indices_file": row_indices_file,
        },
    )
    write_performance_outputs(perf.rows, out, "batch_fit")


def _pad_fit_batch_to_static_size(
    batch: pd.DataFrame, target_size: int
) -> tuple[pd.DataFrame, set[int] | None]:
    """Pad a final MAP batch so JAX sees the same batch shape throughout a run."""
    if target_size <= 0 or len(batch) == 0 or len(batch) >= target_size:
        return batch, None
    pad_count = target_size - len(batch)
    real_row_indices = {int(index) for index in batch.index}
    pad_rows = batch.iloc[[-1]].copy()
    pad_rows = pd.concat([pad_rows] * pad_count)
    pad_rows.index = [-1_000_000_000 - offset for offset in range(pad_count)]
    padded = pd.concat([batch, pad_rows], axis=0)
    return padded, real_row_indices


def _filter_padded_fit_batch_result(
    batch_result: dict[str, list[dict[str, Any]]], real_row_indices: set[int]
) -> dict[str, list[dict[str, Any]]]:
    """Drop synthetic padding rows before checkpoints and final reports."""

    def keep_real_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = []
        for row in rows:
            if "row_index" not in row:
                continue
            if int(row["row_index"]) in real_row_indices:
                kept.append(row)
        return kept

    return {
        "fit_rows": keep_real_rows(batch_result["fit_rows"]),
        "comparison_rows": keep_real_rows(batch_result["comparison_rows"]),
        "hyper_rows": [],
        "trace_rows": [],
    }


def _write_fit_chunk_checkpoint(
    out: Path,
    batch_result: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    chunk_index: int,
) -> None:
    chunk_dir = ensure_dir(out / "_chunks")
    suffix = f"chunk_{chunk_index:06d}"
    if batch_result["fit_rows"]:
        write_dataframe_outputs(
            pd.DataFrame(batch_result["fit_rows"]),
            chunk_dir,
            f"batch_fit_results_{suffix}",
            config,
        )
    if batch_result["comparison_rows"]:
        write_dataframe_outputs(
            pd.DataFrame(batch_result["comparison_rows"]),
            chunk_dir,
            f"batch_fit_photometry_comparison_{suffix}",
            config,
        )
    if batch_result["trace_rows"]:
        write_dataframe_outputs(
            pd.DataFrame(batch_result["trace_rows"]),
            chunk_dir,
            f"batch_fit_trace_{suffix}",
            config,
        )


_COSMOS_RESOURCE_CACHE: dict[tuple[Any, ...], Any] = {}


def _sed_sample_limit(config: dict[str, Any]) -> int:
    reporting = config.get("reporting", {}) or {}
    return max(int(reporting.get("save_sed_samples", 0) or 0), 0)


def _plot_filters(config: dict[str, Any]) -> bool:
    reporting = config.get("reporting", {}) or {}
    return bool(reporting.get("plot_filters", True))


def _plot_ground_truth(config: dict[str, Any]) -> bool:
    reporting = config.get("reporting", {}) or {}
    return bool(reporting.get("plot_ground_truth", False))


def _write_worst_fit_sed_samples(
    context,
    fits: pd.DataFrame,
    comparison: pd.DataFrame,
    config: dict[str, Any],
    out: Path,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or fits.empty:
        return []
    worst_limit = max(1, (limit + 1) // 2)
    best_limit = max(0, limit - worst_limit)
    worst = _ranked_fit_rows(fits, comparison, worst_limit, worst=True)
    used = set(int(value) for value in worst.get("row_index", []))
    best = _ranked_fit_rows(
        fits,
        comparison,
        best_limit + len(used),
        worst=False,
    )
    if not best.empty and used:
        best = best[~best["row_index"].astype(int).isin(used)].head(best_limit)
    manifest = []
    manifest.extend(
        _write_selected_fit_sed_samples(
            context,
            worst,
            config,
            out,
            mode="fit_worst",
            limit=worst_limit,
            selection_reason="worst_fit",
        )
    )
    manifest.extend(
        _write_selected_fit_sed_samples(
            context,
            best,
            config,
            out,
            mode="fit_best",
            limit=best_limit,
            selection_reason="best_fit",
        )
    )
    return manifest


def _write_selected_fit_sed_samples(
    context,
    selected: pd.DataFrame,
    config: dict[str, Any],
    out: Path,
    *,
    mode: str,
    limit: int,
    selection_reason: str,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if selected.empty:
        return []

    row_indices = [int(value) for value in selected["row_index"]]
    wanted = set(row_indices)
    rows_by_index: dict[int, pd.Series] = {}
    for batch in iter_catalog_batches(
        config["catalog_path"],
        columns=required_catalog_columns(config),
        batch_size=max(1024, len(wanted)),
        row_indices=wanted,
    ):
        for row_index, row in batch.iterrows():
            rows_by_index[int(row_index)] = row
        if len(rows_by_index) == len(wanted):
            break

    rows = []
    fit_rows = []
    for _, fit_row in selected.iterrows():
        row_index = int(fit_row["row_index"])
        row = rows_by_index.get(row_index)
        if row is None:
            continue
        rows.append(row)
        fit_rows.append(fit_row.to_dict())
    if not rows:
        return []

    batch = pd.DataFrame(rows)
    batch.index = [int(row["row_index"]) for row in fit_rows]
    manifest = _write_fit_sed_samples(
        context,
        batch,
        fit_rows,
        config,
        out,
        mode=mode,
        chunk_index=0,
        remaining=limit,
    )
    ranks = {
        int(row["row_index"]): int(rank)
        for rank, row in enumerate(selected.to_dict("records"), start=1)
    }
    scores = {
        int(row["row_index"]): float(row["sed_diagnostic_score"])
        for row in selected.to_dict("records")
    }
    for row in manifest:
        row_index = int(row.get("row_index", -1))
        row["selection_reason"] = selection_reason
        row["selection_rank"] = ranks.get(row_index)
        row["selection_score"] = scores.get(row_index)
    return manifest


def _ranked_fit_rows(
    fits: pd.DataFrame, comparison: pd.DataFrame, limit: int, *, worst: bool
) -> pd.DataFrame:
    if limit <= 0:
        return pd.DataFrame()
    work = fits.copy()
    if "row_index" not in work:
        return pd.DataFrame()
    work["row_index"] = work["row_index"].astype(int)
    if not comparison.empty and "row_index" in comparison:
        residual_col = "residual_mag_model_minus_observed"
        if residual_col in comparison:
            residual = (
                comparison[["row_index", residual_col]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            by_row = (
                residual.assign(abs_residual=residual[residual_col].abs())
                .groupby("row_index", as_index=False)["abs_residual"]
                .median()
                .rename(columns={"abs_residual": "median_abs_mag_residual"})
            )
            work = work.merge(by_row, on="row_index", how="left")
    score_column = next(
        (
            column
            for column in (
                "reduced_fit_quality",
                "fit_quality_per_band",
                "photometric_objective_per_band",
                "reduced_chi2",
                "median_abs_mag_residual",
            )
            if column in work
        ),
        None,
    )
    if score_column is not None:
        score = pd.to_numeric(work[score_column], errors="coerce")
    else:
        score = pd.Series(np.zeros(len(work)), index=work.index)
    fill_value = -np.inf if worst else np.inf
    work["sed_diagnostic_score"] = score.replace([np.inf, -np.inf], np.nan).fillna(
        fill_value
    )
    work["sed_diagnostic_score_metric"] = score_column or "constant_zero"
    if "reduced_chi2" not in work:
        work["reduced_chi2"] = work["sed_diagnostic_score"]
    return (
        work.sort_values(
            ["sed_diagnostic_score", "reduced_chi2"],
            ascending=[not worst, not worst],
            na_position="last",
        )
        .drop_duplicates("row_index")
        .head(limit)
    )


def _write_fit_sed_samples(
    context,
    batch: pd.DataFrame,
    fit_rows: list[dict[str, Any]],
    config: dict[str, Any],
    out: Path,
    *,
    mode: str,
    chunk_index: int,
    remaining: int,
) -> list[dict[str, Any]]:
    fit_by_row = {int(row["row_index"]): row for row in fit_rows}
    parameter_rows = []
    selected = []
    for row_index, row in batch.iterrows():
        if len(selected) >= remaining:
            break
        fit_row = fit_by_row.get(int(row_index))
        if fit_row is None:
            continue
        params = parameters_for_row(
            config["model"]["fixed_parameters"],
            config["model"].get("parameter_columns", {}),
            row.to_dict(),
            config.get("redshift", {}),
        )
        for name in list(params):
            fit_key = f"fit_{name}"
            if fit_key in fit_row and pd.notna(fit_row[fit_key]):
                params[name] = float(fit_row[fit_key])
        selected.append((row_index, row))
        parameter_rows.append(params)
    if not selected:
        return []
    selected_batch = pd.DataFrame([row for _, row in selected])
    selected_batch.index = [int(row_index) for row_index, _ in selected]
    return _write_sed_samples(
        context,
        selected_batch,
        parameter_rows,
        config,
        out,
        mode=mode,
        chunk_index=chunk_index,
        remaining=remaining,
    )


def _write_sed_samples(
    context,
    batch: pd.DataFrame,
    parameter_rows: list[dict[str, Any]],
    config: dict[str, Any],
    out: Path,
    *,
    mode: str,
    chunk_index: int,
    remaining: int,
) -> list[dict[str, Any]]:
    if remaining <= 0 or batch.empty or not parameter_rows:
        return []
    n = min(int(remaining), len(batch), len(parameter_rows))
    batch = batch.head(n)
    parameter_rows = parameter_rows[:n]
    parameter_names = _parameter_names_for_sed_rows(parameter_rows)
    parameter_matrix = pd.DataFrame(parameter_rows, columns=parameter_names).to_numpy(
        dtype=float
    )
    sed_result = predict_batch_seds(context, parameter_names, parameter_matrix)
    sed_dir = ensure_dir(out / "sed_diagnostics")
    manifest_rows = []
    for local_index, (row_index, row) in enumerate(batch.iterrows()):
        row_dict = row.to_dict()
        observation = build_observation(int(row_index), row, config["bands"])
        model_result = _model_result_from_sed_batch(
            sed_result, local_index, context, config
        )
        ground_truth_sed, ground_truth_status = _ground_truth_sed_for_row(
            row_dict, int(row_index), context.filters, config
        )
        stem = f"{mode}_chunk_{chunk_index:06d}_row_{int(row_index):08d}"
        output = write_sed_diagnostic_outputs(
            observation,
            model_result,
            sed_dir,
            stem=stem,
            ground_truth_sed=ground_truth_sed,
            include_filters=_plot_filters(config),
        )
        manifest_rows.append(
            {
                **output,
                "mode": mode,
                "chunk_index": int(chunk_index),
                "ground_truth_sed_status": ground_truth_status,
            }
        )
    return manifest_rows


def _parameter_names_for_sed_rows(parameter_rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in parameter_rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def _model_result_from_sed_batch(
    batch_result: BatchSedResult,
    local_index: int,
    context,
    config: dict[str, Any],
) -> ModelResult:
    params = {
        name: float(batch_result.parameter_matrix[local_index, index])
        for index, name in enumerate(batch_result.parameter_names)
    }
    derived = {
        name: float(values[local_index])
        for name, values in batch_result.derived.items()
    }
    photometry = {}
    for band_index, band in enumerate(config["bands"]):
        name = band["name"]
        curve = context.filters[name]
        mag = float(batch_result.model_mags[local_index, band_index])
        photometry[name] = {
            "model_mag_ab": mag,
            "model_flux_fnu_cgs": abmag_to_flux_fnu_cgs(mag),
            "filter_source": curve.source,
            "effective_wavelength_angstrom": curve.effective_wavelength,
            "filter_wave_angstrom": curve.wave,
            "filter_transmission": curve.transmission,
        }
    return ModelResult(
        parameters=params,
        derived=derived,
        wave=batch_result.wave,
        rest_sed=batch_result.rest_sed[local_index],
        dusted_rest_sed=batch_result.dusted_rest_sed[local_index],
        photometry=photometry,
    )


def _ground_truth_sed_for_row(
    row: dict[str, Any],
    row_index: int,
    filters: Any,
    config: dict[str, Any],
) -> tuple[pd.DataFrame | None, str]:
    if not _plot_ground_truth(config):
        return None, "disabled"
    try:
        from ..cosmos import (
            MissingCosmosColumnsError,
            MissingCosmosResourceError,
            flambda_10pc_to_lnu_lsun,
            load_cosmos_sed_resources,
            reconstruct_cosmos_proxy_sed,
        )
    except Exception as exc:  # pragma: no cover - defensive optional import path
        return None, f"import_failed:{type(exc).__name__}"

    cosmos_config = config.get("cosmos_sed", {}) or {}
    cache_key = (
        cosmos_config.get("value_added_data_dir"),
        cosmos_config.get("lephare_data_dir"),
        cosmos_config.get("template_subdir"),
        cosmos_config.get("template_list"),
    )
    try:
        if cache_key not in _COSMOS_RESOURCE_CACHE:
            _COSMOS_RESOURCE_CACHE[cache_key] = load_cosmos_sed_resources(
                cosmos_config
            )
        cosmos_result = reconstruct_cosmos_proxy_sed(
            row,
            row_index,
            _COSMOS_RESOURCE_CACHE[cache_key],
            filters,
            config["bands"],
            cosmos_config,
        )
    except MissingCosmosColumnsError as exc:
        return None, f"missing_columns:{exc}"
    except MissingCosmosResourceError as exc:
        return None, f"missing_resource:{exc}"
    except Exception as exc:  # pragma: no cover - report unexpected data issue
        return None, f"failed:{type(exc).__name__}:{exc}"

    rel_residual = np.asarray(
        list(cosmos_result.relative_residuals_vs_catalog_abs.values()), dtype=float
    )
    finite_rel = rel_residual[np.isfinite(rel_residual)]
    norm_bands = sorted(cosmos_result.synthetic_abs_fluxes_after)
    return (
        pd.DataFrame(
            {
                "wave_angstrom": cosmos_result.wave_angstrom,
                "ground_truth_lnu_lsun_per_hz": flambda_10pc_to_lnu_lsun(
                    cosmos_result.wave_angstrom, cosmos_result.flambda_scaled
                ),
                "ground_truth_unscaled_lnu_lsun_per_hz": flambda_10pc_to_lnu_lsun(
                    cosmos_result.wave_angstrom, cosmos_result.flambda_unscaled
                ),
                "ground_truth_label": "COSMOS proxy",
                "ground_truth_scale_factor": cosmos_result.alpha,
                "ground_truth_normalization_bands": ",".join(norm_bands),
                "ground_truth_norm_median_abs_rel_residual": (
                    float(np.nanmedian(np.abs(finite_rel)))
                    if finite_rel.size
                    else float("nan")
                ),
                "ground_truth_norm_max_abs_rel_residual": (
                    float(np.nanmax(np.abs(finite_rel)))
                    if finite_rel.size
                    else float("nan")
                ),
            }
        ),
        "ok",
    )


def fit_population(
    config: dict[str, Any],
    out_dir: str | Path,
    limit: int | None = 25,
    batch_size: int = 256,
    row_indices_file: str | None = None,
    map_init_file: str | None = None,
) -> None:
    """Fit chunked hierarchical population MAP models with JAX-vmapped Adam."""
    out = ensure_dir(out_dir)
    perf = PerformanceRecorder(_verbose_benchmark(config))
    reporting_level = _reporting_level(config)
    row_indices = _load_row_indices_set(row_indices_file)
    map_init = (
        pd.read_csv(map_init_file).set_index("row_index") if map_init_file else None
    )
    columns = required_catalog_columns(config)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    perf.mark("load_context", n_bands=len(config["bands"]))

    comparison_rows = []
    fit_rows = []
    hyper_rows = []
    trace_rows = []
    sed_manifest_rows = []
    sed_samples_written = 0
    sed_sample_limit = _sed_sample_limit(config)
    total = _progress_total(config["catalog_path"], limit, row_indices)
    with _make_progress_bar(
            total=total, desc="population-fit", unit="galaxy"
    ) as progress:
        chunk_index = 0
        for batch in iter_catalog_batches(
            config["catalog_path"],
            columns=columns,
            batch_size=batch_size,
            limit=limit,
            row_indices=row_indices,
        ):
            perf.mark("read_chunk", chunk_index=chunk_index, n_rows=len(batch))
            batch_result = _fit_dataframe_batch(
                context,
                batch,
                config,
                population=True,
                chunk_index=chunk_index,
                map_init=map_init,
                perf=perf,
            )
            fit_rows.extend(batch_result["fit_rows"])
            comparison_rows.extend(batch_result["comparison_rows"])
            hyper_rows.extend(batch_result["hyper_rows"])
            trace_rows.extend(batch_result["trace_rows"])
            if sed_samples_written < sed_sample_limit:
                new_rows = _write_fit_sed_samples(
                    context,
                    batch,
                    batch_result["fit_rows"],
                    config,
                    out,
                    mode="population_fit",
                    chunk_index=chunk_index,
                    remaining=sed_sample_limit - sed_samples_written,
                )
                sed_manifest_rows.extend(new_rows)
                sed_samples_written += len(new_rows)
                perf.mark(
                    "write_sed_samples",
                    chunk_index=chunk_index,
                    n_rows=len(new_rows),
                )
            perf.mark(
                "fit_population_chunk", chunk_index=chunk_index, n_rows=len(batch)
            )
            _update_progress(
                progress, row_index=int(batch.index[-1]), amount=len(batch)
            )
            chunk_index += 1

    fits = pd.DataFrame(fit_rows)
    comparison = pd.DataFrame(comparison_rows)
    write_dataframe_outputs(fits, out, "population_fit_results", config)
    write_dataframe_outputs(
        comparison, out, "population_fit_photometry_comparison", config
    )
    hyper_frame = pd.DataFrame(hyper_rows)
    if hyper_rows:
        write_dataframe_outputs(hyper_frame, out, "population_hyperparameters", config)
    perf.mark("write_primary_outputs", n_rows=len(fits))
    if trace_rows:
        trace = pd.DataFrame(trace_rows)
        write_dataframe_outputs(trace, out, "population_fit_trace", config)
        write_trace_truth_outputs(
            trace,
            out,
            label="population_fit",
            make_plots=reporting_level == "full",
        )
    write_batch_outputs(
        comparison,
        out,
        label="population_fit",
        reporting_level=reporting_level,
        config=config,
    )
    if sed_manifest_rows:
        pd.DataFrame(sed_manifest_rows).to_csv(
            out / "sed_diagnostics_manifest.csv", index=False
        )
    if reporting_level == "full":
        write_population_corner_outputs(
            fits, list(config["fit"]["free_parameters"]), out, config=config
        )
    write_fit_diagnostic_outputs(
        fits,
        comparison,
        config,
        out,
        label="population_fit",
        hyperparameters=hyper_frame,
    )
    write_nebular_diagnostic_outputs(
        context,
        fits,
        out,
        label="population_fit",
        make_plots=reporting_level == "full",
    )
    perf.mark("write_reports")
    write_json(out / "normalized_config.json", config)
    write_json(
        out / "population_fit_run_config.json",
        {
            "rows_written": len(comparison_rows),
            "limit": limit,
            "batch_size": batch_size,
            "row_indices_file": row_indices_file,
            "map_init_file": map_init_file,
        },
    )
    write_performance_outputs(perf.rows, out, "population_fit")


def _comparison_for_batch(observation, result, params, config):
    from ..model import comparison_rows

    context_values = _row_context(observation.row, params, config)
    for row in comparison_rows(observation, result):
        row["row_index"] = observation.row_index
        row.update(context_values)
        yield row


def _posterior_predictive_rows(
    row_index: int, result, context_values: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for band_index, band in enumerate(result.band_names):
        values = result.posterior_model_mags[:, band_index]
        obs = float(result.observed_mag[band_index])
        med = float(pd.Series(values).quantile(0.50))
        rows.append(
            {
                "row_index": row_index,
                "band": band,
                "observed_mag_ab": obs,
                "sigma_mag": float(result.sigma_mag[band_index]),
                "model_mag_q05": float(pd.Series(values).quantile(0.05)),
                "model_mag_q16": float(pd.Series(values).quantile(0.16)),
                "model_mag_median": med,
                "model_mag_q84": float(pd.Series(values).quantile(0.84)),
                "model_mag_q95": float(pd.Series(values).quantile(0.95)),
                "residual_mag_median_model_minus_observed": med - obs,
                **context_values,
            }
        )
    return rows


def _posterior_sample_rows(row_index: int, result) -> list[dict[str, Any]]:
    names = list(result.samples)
    n_samples = len(result.samples[names[0]]) if names else 0
    rows = []
    for sample_index in range(n_samples):
        rows.append(
            {
                "row_index": row_index,
                "sample_index": sample_index,
                **{name: float(result.samples[name][sample_index]) for name in names},
            }
        )
    return rows


def _posterior_truth_values(context_values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in context_values.items()
        if key.startswith("truth_")
        or key.startswith("truth_source_")
        or key.startswith("truth_kind_")
        or key in {"redshift_truth", "redshift_truth_source", "z_obs", "z_obs_source"}
    }


def _row_context(
    row: dict[str, Any], params: dict[str, float], config: dict[str, Any]
) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    values["z_obs"] = float(params["z_obs"])
    values["dust_parameter_active"] = bool(is_forward_active(config, "dust_av"))
    values["dust_parameter_inferred"] = bool(is_inferred(config, "dust_av"))
    values["dust_model"] = str(config.get("dust_model", "salim"))
    redshift = config.get("redshift", {})
    redshift_initial = str(redshift.get("initial", "catalog_column"))
    if redshift_initial == "random_uniform":
        z_source = "random_uniform"
    elif redshift_initial == "fixed":
        z_source = "fixed_value"
    else:
        z_source = truth_column_from_spec(redshift.get("column")) or "fixed_value"
    values["z_obs_source"] = z_source
    truth_col = redshift.get("truth_column") or config.get("truth", {}).get(
        "redshift_column"
    )
    truth_col = truth_column_from_spec(truth_col)
    if truth_col and truth_col in row and pd.notna(row[truth_col]):
        values["redshift_truth_source"] = truth_col
        values["redshift_truth"] = float(row[truth_col])
        values["delta_z_obs_minus_truth"] = values["z_obs"] - values["redshift_truth"]
    for key, value in params.items():
        values[f"param_{key}"] = float(value)
    for truth_name, column in (
        config.get("truth", {}).get("parameter_columns") or {}
    ).items():
        truth_value = truth_value_from_spec(row, column)
        if truth_value is not None:
            truth_column = truth_column_from_spec(column)
            if truth_column:
                values[f"truth_source_{truth_name}"] = truth_column
            values[f"truth_kind_{truth_name}"] = _truth_kind(truth_name, column)
            values[f"truth_{truth_name}"] = float(truth_value)
            param_key = f"param_{truth_name}"
            if param_key in values:
                values[f"delta_{truth_name}"] = (
                    values[param_key] - values[f"truth_{truth_name}"]
                )
    for column in config.get("extra_columns", []):
        if column in row and pd.notna(row[column]):
            values[f"catalog_{column}"] = float(row[column])
    return values


def _truth_kind(truth_name: str, spec: Any) -> str:
    if truth_name in {"dust_av", "log10_metallicity"}:
        return "proxy"
    if isinstance(spec, dict) and (
        "scale" in spec
        or "offset" in spec
        or spec.get("transform") not in {None, "linear"}
    ):
        return "proxy"
    return "direct"


def _fit_dataframe_batch(
    context,
    batch: pd.DataFrame,
    config: dict[str, Any],
    population: bool = False,
    chunk_index: int = 0,
    map_init: pd.DataFrame | None = None,
    perf: PerformanceRecorder | None = None,
) -> dict[str, list[dict[str, Any]]]:
    observed_mag, observed_flux, sigma_mag = _photometry_arrays(batch, config["bands"])
    flux_error = _flux_error_arrays(batch, config["bands"], observed_flux, sigma_mag)
    valid_band_mask = _valid_band_mask(
        observed_mag, observed_flux, sigma_mag, flux_error, config["fit"]
    )
    if perf is not None:
        perf.mark("prepare_photometry", chunk_index=chunk_index, n_rows=len(batch))
    base_rows = [
        parameters_for_row(
            config["model"]["fixed_parameters"],
            config["model"].get("parameter_columns", {}),
            row.to_dict(),
            config.get("redshift", {}),
        )
        for _, row in batch.iterrows()
    ]
    if perf is not None:
        perf.mark("prepare_parameters", chunk_index=chunk_index, n_rows=len(batch))
    if population:
        truth_theta = _truth_parameter_matrix(
            batch, config, list(config["fit"]["free_parameters"])
        )
        if perf is not None:
            perf.mark("prepare_truth", chunk_index=chunk_index, n_rows=len(batch))
        pop_result = fit_population_batch_adam(
            context,
            base_rows,
            observed_mag,
            sigma_mag,
            config["fit"],
            initial_theta=_initial_theta_from_map(batch, config, map_init),
            truth_theta=truth_theta,
            observed_flux=observed_flux,
            flux_error=flux_error,
        )
        if perf is not None:
            perf.mark("jax_optimize", chunk_index=chunk_index, n_rows=len(batch))
        fit_result = pop_result.batch
        hyper_rows = [
            {
                "chunk_index": chunk_index,
                "n_galaxies": len(batch),
                "kind": "gaussian",
                "parameter": name,
                "population_mu": pop_result.hyper_mu[name],
                "population_sigma": pop_result.hyper_sigma[name],
                "loss": pop_result.loss,
                "device": fit_result.device,
            }
            for name in fit_result.free_parameter_names
        ]
        for relation in pop_result.hyper_relations:
            hyper_rows.append(
                {
                    "chunk_index": chunk_index,
                    "n_galaxies": len(batch),
                    "kind": "relation",
                    "parameter": relation["target_parameter"],
                    "target_parameter": relation["target_parameter"],
                    "predictor_parameter": relation["predictor_parameter"],
                    "population_pivot": relation["pivot"],
                    "population_intercept": relation["intercept"],
                    "population_slope": relation["slope"],
                    "population_sigma": relation["sigma"],
                    "loss": pop_result.loss,
                    "device": fit_result.device,
                }
            )
    else:
        truth_theta = _truth_parameter_matrix(
            batch, config, list(config["fit"]["free_parameters"])
        )
        if perf is not None:
            perf.mark("prepare_truth", chunk_index=chunk_index, n_rows=len(batch))
        fit_result = fit_galaxy_batch_adam(
            context,
            base_rows,
            observed_mag,
            sigma_mag,
            config["fit"],
            truth_theta=truth_theta,
            observed_flux=observed_flux,
            flux_error=flux_error,
        )
        if perf is not None:
            perf.mark("jax_optimize", chunk_index=chunk_index, n_rows=len(batch))
        hyper_rows = []

    fit_rows = []
    comparison_rows = []
    band_names = [band["name"] for band in config["bands"]]
    filter_curves = [context.filters[name] for name in band_names]
    param_matrix = fit_result.best_parameter_matrix
    derived = predict_batch_derived(context, fit_result.parameter_names, param_matrix)
    if perf is not None:
        perf.mark("derive_quantities", chunk_index=chunk_index, n_rows=len(batch))
    for local_index, (row_index, row) in enumerate(batch.iterrows()):
        params = {
            name: float(param_matrix[local_index, param_index])
            for param_index, name in enumerate(fit_result.parameter_names)
        }
        derived_values = {
            f"fit_{name}": float(values[local_index])
            for name, values in derived.items()
        }
        context_values = {
            **_row_context(row.to_dict(), params, config),
            **{f"fit_{key}": value for key, value in params.items()},
            **derived_values,
        }
        if "z_obs" in fit_result.free_parameter_names:
            context_values["z_obs_source"] = "DSPS fit"
        context_values["redshift_initial_mode"] = str(
            config.get("redshift", {}).get("initial", "catalog_column")
        )
        context_values["redshift_prior_mode"] = str(
            (config.get("redshift", {}).get("prior_z") or {}).get("mode", "none")
        )
        z_initial = (
            _initial_value(
                config["fit"]["free_parameters"]["z_obs"],
                "z_obs",
                base_rows[local_index],
            )
            if "z_obs" in config["fit"]["free_parameters"]
            else base_rows[local_index].get("z_obs", np.nan)
        )
        n_bands = len(config["bands"])
        n_valid_bands = int(np.asarray(valid_band_mask[local_index]).sum())
        flux_values = np.asarray(observed_flux[local_index], dtype=float)
        flux_errors = np.asarray(flux_error[local_index], dtype=float)
        finite_flux_error = np.isfinite(flux_errors) & (flux_errors > 0.0)
        n_nondetected_bands = int(
            (np.isfinite(flux_values) & (flux_values <= 0.0) & finite_flux_error).sum()
        )
        n_masked_bands = int(n_bands - n_valid_bands)
        n_free_effective = len(config["fit"]["free_parameters"])
        dof = max(n_valid_bands - n_free_effective, 1)
        chi2_value = float(fit_result.chi2[local_index])
        photometric_objective = float(fit_result.photometric_objective[local_index])
        photometric_likelihood = str(
            config["fit"].get("photometric_likelihood", "gaussian")
        )
        student_t_dof = float(config["fit"].get("student_t_dof", 2.0))
        fit_quality_metric = (
            "student_t_neg2loglike"
            if photometric_likelihood == "student_t"
            else "gaussian_chi2"
        )
        fit_rows.append(
            {
                "row_index": int(row_index),
                "chunk_index": int(chunk_index),
                "success": bool(fit_result.success[local_index]),
                "message": fit_result.message,
                "chi2": chi2_value,
                "chi2_per_band": chi2_value / max(n_valid_bands, 1),
                "reduced_chi2": chi2_value / dof,
                "reduced_chi2_dof": chi2_value / dof,
                "gaussian_chi2": chi2_value,
                "gaussian_chi2_per_band": chi2_value / max(n_valid_bands, 1),
                "reduced_gaussian_chi2": chi2_value / dof,
                "photometric_objective": photometric_objective,
                "photometric_objective_per_band": photometric_objective
                / max(n_valid_bands, 1),
                "fit_quality": photometric_objective,
                "fit_quality_per_band": photometric_objective / max(n_valid_bands, 1),
                "reduced_fit_quality": photometric_objective / dof,
                "fit_quality_metric": fit_quality_metric,
                "photometric_likelihood": photometric_likelihood,
                "student_t_dof": student_t_dof,
                "gradient_norm": float(fit_result.gradient_norm[local_index]),
                "n_bands": n_bands,
                "n_valid_bands": n_valid_bands,
                "n_masked_bands": n_masked_bands,
                "n_nondetected_bands": n_nondetected_bands,
                "n_upper_limit_bands": 0,
                "nondetection_policy": str(
                    config.get("selection", {}).get("nondetection_policy", "drop")
                ),
                "n_free_effective": n_free_effective,
                "dof": dof,
                "z_initial": float(z_initial),
                "device": fit_result.device,
                **{f"fit_{key}": value for key, value in params.items()},
                **derived_values,
                **context_values,
            }
        )
        for band_index, band in enumerate(config["bands"]):
            model_mag = float(fit_result.model_mags[local_index, band_index])
            obs_mag = float(observed_mag[local_index, band_index])
            obs_flux = float(observed_flux[local_index, band_index])
            obs_flux_error = float(flux_error[local_index, band_index])
            sigma = float(sigma_mag[local_index, band_index])
            model_flux = abmag_to_flux_fnu_cgs(model_mag)
            flux_ratio = model_flux / obs_flux if obs_flux > 0 else float("nan")
            residual = obs_mag - model_mag
            chi_flux = (
                (model_flux - obs_flux) / obs_flux_error
                if obs_flux_error > 0
                else float("nan")
            )
            likelihood_space = str(config["fit"].get("likelihood_space", "flux")).lower()
            if likelihood_space == "flux":
                sigma_likelihood = float(
                    np.sqrt(
                        obs_flux_error**2
                        + (
                            float(config["fit"].get("flux_error_floor_frac", 0.0))
                            * obs_flux
                        )
                        ** 2
                        + float(config["fit"].get("flux_error_jitter", 0.0)) ** 2
                    )
                )
                chi_likelihood = (
                    (obs_flux - model_flux) / sigma_likelihood
                    if sigma_likelihood > 0
                    else float("nan")
                )
            else:
                sigma_likelihood = sigma
                chi_likelihood = residual / sigma if sigma > 0 else float("nan")
            chi2_contribution = chi_likelihood**2
            if photometric_likelihood == "student_t":
                objective_contribution = (
                    (student_t_dof + 1.0)
                    * np.log1p(chi2_contribution / student_t_dof)
                    if np.isfinite(chi2_contribution)
                    else float("nan")
                )
            else:
                objective_contribution = chi2_contribution
            comparison_rows.append(
                {
                    "row_index": int(row_index),
                    "band": band["name"],
                    "column": band["column"],
                    "effective_wavelength_angstrom": filter_curves[
                        band_index
                    ].effective_wavelength,
                    "observed_flux_fnu_cgs": obs_flux,
                    "observed_flux_error_fnu_cgs": obs_flux_error,
                    "band_used_in_likelihood": bool(
                        valid_band_mask[local_index, band_index]
                    ),
                    "band_is_nondetection": bool(
                        np.isfinite(obs_flux) and obs_flux <= 0.0 and obs_flux_error > 0.0
                    ),
                    "n_valid_bands": n_valid_bands,
                    "n_masked_bands": n_masked_bands,
                    "n_nondetected_bands": n_nondetected_bands,
                    "n_upper_limit_bands": 0,
                    "n_free_effective": n_free_effective,
                    "dof": dof,
                    "observed_mag_ab": obs_mag,
                    "sigma_mag": sigma,
                    "model_flux_fnu_cgs": model_flux,
                    "model_mag_ab": model_mag,
                    "residual_mag_observed_minus_model": residual,
                    "residual_mag_model_minus_observed": -residual,
                    "flux_ratio_model_over_observed": flux_ratio,
                    "fractional_flux_residual_model_minus_observed": flux_ratio - 1.0,
                    "chi": residual / sigma if sigma > 0 else float("nan"),
                    "chi_flux": chi_flux,
                    "chi_likelihood": chi_likelihood,
                    "photometric_objective_contribution": objective_contribution,
                    "likelihood_sigma": sigma_likelihood,
                    "photometric_likelihood": photometric_likelihood,
                    "student_t_dof": student_t_dof,
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
    if perf is not None:
        perf.mark("materialize_rows", chunk_index=chunk_index, n_rows=len(batch))
    return {
        "fit_rows": fit_rows,
        "comparison_rows": comparison_rows,
        "hyper_rows": hyper_rows,
        "trace_rows": trace_rows,
    }


def _truth_parameter_matrix(
    batch: pd.DataFrame, config: dict[str, Any], free_names: list[str]
) -> np.ndarray | None:
    redshift = config.get("redshift", {})
    redshift_truth_spec = redshift.get("truth_column") or config.get("truth", {}).get(
        "redshift_column"
    )
    parameter_specs = config.get("truth", {}).get("parameter_columns") or {}
    rows = []
    for _, row in batch.iterrows():
        values = []
        row_dict = row.to_dict()
        for name in free_names:
            spec = redshift_truth_spec if name == "z_obs" else parameter_specs.get(name)
            truth_value = truth_value_from_spec(row_dict, spec) if spec else None
            values.append(float(truth_value) if truth_value is not None else np.nan)
        rows.append(values)
    truth = np.asarray(rows, dtype=float)
    return truth if np.isfinite(truth).any() else None


def _initial_theta_from_map(
    batch: pd.DataFrame, config: dict[str, Any], map_init: pd.DataFrame | None
) -> np.ndarray | None:
    if map_init is None:
        return None
    free_names = list(config["fit"]["free_parameters"])
    rows = []
    for row_index in batch.index:
        if row_index not in map_init.index:
            return None
        item = map_init.loc[row_index]
        if isinstance(item, pd.DataFrame):
            item = item.iloc[0]
        values = []
        for name in free_names:
            col = f"fit_{name}"
            if col not in item or not np.isfinite(item[col]):
                return None
            values.append(float(item[col]))
        rows.append(values)
    return np.asarray(rows, dtype=float)


def _initial_value(
    spec: dict[str, Any], name: str, base_params: dict[str, float]
) -> float:
    value = spec.get("initial", base_params.get(name, 0.0))
    if isinstance(value, str):
        value = base_params[name] if value == "from_base" else value
    return float(value)


def _forward_dataframe_batch(
    context, batch: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
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
    parameter_matrix = pd.DataFrame(base_rows, columns=parameter_names).to_numpy(
        dtype=float
    )
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
                    "effective_wavelength_angstrom": filter_curves[
                        band_index
                    ].effective_wavelength,
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


def _photometry_arrays(
    batch: pd.DataFrame, band_configs: list[dict[str, Any]]
) -> tuple[Any, Any, Any]:
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
            raise ValueError(
                f"Unsupported photometry units for {band['name']}: {units}"
            )
        mag_columns.append(mag)
        flux_columns.append(flux)
        sigma_columns.append(_sigma_mag_array(batch, band, flux, units))
    return (
        pd.DataFrame(mag_columns).transpose().to_numpy(dtype=float),
        pd.DataFrame(flux_columns).transpose().to_numpy(dtype=float),
        pd.DataFrame(sigma_columns).transpose().to_numpy(dtype=float),
    )


def _sigma_mag_array(
    batch: pd.DataFrame, band: dict[str, Any], flux: Any, units: str
) -> list[float]:
    fallback = float(band.get("sigma_mag", 0.05))
    error_column = band.get("error_column")
    if not error_column or error_column not in batch:
        return [fallback] * len(batch)

    raw_errors = batch[str(error_column)].astype(float).to_numpy()
    error_units = band.get("error_units", units)
    floor = band.get("sigma_mag_floor")
    ceiling = band.get("sigma_mag_ceiling")
    sigma = []
    flux_values = np.asarray(flux, dtype=float)
    for flux_value, raw_error in zip(flux_values, raw_errors, strict=True):
        value = float("nan")
        if np.isfinite(raw_error) and raw_error > 0.0:
            if error_units == "abmag":
                value = float(raw_error)
                if floor is not None and np.isfinite(floor):
                    value = max(value, float(floor))
                if ceiling is not None and np.isfinite(ceiling):
                    value = min(value, float(ceiling))
            elif error_units == "fnu_cgs":
                value = flux_error_to_sigma_mag(
                    float(flux_value), float(raw_error), floor=floor, ceiling=ceiling
                )
            elif error_units in {"microjy", "ujy"}:
                value = flux_error_to_sigma_mag(
                    float(flux_value),
                    microjy_to_flux_fnu_cgs(float(raw_error)),
                    floor=floor,
                    ceiling=ceiling,
                )
            else:
                raise ValueError(
                    f"Unsupported photometry error units for {band['name']}: {error_units}"
                )
        sigma.append(value if np.isfinite(value) and value > 0.0 else fallback)
    return sigma


def _flux_error_arrays(
    batch: pd.DataFrame,
    band_configs: list[dict[str, Any]],
    observed_flux: np.ndarray,
    sigma_mag: np.ndarray,
) -> np.ndarray:
    columns = []
    for band_index, band in enumerate(band_configs):
        flux = np.asarray(observed_flux[:, band_index], dtype=float)
        fallback = np.asarray(
            magerr_to_fluxerr_fnu_cgs(flux, sigma_mag[:, band_index]), dtype=float
        )
        error_column = band.get("error_column")
        if not error_column or error_column not in batch:
            columns.append(fallback)
            continue
        raw_errors = batch[str(error_column)].astype(float).to_numpy()
        error_units = band.get("error_units", band.get("units", "fnu_cgs"))
        if error_units == "fnu_cgs":
            errors = raw_errors
        elif error_units in {"microjy", "ujy"}:
            errors = np.asarray(
                [microjy_to_flux_fnu_cgs(value) for value in raw_errors], dtype=float
            )
        elif error_units == "abmag":
            errors = np.asarray(
                magerr_to_fluxerr_fnu_cgs(flux, raw_errors), dtype=float
            )
        else:
            raise ValueError(
                f"Unsupported photometry error units for {band['name']}: {error_units}"
            )
        errors = np.where(np.isfinite(errors) & (errors > 0.0), errors, fallback)
        columns.append(errors)
    return pd.DataFrame(columns).transpose().to_numpy(dtype=float)


def _valid_band_mask(
    observed_mag: np.ndarray,
    observed_flux: np.ndarray,
    sigma_mag: np.ndarray,
    flux_error: np.ndarray,
    fit_config: dict[str, Any],
) -> np.ndarray:
    if str(fit_config.get("likelihood_space", "flux")).lower() == "flux":
        sigma = np.asarray(flux_error, dtype=float)
        observed = np.asarray(observed_flux, dtype=float)
    else:
        sigma = np.asarray(sigma_mag, dtype=float)
        observed = np.asarray(observed_mag, dtype=float)
    return np.isfinite(observed) & np.isfinite(sigma) & (sigma > 0.0)


def _load_row_indices_set(row_indices_file: str | None) -> set[int] | None:
    if not row_indices_file:
        return None
    return set(load_row_indices(row_indices_file))


def _progress_total(
    catalog_path: str | Path, limit: int | None, row_indices: set[int] | None = None
) -> int | None:
    if row_indices is not None:
        if limit is None:
            return len(row_indices)
        return min(int(limit), len(row_indices))
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

    def __enter__(self) -> _NullProgress:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _make_progress_bar(total: int | None, desc: str, unit: str) -> Any:
    if tqdm is None:
        return _NullProgress()
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.2,
        smoothing=0.05,
    )


def _update_progress(progress: Any, row_index: int, amount: int = 1) -> None:
    progress.update(amount)
    progress.set_postfix_str(f"row={row_index}", refresh=False)
