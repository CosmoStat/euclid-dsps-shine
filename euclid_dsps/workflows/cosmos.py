"""COSMOS-template pseudo-SED workflow entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..cosmos import (
    compare_cosmos_to_dsps_rest_sed,
    cosmos_abs_flux_rows,
    cosmos_catalog_columns,
    cosmos_diagnostic_row,
    cosmos_sed_long_rows,
    grouped_metric_summary,
    load_cosmos_sed_resources,
    observed_photometry_chi2_summary,
    observed_photometry_target_rows,
    photometry_target_sets,
    population_validation_summary,
    reconstruct_cosmos_proxy_sed,
    resolve_value_added_data_dir,
    validate_cosmos_catalog,
)
from ..filters import load_filters
from ..fit import fit_galaxy_batch_adam, fit_population_batch_adam
from ..io import (
    abmag_to_flux_fnu_cgs,
    ensure_dir,
    iter_catalog_batches,
    read_catalog,
    required_catalog_columns,
    write_dataframe_outputs,
    write_json,
)
from ..model import (
    BatchSedResult,
    ModelResult,
    load_context,
    parameters_for_row,
    predict_batch_seds,
)
from ..performance import PerformanceRecorder, write_performance_outputs
from ..reporting.core import write_batch_outputs, write_fit_diagnostic_outputs
from ..reporting.cosmos import (
    plot_branch1_metric_summary,
    plot_branch1_residual_heatmap,
    plot_cosmos_dsps_rest_comparison,
    plot_cosmos_sed_example,
    plot_cosmos_sed_sample_set,
    plot_fraction_diagnostics,
    plot_observed_flux_residuals,
    plot_population_validation_summary,
    plot_rest_color_residuals,
    plot_synthetic_vs_catalog_abs_flux,
    plot_template_pair_heatmap,
    plot_worst_sed_grid,
    write_cosmos_output_manifest,
)
from .core import (
    _make_progress_bar,
    _photometry_arrays,
    _reporting_level,
    _row_context,
    _truth_parameter_matrix,
    _update_progress,
    _verbose_benchmark,
)


def reconstruct_cosmos_seds(
    config: dict[str, Any],
    out_dir: str | Path,
    limit: int | None = 10,
    batch_size: int = 1000,
    index: int | None = None,
    compare_dsps: bool = False,
    fit_dsps: bool = False,
    population_dsps: bool = False,
    sample_plot_count: int | None = None,
) -> pd.DataFrame:
    """Run COSMOS-template SED reconstruction for a small catalog sample."""
    out = ensure_dir(out_dir)
    perf = PerformanceRecorder(_verbose_benchmark(config))
    full_report = _reporting_level(config) == "full"
    available_columns = _available_catalog_columns(config["catalog_path"])
    cosmos_config = config.get("cosmos_sed", {})
    validation_columns = cosmos_catalog_columns(
        config, available_columns=available_columns, include_optional=True
    )
    selected = _read_selected_rows(
        config,
        validation_columns,
        limit=limit,
        batch_size=batch_size,
        index=index,
    )
    validation_report = validate_cosmos_catalog(
        selected, config, available_columns=available_columns
    )
    validation_report["missing_branch2_target_columns"] = _missing_branch2_columns(
        config, available_columns
    )
    validation_report["value_added_data"] = _value_added_data_report(cosmos_config)
    write_json(out / "cosmos_sed_validation.json", validation_report)
    perf.mark("read_catalog", n_rows=len(selected))

    resources = load_cosmos_sed_resources(cosmos_config)
    filters = load_filters(config["bands"])
    context = None
    dsps_mode = _dsps_mode(compare_dsps, fit_dsps, population_dsps)
    if dsps_mode != "none":
        context = load_context(
            config["ssp_path"],
            filters,
            n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
            cosmos_config=cosmos_config,
            nebular_emission=config.get("nebular_emission", "ssp_flux"),
            model_config=config.get("model"),
        )
    perf.mark("load_resources", n_bands=len(config["bands"]), dsps_mode=dsps_mode)

    diagnostics: list[dict[str, Any]] = []
    abs_flux_rows: list[dict[str, Any]] = []
    sed_frames: list[pd.DataFrame] = []
    branch1_metric_rows: list[dict[str, Any]] = []
    branch1_comparison_frames: list[pd.DataFrame] = []
    branch2_rows: list[dict[str, Any]] = []
    likelihood_rows: list[dict[str, Any]] = []
    dsps_fit_rows: list[dict[str, Any]] = []
    dsps_hyper_rows: list[dict[str, Any]] = []
    dsps_trace_rows: list[dict[str, Any]] = []
    manifest = ["cosmos_sed_validation.json"]
    first_result = None
    first_branch1 = None
    sample_results: list[Any] = []
    sample_comparison_frames: list[pd.DataFrame] = []
    plot_count = int(
        sample_plot_count
        if sample_plot_count is not None
        else cosmos_config.get("sample_plot_count", 12)
    )

    total = len(selected)
    with _make_progress_bar(total=total, desc="cosmos-check", unit="galaxy") as progress:
        for chunk_index, batch in enumerate(_dataframe_chunks(selected, batch_size)):
            cosmos_results = []
            for row_index, row in batch.iterrows():
                row_dict = row.to_dict()
                result = reconstruct_cosmos_proxy_sed(
                    row_dict,
                    int(row_index),
                    resources,
                    filters,
                    config["bands"],
                    cosmos_config,
                )
                cosmos_results.append(result)
                diagnostics.append(cosmos_diagnostic_row(result))
                abs_flux_rows.extend(cosmos_abs_flux_rows(result))
                sed_frames.append(cosmos_sed_long_rows(result))
                if len(sample_results) < plot_count:
                    sample_results.append(result)
                if first_result is None:
                    first_result = result
                    _print_abs_flux_table(result)

            if context is not None and cosmos_results:
                batch_dsps = _batch_dsps_results(
                    context=context,
                    batch=batch,
                    config=config,
                    mode=dsps_mode,
                    chunk_index=chunk_index,
                )
                dsps_fit_rows.extend(batch_dsps["fit_rows"])
                dsps_hyper_rows.extend(batch_dsps["hyper_rows"])
                dsps_trace_rows.extend(batch_dsps["trace_rows"])
                likelihood_rows.extend(batch_dsps["likelihood_rows"])
                for local_index, (row_index, row) in enumerate(batch.iterrows()):
                    dsps_result = _model_result_from_batch(
                        batch_dsps["sed_result"],
                        local_index,
                        row.to_dict(),
                        int(row_index),
                        context,
                        config,
                    )
                    metrics, comparison = compare_cosmos_to_dsps_rest_sed(
                        cosmos_results[local_index],
                        dsps_result,
                        filters,
                        cosmos_config,
                    )
                    metrics["dsps_mode"] = dsps_mode
                    branch1_metric_rows.append(metrics)
                    if not comparison.empty:
                        comparison["dsps_mode"] = dsps_mode
                        branch1_comparison_frames.append(comparison)
                        if len(sample_comparison_frames) < plot_count:
                            sample_comparison_frames.append(comparison)
                        if first_branch1 is None:
                            first_branch1 = comparison
                branch2_rows.extend(
                    observed_photometry_target_rows(
                        row.to_dict(),
                        int(row_index),
                        dsps_result,
                        config["bands"],
                        target_set_names=cosmos_config.get(
                            "observed_photometry_target_sets"
                        ),
                    )
                )
            _update_progress(
                progress, row_index=int(batch.index[-1]), amount=len(batch)
            )
            perf.mark("cosmos_chunk", chunk_index=chunk_index, n_rows=len(batch))

    diagnostics_frame = pd.DataFrame(diagnostics)
    manifest.extend(
        write_dataframe_outputs(
            diagnostics_frame, out, "cosmos_sed_diagnostics", config
        )
    )
    perf.mark("write_primary_cosmos_outputs", n_rows=len(diagnostics_frame))

    abs_flux_frame = pd.DataFrame(abs_flux_rows)
    manifest.extend(
        write_dataframe_outputs(
            abs_flux_frame, out, "synthetic_vs_catalog_abs_flux", config
        )
    )
    if full_report:
        plot_synthetic_vs_catalog_abs_flux(
            abs_flux_frame, out / "synthetic_vs_catalog_abs_flux.png"
        )
        manifest.append("synthetic_vs_catalog_abs_flux.png")

    if sed_frames:
        sed_frame = pd.concat(sed_frames, ignore_index=True)
        sed_frame.to_parquet(out / "cosmos_seds.parquet", index=False)
        manifest.append("cosmos_seds.parquet")
        first_sed = sed_frames[0]
        first_sed.to_csv(out / "cosmos_sed_example.csv", index=False)
        manifest.append("cosmos_sed_example.csv")
    if first_result is not None:
        if full_report:
            plot_cosmos_sed_example(first_result, out / "cosmos_sed_example.png")
            manifest.append("cosmos_sed_example.png")
    if sample_results and full_report:
        plot_cosmos_sed_sample_set(
            sample_results,
            out / "cosmos_sed_sample_set.png",
            max_seds=plot_count,
            comparisons=sample_comparison_frames,
        )
        manifest.append("cosmos_sed_sample_set.png")

    if full_report:
        plot_template_pair_heatmap(
            diagnostics_frame, out / "cosmos_template_pair_heatmap.png"
        )
        manifest.append("cosmos_template_pair_heatmap.png")
        plot_fraction_diagnostics(
            diagnostics_frame, out / "cosmos_fraction_diagnostics.png"
        )
        manifest.append("cosmos_fraction_diagnostics.png")

    if branch1_metric_rows:
        branch1 = pd.DataFrame(branch1_metric_rows)
        manifest.extend(
            write_dataframe_outputs(branch1, out, "branch1_rest_sed_metrics", config)
        )
        grouped = grouped_metric_summary(branch1, "rms_log_sed_residual")
        grouped.to_csv(out / "branch1_rest_sed_metrics_by_group.csv", index=False)
        manifest.append("branch1_rest_sed_metrics_by_group.csv")
        if full_report:
            plot_branch1_metric_summary(branch1, out / "branch1_rest_sed_metrics.png")
            manifest.append("branch1_rest_sed_metrics.png")
            plot_rest_color_residuals(branch1, out / "branch1_rest_color_residuals.png")
            manifest.append("branch1_rest_color_residuals.png")
            plot_branch1_residual_heatmap(
                branch1, out / "branch1_rms_residual_heatmap.png"
            )
            manifest.append("branch1_rms_residual_heatmap.png")
        population = population_validation_summary(
            branch1, "rms_log_sed_residual", "branch1_rest_sed_rms"
        )
        if not population.empty:
            population.to_csv(out / "branch1_population_validation.csv", index=False)
            manifest.append("branch1_population_validation.csv")
            if full_report:
                plot_population_validation_summary(
                    population, out / "branch1_population_validation.png"
                )
                manifest.append("branch1_population_validation.png")
    if branch1_comparison_frames:
        branch1_comparison = pd.concat(branch1_comparison_frames, ignore_index=True)
        manifest.extend(
            write_dataframe_outputs(
                branch1_comparison, out, "branch1_rest_sed_comparison", config
            )
        )
        if branch1_metric_rows and full_report:
            plot_worst_sed_grid(
                pd.DataFrame(branch1_metric_rows),
                branch1_comparison,
                out / "branch1_worst_sed_grid.png",
                n=16,
            )
            manifest.append("branch1_worst_sed_grid.png")
    if first_branch1 is not None and full_report:
        row_index = int(first_branch1["row_index"].iloc[0])
        plot_cosmos_dsps_rest_comparison(
            first_branch1, out / "branch1_rest_sed_comparison_example.png", row_index
        )
        manifest.append("branch1_rest_sed_comparison_example.png")

    if branch2_rows:
        branch2 = pd.DataFrame(branch2_rows)
        manifest.extend(
            write_dataframe_outputs(
                branch2, out, "branch2_observed_photometry_metrics", config
            )
        )
        chi2 = observed_photometry_chi2_summary(branch2)
        if not chi2.empty:
            chi2.to_csv(out / "branch2_observed_photometry_chi2.csv", index=False)
            manifest.append("branch2_observed_photometry_chi2.csv")
        grouped = grouped_metric_summary(branch2, "relative_flux_residual")
        grouped.to_csv(
            out / "branch2_observed_photometry_metrics_by_group.csv", index=False
        )
        manifest.append("branch2_observed_photometry_metrics_by_group.csv")
        if full_report:
            plot_observed_flux_residuals(
                branch2, out / "branch2_observed_flux_residuals.png"
            )
            manifest.append("branch2_observed_flux_residuals.png")
        population = population_validation_summary(
            branch2, "relative_flux_residual", "branch2_relative_flux_residual"
        )
        if not population.empty:
            population.to_csv(out / "branch2_population_validation.csv", index=False)
            manifest.append("branch2_population_validation.csv")
            if full_report:
                plot_population_validation_summary(
                    population, out / "branch2_population_validation.png"
                )
                manifest.append("branch2_population_validation.png")

    if dsps_fit_rows:
        dsps_fit_frame = pd.DataFrame(dsps_fit_rows)
        manifest.extend(
            write_dataframe_outputs(
                dsps_fit_frame, out, "cosmos_dsps_fit_results", config
            )
        )
    else:
        dsps_fit_frame = pd.DataFrame()
    if dsps_hyper_rows:
        dsps_hyper_frame = pd.DataFrame(dsps_hyper_rows)
        manifest.extend(
            write_dataframe_outputs(
                dsps_hyper_frame,
                out,
                "cosmos_dsps_population_hyperparameters",
                config,
            )
        )
        write_json(
            out / "cosmos_dsps_population_report.json",
            {
                "model_kind": "chunk_regularized_population_map",
                "is_learned_population_model": False,
                "interpretation": (
                    "The current population mode jointly regularizes fitted "
                    "parameters inside each processed chunk. It is useful for "
                    "stabilizing MAP fits and producing hyperparameter "
                    "diagnostics, but it is not a learned galaxy-population "
                    "prior comparable to pop-cosmos."
                ),
                "recommended_use": (
                    "Use the population validation CSV/PNG outputs to identify "
                    "where residuals depend on color_kind, redshift, magnitude, "
                    "SFR, metallicity, template pair, or dust curve pair."
                ),
            },
        )
        manifest.append("cosmos_dsps_population_report.json")
    else:
        dsps_hyper_frame = pd.DataFrame()
    if dsps_trace_rows:
        manifest.extend(
            write_dataframe_outputs(
                pd.DataFrame(dsps_trace_rows), out, "cosmos_dsps_fit_trace", config
            )
        )

    if likelihood_rows:
        likelihood = pd.DataFrame(likelihood_rows)
        manifest.extend(
            write_dataframe_outputs(
                likelihood,
                out,
                "cosmos_dsps_likelihood_photometry_comparison",
                config,
            )
        )
        write_batch_outputs(
            likelihood,
            out,
            label="cosmos_dsps_likelihood",
            reporting_level=_reporting_level(config),
            config=config,
        )
        if not dsps_fit_frame.empty:
            write_fit_diagnostic_outputs(
                dsps_fit_frame,
                likelihood,
                config,
                out,
                label="cosmos_dsps_likelihood",
                hyperparameters=dsps_hyper_frame,
            )
            manifest.extend(
                [
                    "cosmos_dsps_likelihood_parameter_audit.csv",
                    "cosmos_dsps_likelihood_objective_components.csv",
                ]
            )
        manifest.extend(
            [
                "cosmos_dsps_likelihood_summary.json",
                "cosmos_dsps_likelihood_summary_by_band.csv",
                "cosmos_dsps_likelihood_summary_by_galaxy.csv",
                "cosmos_dsps_likelihood_truth_metrics.csv",
            ]
        )
        if full_report:
            manifest.extend(
                [
                    "cosmos_dsps_likelihood_dashboard.png",
                    "cosmos_dsps_likelihood_observed_vs_model.png",
                    "cosmos_dsps_likelihood_residuals_by_band.png",
                    "cosmos_dsps_likelihood_redshift_truth.png",
                    "cosmos_dsps_likelihood_parameter_truth.png",
                ]
            )
    perf.mark("write_reports")

    write_json(out / "normalized_config.json", config)
    manifest.append("normalized_config.json")
    write_json(
        out / "cosmos_sed_run_config.json",
        {
            "limit": limit,
            "batch_size": batch_size,
            "index": index,
            "compare_dsps": compare_dsps,
            "fit_dsps": fit_dsps,
            "population_dsps": population_dsps,
            "dsps_mode": dsps_mode,
            "sample_plot_count": plot_count,
            "n_rows": int(len(selected)),
            "template_list_path": str(resources.template_list_path),
            "extinction_dir": str(resources.extinction_dir),
            "n_templates": len(resources.templates),
            "n_extinction_curves": len(resources.extinction_curves),
        },
    )
    manifest.append("cosmos_sed_run_config.json")
    write_performance_outputs(perf.rows, out, "cosmos_sed")
    manifest.extend(
        ["cosmos_sed_performance_benchmark.csv", "cosmos_sed_performance_summary.json"]
    )
    write_cosmos_output_manifest(out, manifest)
    return diagnostics_frame


def _dsps_mode(compare_dsps: bool, fit_dsps: bool, population_dsps: bool) -> str:
    if population_dsps:
        return "population"
    if fit_dsps:
        return "map"
    if compare_dsps:
        return "forward"
    return "none"


def _dataframe_chunks(frame: pd.DataFrame, batch_size: int):
    size = max(int(batch_size), 1)
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size]


def _batch_dsps_results(
    context,
    batch: pd.DataFrame,
    config: dict[str, Any],
    mode: str,
    chunk_index: int,
) -> dict[str, Any]:
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
    fit_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    likelihood_rows: list[dict[str, Any]] = []
    observed_mag = None
    observed_flux = None
    sigma_mag = None

    if mode == "forward":
        parameter_matrix = pd.DataFrame(base_rows, columns=parameter_names).to_numpy(
            dtype=float
        )
    elif mode in {"map", "population"}:
        observed_mag, observed_flux, sigma_mag = _photometry_arrays(
            batch, config["bands"]
        )
        truth_theta = _truth_parameter_matrix(
            batch, config, list(config["fit"]["free_parameters"])
        )
        if mode == "population":
            pop_result = fit_population_batch_adam(
                context,
                base_rows,
                observed_mag,
                sigma_mag,
                config["fit"],
                truth_theta=truth_theta,
            )
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
            fit_result = fit_galaxy_batch_adam(
                context,
                base_rows,
                observed_mag,
                sigma_mag,
                config["fit"],
                truth_theta=truth_theta,
            )
        parameter_names = fit_result.parameter_names
        parameter_matrix = fit_result.best_parameter_matrix
        trace_rows = [
            {
                "chunk_index": chunk_index,
                "dsps_mode": mode,
                **entry,
            }
            for entry in fit_result.trace
        ]
    else:
        raise ValueError(f"Unsupported DSPS comparison mode: {mode}")

    sed_result = predict_batch_seds(context, parameter_names, parameter_matrix)
    if mode in {"map", "population"}:
        fit_rows = _batch_fit_rows(
            batch, fit_result, sed_result, config, mode, chunk_index
        )
        likelihood_rows = _batch_likelihood_rows(
            batch,
            fit_result,
            sed_result,
            observed_mag,
            observed_flux,
            sigma_mag,
            context,
            config,
            mode,
        )
    return {
        "sed_result": sed_result,
        "fit_rows": fit_rows,
        "hyper_rows": hyper_rows,
        "trace_rows": trace_rows,
        "likelihood_rows": likelihood_rows,
    }


def _batch_fit_rows(
    batch: pd.DataFrame,
    fit_result,
    sed_result: BatchSedResult,
    config: dict[str, Any],
    mode: str,
    chunk_index: int,
) -> list[dict[str, Any]]:
    rows = []
    for local_index, (row_index, _) in enumerate(batch.iterrows()):
        n_bands = len(config["bands"])
        params = {
            name: float(sed_result.parameter_matrix[local_index, index])
            for index, name in enumerate(sed_result.parameter_names)
            if not name.endswith("_prior_sigma")
        }
        derived = {
            f"fit_{name}": float(values[local_index])
            for name, values in sed_result.derived.items()
        }
        rows.append(
            {
                "row_index": int(row_index),
                "chunk_index": int(chunk_index),
                "dsps_mode": mode,
                "success": bool(fit_result.success[local_index]),
                "message": fit_result.message,
                "chi2": float(fit_result.chi2[local_index]),
                "reduced_chi2": float(fit_result.chi2[local_index]) / max(n_bands, 1),
                "gradient_norm": float(fit_result.gradient_norm[local_index]),
                "n_bands": n_bands,
                "device": fit_result.device,
                **{f"fit_{key}": value for key, value in params.items()},
                **derived,
            }
        )
    return rows


def _batch_likelihood_rows(
    batch: pd.DataFrame,
    fit_result,
    sed_result: BatchSedResult,
    observed_mag,
    observed_flux,
    sigma_mag,
    context,
    config: dict[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    if observed_mag is None or observed_flux is None or sigma_mag is None:
        return rows
    band_names = [band["name"] for band in config["bands"]]
    filter_curves = [context.filters[name] for name in band_names]
    for local_index, (row_index, row) in enumerate(batch.iterrows()):
        params = {
            name: float(fit_result.best_parameter_matrix[local_index, param_index])
            for param_index, name in enumerate(fit_result.parameter_names)
            if not name.endswith("_prior_sigma")
        }
        context_values = _row_context(row.to_dict(), params, config)
        fit_values = {f"fit_{key}": value for key, value in params.items()}
        fit_values.update(
            {
                f"fit_{name}": float(values[local_index])
                for name, values in sed_result.derived.items()
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
            rows.append(
                {
                    "row_index": int(row_index),
                    "dsps_mode": mode,
                    "band": band["name"],
                    "column": band["column"],
                    "error_column": band.get("error_column", ""),
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
                    **fit_values,
                    **context_values,
                }
            )
    return rows


def _model_result_from_batch(
    batch_result: BatchSedResult,
    local_index: int,
    row: dict[str, Any],
    row_index: int,
    context,
    config: dict[str, Any],
) -> ModelResult:
    params = {
        name: float(batch_result.parameter_matrix[local_index, index])
        for index, name in enumerate(batch_result.parameter_names)
        if not name.endswith("_prior_sigma")
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


def _read_selected_rows(
    config: dict[str, Any],
    cosmos_columns: list[str],
    limit: int | None,
    batch_size: int,
    index: int | None,
) -> pd.DataFrame:
    columns = set(cosmos_columns)
    columns.update(required_catalog_columns(config))
    available = _available_catalog_columns(config["catalog_path"])
    columns.update(_present_branch2_columns(config, available))
    columns = sorted(column for column in columns if column in available)
    if index is not None:
        frames = list(
            iter_catalog_batches(
                config["catalog_path"],
                columns=columns,
                batch_size=batch_size,
                row_indices={int(index)},
            )
        )
        if not frames:
            raise ValueError(f"Catalog row index not found: {index}")
        return pd.concat(frames)
    if limit is None:
        frames = list(
            iter_catalog_batches(
                config["catalog_path"],
                columns=columns,
                batch_size=batch_size,
                limit=None,
            )
        )
        return pd.concat(frames) if frames else pd.DataFrame(columns=columns)
    return read_catalog(config["catalog_path"], columns=columns, nrows=limit)


def _available_catalog_columns(path: str | Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema.names)


def _present_branch2_columns(
    config: dict[str, Any], available_columns: set[str]
) -> set[str]:
    columns = set()
    target_set_names = config.get("cosmos_sed", {}).get(
        "observed_photometry_target_sets"
    )
    for target_set in photometry_target_sets(config["bands"], target_set_names):
        for item in target_set["bands"]:
            for key in ("target_column", "error_column"):
                column = item.get(key)
                if column and column in available_columns:
                    columns.add(column)
    return columns


def _missing_branch2_columns(
    config: dict[str, Any], available_columns: set[str]
) -> list[str]:
    missing = set()
    target_set_names = config.get("cosmos_sed", {}).get(
        "observed_photometry_target_sets"
    )
    for target_set in photometry_target_sets(config["bands"], target_set_names):
        for item in target_set["bands"]:
            for key in ("target_column", "error_column"):
                column = item.get(key)
                if column and column not in available_columns:
                    missing.add(column)
    return sorted(missing)


def _value_added_data_report(cosmos_config: dict[str, Any]) -> dict[str, Any]:
    value_added_dir = resolve_value_added_data_dir(cosmos_config)
    if value_added_dir is None:
        return {
            "configured": False,
            "role": (
                "Not configured. COSMOS templates/extinction curves are loaded from "
                "LePhare paths instead."
            ),
        }
    sed_dir = value_added_dir / "galaxy_seds"
    extinct_dir = value_added_dir / "galaxy_extincts"
    filter_dir = value_added_dir / "filters"
    sed_files = sorted(sed_dir.glob("*.csv"))
    return {
        "configured": True,
        "path": str(value_added_dir),
        "galaxy_seds_dir": str(sed_dir),
        "galaxy_seds_count": len(sed_files),
        "galaxy_seds_first": sed_files[0].name if sed_files else None,
        "galaxy_seds_last": sed_files[-1].name if sed_files else None,
        "galaxy_extincts_dir": str(extinct_dir),
        "galaxy_extincts_count": len(list(extinct_dir.glob("*.csv"))),
        "filters_dir": str(filter_dir),
        "filters_count": len(list(filter_dir.glob("*.csv"))),
        "role": (
            "Primary local SciPIC value-added library. It provides the COSMOS "
            "template family used by sed_cosmos_* and flat-spectrum attenuation "
            "files used to derive k(lambda). It is a better local resource "
            "source than an external LePhare cache, but still template-level "
            "pseudo truth rather than physical spectra."
        ),
    }


def _print_abs_flux_table(result) -> None:
    print("Synthetic vs catalog absolute Euclid fluxes:")
    for band in result.catalog_abs_fluxes:
        print(
            f"  {band}: synthetic={result.synthetic_abs_fluxes_after[band]:.6e} "
            f"catalog={result.catalog_abs_fluxes[band]:.6e} "
            f"rel_resid={result.relative_residuals_vs_catalog_abs[band]:+.3e}"
        )
