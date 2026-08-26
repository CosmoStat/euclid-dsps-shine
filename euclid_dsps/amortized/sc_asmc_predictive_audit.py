"""Full-catalogue post-freeze photometric audits for SC-ASMC-EM."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.config import load_config
from euclid_dsps.parameter_vectors import model_mags_from_theta_matrix_jax
from euclid_dsps.photometry import abmag_to_fnu_cgs_jax

from .data import load_photometry_arrays_from_config
from .mira import FENIKS_SPLINE15D_PARAMETERS
from .posterior_bank import C0_SCOPE_STATEMENT, sha256_file
from .sc_asmc_postfreeze import (
    bind_frozen_training_config,
    validate_postfreeze_gate,
)
from .sc_asmc_training import prepare_sc_runtime

POSTERIOR_METHODS = ("q0", "smc_em1", "q1", "smc_em2")


def run_full_catalogue_predictive_audit(
    *,
    training_config_path: str | Path,
    truth_config_path: str | Path,
    run_root: str | Path,
    closure_root: str | Path,
    out_dir: str | Path,
    posterior_draws: int = 16,
    decoder_pairs_per_batch: int = 128,
) -> dict[str, Any]:
    """Audit truth-forward and all posterior photometry after model freeze."""
    if int(posterior_draws) <= 0:
        raise ValueError("posterior_draws must be positive")
    root = Path(run_root)
    closure = Path(closure_root)
    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit: {output}")
    output.mkdir(parents=True, exist_ok=True)

    validate_postfreeze_gate(root)
    closure_receipt = _read_json(closure / "truth_closure_receipt.json")
    if closure_receipt.get("status") != "PASS":
        raise ValueError("predictive audit requires a PASS truth closure")
    if closure_receipt.get("final_receipt_sha256") != sha256_file(
        root / "FINAL_RECEIPT.json"
    ):
        raise ValueError("truth closure is not bound to the frozen final receipt")

    manifest = _read_json(root / "manifest" / "run_manifest.json")
    training_config = bind_frozen_training_config(
        load_config(Path(training_config_path).resolve()), manifest
    )
    truth_config = load_config(Path(truth_config_path).resolve())
    truth_config["catalog_path"] = str(Path(manifest["dataset"]["path"]).resolve())
    parameters = tuple(FENIKS_SPLINE15D_PARAMETERS)
    mappings = (truth_config.get("truth", {}) or {}).get("parameter_columns") or {}
    if set(mappings) != set(parameters):
        raise ValueError("truth config must map the exact spline15d closure parameters")

    arrays = load_photometry_arrays_from_config(
        truth_config,
        batch_size=int(
            (truth_config.get("amortized", {}) or {})
            .get("data", {})
            .get("catalog_batch_size", 10_000)
        ),
    )
    if arrays.truth is None or arrays.row_index is None:
        raise ValueError("predictive audit requires truth and stable row indices")
    catalogue_rows = np.asarray(arrays.row_index, dtype=np.int64)
    row_lookup = {int(row): index for index, row in enumerate(catalogue_rows)}
    if len(row_lookup) != len(catalogue_rows):
        raise ValueError("catalogue row indices are not unique")

    runtime = prepare_sc_runtime(
        training_config,
        output / "runtime",
        feature_train_rows=manifest["artifacts"]["feature_train_rows"]["path"],
        heldout_rows=manifest["artifacts"]["heldout_rows"]["path"],
    )
    if tuple(runtime.parameter_names) != parameters:
        raise ValueError("runtime parameter order differs from closure order")
    decode = _make_parallel_theta_decoder(
        runtime,
        pairs_per_batch=int(decoder_pairs_per_batch),
    )

    selected_rows = np.load(
        manifest["artifacts"]["selected_rows"]["path"], allow_pickle=False
    ).astype(np.int64)
    selected_index = np.asarray([row_lookup[int(row)] for row in selected_rows])
    truth_theta = np.column_stack(
        [np.asarray(arrays.truth[name])[selected_index] for name in parameters]
    )
    truth_flux = decode(truth_theta)
    truth_residual = (
        truth_flux - np.asarray(arrays.flux)[selected_index]
    ) / np.asarray(arrays.flux_err)[selected_index]
    truth_mask = np.asarray(arrays.mask)[selected_index]

    summary_rows = residual_summary_rows(
        "truth_forward",
        truth_residual[None, ...],
        truth_mask,
        tuple(arrays.band_names),
    )
    object_frames = [
        object_predictive_rows(
            "truth_forward",
            selected_rows,
            np.asarray(arrays.object_id)[selected_index].astype(str),
            truth_residual[None, ...],
            truth_mask,
        )
    ]
    print(
        "[sc-asmc][predictive-audit] "
        f"truth_forward objects={len(selected_rows)} complete",
        flush=True,
    )

    method_objects: dict[str, int] = {}
    for method in POSTERIOR_METHODS:
        residual_chunks = []
        mask_chunks = []
        object_chunks = []
        method_dir = closure / "posterior_samples_all_methods" / method
        paths = sorted(method_dir.glob("posterior_samples_*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no posterior closure shards for {method}")
        for shard_index, path in enumerate(paths):
            frame = pd.read_parquet(path)
            rows, object_ids, theta = _posterior_theta_block(
                frame,
                parameters=parameters,
                requested_draws=int(posterior_draws),
            )
            indices = np.asarray([row_lookup[int(row)] for row in rows])
            model_flux = decode(theta.reshape(-1, len(parameters))).reshape(
                theta.shape[0], theta.shape[1], len(arrays.band_names)
            )
            observed_flux = np.asarray(arrays.flux)[indices]
            observed_error = np.asarray(arrays.flux_err)[indices]
            mask = np.asarray(arrays.mask)[indices]
            residual = (model_flux - observed_flux[None, ...]) / observed_error[
                None, ...
            ]
            residual_chunks.append(residual)
            mask_chunks.append(mask)
            object_chunks.append(
                object_predictive_rows(method, rows, object_ids, residual, mask)
            )
            print(
                "[sc-asmc][predictive-audit] "
                f"method={method} shard={shard_index + 1}/{len(paths)} "
                f"objects={len(rows)}",
                flush=True,
            )
        residual = np.concatenate(residual_chunks, axis=1)
        mask = np.concatenate(mask_chunks, axis=0)
        method_objects[method] = int(mask.shape[0])
        summary_rows.extend(
            residual_summary_rows(method, residual, mask, tuple(arrays.band_names))
        )
        object_frames.append(pd.concat(object_chunks, ignore_index=True))

    summary = pd.DataFrame(summary_rows)
    summary_path = output / "predictive_residuals_by_band.csv"
    _atomic_csv(summary_path, summary)
    objects = pd.concat(object_frames, ignore_index=True)
    object_path = output / "predictive_diagnostics_by_object.parquet"
    _atomic_parquet(object_path, objects)
    return finalize_existing_predictive_audit(
        out_dir=output,
        closure_root=closure,
        final_receipt_sha256=sha256_file(root / "FINAL_RECEIPT.json"),
        recovered_after_postprocessing_failure=False,
    )


def finalize_existing_predictive_audit(
    *,
    out_dir: str | Path,
    closure_root: str | Path,
    final_receipt_sha256: str | None = None,
    recovered_after_postprocessing_failure: bool = True,
) -> dict[str, Any]:
    """Finalize already-decoded predictive artifacts without rerunning DSPS."""
    output = Path(out_dir)
    closure = Path(closure_root)
    summary_path = output / "predictive_residuals_by_band.csv"
    object_path = output / "predictive_diagnostics_by_object.parquet"
    if not summary_path.is_file() or not object_path.is_file():
        raise FileNotFoundError("predictive summary and object diagnostics are required")
    closure_path = closure / "truth_closure_receipt.json"
    closure_receipt = _read_json(closure_path)
    if closure_receipt.get("status") != "PASS":
        raise ValueError("predictive finalization requires a PASS truth closure")
    expected_final_hash = str(closure_receipt.get("final_receipt_sha256", ""))
    if not expected_final_hash:
        raise ValueError("truth closure receipt lacks final-receipt provenance")
    if final_receipt_sha256 is not None and str(final_receipt_sha256) != expected_final_hash:
        raise ValueError("predictive finalization final-receipt hash mismatch")

    summary = pd.read_csv(summary_path)
    objects = pd.read_parquet(object_path)
    _validate_finalization_tables(summary, objects)
    method_summary = predictive_method_summary(summary, objects)
    method_summary_path = output / "predictive_method_summary.csv"
    _atomic_csv(method_summary_path, method_summary)
    plot_paths = write_predictive_audit_plots(summary, objects, output)

    forward_checks = _truth_forward_checks(summary)
    posterior_checks = _posterior_predictive_checks(summary, objects)
    method_objects = {
        method: int(group["row_index"].nunique())
        for method, group in objects.groupby("method", sort=False)
        if method != "truth_forward"
    }
    truth_objects = int(
        objects.loc[objects["method"] == "truth_forward", "row_index"].nunique()
    )
    draws = {
        str(row.method): int(row.draws_per_object)
        for row in summary.drop_duplicates("method").itertuples()
    }
    posterior_draws = {draws[method] for method in POSTERIOR_METHODS}
    if len(posterior_draws) != 1:
        raise ValueError("posterior methods do not share one dense-draw count")
    plot_records = {name: _file_record(path) for name, path in plot_paths.items()}
    receipt = {
        "status": "PASS",
        "phase": "postfreeze_full_catalogue_predictive_audit",
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "training_frozen_before_truth": True,
        "truth_used_for_training_or_checkpoint_selection": False,
        "recovered_after_postprocessing_failure": bool(
            recovered_after_postprocessing_failure
        ),
        "final_receipt_sha256": expected_final_hash,
        "truth_closure_receipt_sha256": sha256_file(closure_path),
        "selected_truth_forward_objects": truth_objects,
        "resolved_posterior_objects": method_objects,
        "posterior_draws_per_object": int(next(iter(posterior_draws))),
        "forward_model_checks": forward_checks,
        "forward_model_gate_pass": bool(all(forward_checks.values())),
        "posterior_predictive_checks": posterior_checks,
        "posterior_predictive_gate_pass": bool(all(posterior_checks.values())),
        "scientific_gate_pass": bool(
            all(forward_checks.values()) and all(posterior_checks.values())
        ),
        "artifacts": {
            "summary": _file_record(summary_path),
            "method_summary": _file_record(method_summary_path),
            "objects": _file_record(object_path),
            "plot": plot_records["overview_png"],
            "plots": plot_records,
        },
    }
    _atomic_json(output / "predictive_audit_receipt.json", receipt)
    return receipt


def residual_summary_rows(
    method: str,
    residual: np.ndarray,
    mask: np.ndarray,
    band_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Summarize likelihood-normalized residual draws for every band."""
    values = np.asarray(residual, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != valid.shape:
        raise ValueError("residual must be [draw,object,band] and match mask")
    rows = []
    for band_index, band in enumerate(band_names):
        selected = values[:, valid[:, band_index], band_index].reshape(-1)
        finite = np.isfinite(selected)
        clean = selected[finite]
        if not len(clean):
            raise ValueError(f"no finite residual for {method}/{band}")
        rows.append(
            {
                "method": str(method),
                "band": str(band),
                "objects": int(np.sum(valid[:, band_index])),
                "draws_per_object": int(values.shape[0]),
                "median_normalized_residual": float(np.median(clean)),
                "mean_normalized_residual": float(np.mean(clean)),
                "rms_normalized_residual": float(np.sqrt(np.mean(clean**2))),
                "q05_normalized_residual": float(np.quantile(clean, 0.05)),
                "q95_normalized_residual": float(np.quantile(clean, 0.95)),
                "fraction_abs_lt_1": float(np.mean(np.abs(clean) < 1.0)),
                "fraction_abs_lt_3": float(np.mean(np.abs(clean) < 3.0)),
                "finite_fraction": float(np.mean(finite)),
            }
        )
    return rows


def object_predictive_rows(
    method: str,
    row_indices: np.ndarray,
    object_ids: np.ndarray,
    residual: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    """Return one full-draw predictive diagnostic row per catalogue object."""
    values = np.asarray(residual, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    squared = np.where(valid[None, ...], values**2, np.nan)
    reduced_chi2 = np.nanmean(squared, axis=2)
    return pd.DataFrame(
        {
            "method": str(method),
            "row_index": np.asarray(row_indices, dtype=np.int64),
            "object_id": np.asarray(object_ids).astype(str),
            "posterior_median_reduced_chi2": np.nanmedian(reduced_chi2, axis=0),
            "posterior_mean_reduced_chi2": np.nanmean(reduced_chi2, axis=0),
            "valid_bands": np.sum(valid, axis=1),
        }
    )


def _posterior_theta_block(
    frame: pd.DataFrame,
    *,
    parameters: tuple[str, ...],
    requested_draws: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"row_index", "object_id", "sample_id", *parameters}
    if not required.issubset(frame.columns):
        raise ValueError("posterior shard lacks predictive-audit columns")
    frame = frame.sort_values(["row_index", "sample_id"], kind="stable")
    counts = frame.groupby("row_index", sort=False)["sample_id"].size().to_numpy()
    if not len(counts) or np.any(counts != counts[0]):
        raise ValueError("posterior shard has unequal draw counts")
    available = int(counts[0])
    if requested_draws > available:
        raise ValueError("requested posterior draws exceed stored closure draws")
    objects = frame.drop_duplicates("row_index", keep="first")
    theta = (
        frame.loc[:, parameters]
        .to_numpy(dtype=np.float32)
        .reshape(len(objects), available, len(parameters))
    )
    draw_index = np.linspace(0, available - 1, requested_draws, dtype=int)
    return (
        objects["row_index"].to_numpy(dtype=np.int64),
        objects["object_id"].astype(str).to_numpy(),
        theta[:, draw_index, :].transpose(1, 0, 2),
    )


def _make_parallel_theta_decoder(runtime, *, pairs_per_batch: int):
    devices = tuple(jax.local_devices())
    if not devices:
        raise RuntimeError("predictive audit requires a JAX device")
    if pairs_per_batch < len(devices):
        raise ValueError("decoder batch must be at least the local device count")
    per_device = max(1, int(pairs_per_batch) // len(devices))
    fixed_pairs = per_device * len(devices)

    def decode_device(theta):
        magnitudes = model_mags_from_theta_matrix_jax(
            runtime.context,
            runtime.model_args,
            theta,
            tuple(runtime.parameter_names),
        )
        return abmag_to_fnu_cgs_jax(magnitudes)

    parallel_decode = jax.pmap(decode_device, devices=devices)

    def decode(theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(runtime.parameter_names):
            raise ValueError("theta decoder expects [pairs,parameters]")
        chunks = []
        for start in range(0, len(values), fixed_pairs):
            block = values[start : start + fixed_pairs]
            count = len(block)
            if count < fixed_pairs:
                block = np.pad(block, ((0, fixed_pairs - count), (0, 0)), mode="edge")
            sharded = block.reshape(len(devices), per_device, block.shape[1])
            decoded = np.asarray(jax.device_get(parallel_decode(jnp.asarray(sharded))))
            chunks.append(decoded.reshape(fixed_pairs, -1)[:count])
        return np.concatenate(chunks, axis=0)

    return decode


def _validate_finalization_tables(
    summary: pd.DataFrame, objects: pd.DataFrame
) -> None:
    summary_columns = {
        "method",
        "band",
        "objects",
        "draws_per_object",
        "median_normalized_residual",
        "mean_normalized_residual",
        "rms_normalized_residual",
        "fraction_abs_lt_1",
        "fraction_abs_lt_3",
        "finite_fraction",
    }
    object_columns = {
        "method",
        "row_index",
        "posterior_median_reduced_chi2",
        "posterior_mean_reduced_chi2",
    }
    if not summary_columns.issubset(summary.columns):
        raise ValueError("predictive summary lacks finalization columns")
    if not object_columns.issubset(objects.columns):
        raise ValueError("predictive object table lacks finalization columns")
    expected = {"truth_forward", *POSTERIOR_METHODS}
    if set(summary["method"]) != expected or set(objects["method"]) != expected:
        raise ValueError("predictive artifacts do not contain every required method")
    band_counts = summary.groupby("method")["band"].nunique()
    if band_counts.nunique() != 1 or int(band_counts.iloc[0]) <= 0:
        raise ValueError("predictive methods do not share one complete band set")


def predictive_method_summary(
    summary: pd.DataFrame, objects: pd.DataFrame
) -> pd.DataFrame:
    """Return compact method-level diagnostics without discarding source draws."""
    rows = []
    for method in ("truth_forward", *POSTERIOR_METHODS):
        band = summary[summary["method"] == method]
        per_object = objects[objects["method"] == method]
        rows.append(
            {
                "method": method,
                "objects": int(per_object["row_index"].nunique()),
                "draws_per_object": int(band["draws_per_object"].iloc[0]),
                "median_abs_band_bias": float(
                    np.median(np.abs(band["median_normalized_residual"]))
                ),
                "max_abs_band_bias": float(
                    np.max(np.abs(band["median_normalized_residual"]))
                ),
                "median_band_rms": float(np.median(band["rms_normalized_residual"])),
                "max_band_rms": float(np.max(band["rms_normalized_residual"])),
                "median_fraction_abs_lt_1": float(
                    np.median(band["fraction_abs_lt_1"])
                ),
                "median_fraction_abs_lt_3": float(
                    np.median(band["fraction_abs_lt_3"])
                ),
                "median_object_posterior_median_reduced_chi2": float(
                    np.median(per_object["posterior_median_reduced_chi2"])
                ),
                "median_object_posterior_mean_reduced_chi2": float(
                    np.median(per_object["posterior_mean_reduced_chi2"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _truth_forward_checks(summary: pd.DataFrame) -> dict[str, bool]:
    oracle = summary[summary["method"] == "truth_forward"]
    return {
        "all_finite": bool(np.all(oracle["finite_fraction"] == 1.0)),
        "max_abs_median_at_most_0p2": bool(
            np.max(np.abs(oracle["median_normalized_residual"])) <= 0.2
        ),
        "rms_between_0p8_and_1p2": bool(
            np.all(
                (oracle["rms_normalized_residual"] >= 0.8)
                & (oracle["rms_normalized_residual"] <= 1.2)
            )
        ),
    }


def _posterior_predictive_checks(
    summary: pd.DataFrame, objects: pd.DataFrame
) -> dict[str, bool]:
    checks = {}
    for method in ("q1", "smc_em2"):
        band = summary[summary["method"] == method]
        per_object = objects[objects["method"] == method]
        checks[f"{method}_all_finite"] = bool(
            np.all(band["finite_fraction"] == 1.0)
        )
        checks[f"{method}_max_band_rms_at_most_2"] = bool(
            np.max(band["rms_normalized_residual"]) <= 2.0
        )
        checks[f"{method}_median_object_reduced_chi2_at_most_2"] = bool(
            np.median(per_object["posterior_median_reduced_chi2"]) <= 2.0
        )
    return checks


def write_predictive_audit_plots(
    summary: pd.DataFrame, objects: pd.DataFrame, output: Path
) -> dict[str, Path]:
    """Write overview, oracle, SMC zoom, and per-object predictive plots."""
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths.update(
        _write_method_band_plot(
            summary,
            output,
            methods=("truth_forward", *POSTERIOR_METHODS),
            stem="full_catalogue_predictive_residuals",
            title="Full-catalogue photometric residual audit",
            broad_scale=True,
        )
    )
    paths.update(_write_truth_forward_plot(summary, output))
    paths.update(
        _write_method_band_plot(
            summary,
            output,
            methods=("truth_forward", "smc_em1", "smc_em2"),
            stem="posterior_predictive_smc_zoom",
            title="Posterior predictive residuals: truth and SMC",
            broad_scale=False,
        )
    )
    paths.update(_write_reduced_chi2_ecdf(objects, output))
    return paths


def _write_method_band_plot(
    summary: pd.DataFrame,
    output: Path,
    *,
    methods: tuple[str, ...],
    stem: str,
    title: str,
    broad_scale: bool,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("median_normalized_residual", "Median normalized residual"),
        ("rms_normalized_residual", "RMS normalized residual"),
        ("fraction_abs_lt_1", "Fraction |residual| < 1"),
    )
    colors = {
        "truth_forward": "#171717",
        "q0": "#3B6FB6",
        "smc_em1": "#D47A24",
        "q1": "#7A5195",
        "smc_em2": "#28846B",
    }
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    bands = list(summary["band"].drop_duplicates())
    x = np.arange(len(bands))
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        for method in methods:
            rows = summary[summary["method"] == method].set_index("band").loc[bands]
            axis.plot(
                x,
                rows[metric],
                marker="o",
                ms=3,
                lw=1.5,
                label=method,
                color=colors[method],
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    if broad_scale:
        axes[0].set_yscale("symlog", linthresh=0.1)
        axes[1].set_yscale("log")
    axes[0].axhline(0.0, color="0.6", lw=1)
    axes[1].axhline(1.0, color="0.6", lw=1)
    axes[2].axhline(0.6827, color="0.6", lw=1)
    axes[0].legend(ncol=len(methods), frameon=False, loc="upper center")
    axes[-1].set_xticks(x, bands, rotation=45, ha="right")
    figure.suptitle(title)
    figure.tight_layout()
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    figure.savefig(png, dpi=200)
    figure.savefig(pdf)
    plt.close(figure)
    prefix = "overview" if stem == "full_catalogue_predictive_residuals" else "smc_zoom"
    return {f"{prefix}_png": png, f"{prefix}_pdf": pdf}


def _write_truth_forward_plot(
    summary: pd.DataFrame, output: Path
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    oracle = summary[summary["method"] == "truth_forward"].copy()
    bands = list(oracle["band"])
    x = np.arange(len(bands))
    figure, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    axes[0].plot(x, oracle["mean_normalized_residual"], "o-", label="mean")
    axes[0].plot(x, oracle["median_normalized_residual"], "s-", label="median")
    axes[0].axhline(0.0, color="0.5", lw=1)
    axes[0].set_ylabel("Normalized residual")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(x, oracle["rms_normalized_residual"], "o-", color="#28846B")
    axes[1].axhline(1.0, color="0.5", lw=1)
    axes[1].set_ylabel("RMS")
    axes[2].plot(x, oracle["fraction_abs_lt_1"], "o-", label="|r| < 1")
    axes[2].plot(x, oracle["fraction_abs_lt_3"], "s-", label="|r| < 3")
    axes[2].axhline(0.6827, color="0.5", lw=1, ls="--")
    axes[2].axhline(0.9973, color="0.5", lw=1, ls=":")
    axes[2].set_ylabel("Empirical fraction")
    axes[2].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[-1].set_xticks(x, bands, rotation=45, ha="right")
    figure.suptitle("DSPS forward model evaluated at FENIKS truth")
    figure.tight_layout()
    png = output / "truth_forward_residuals_by_band.png"
    pdf = output / "truth_forward_residuals_by_band.pdf"
    figure.savefig(png, dpi=200)
    figure.savefig(pdf)
    plt.close(figure)
    return {"truth_forward_png": png, "truth_forward_pdf": pdf}


def _write_reduced_chi2_ecdf(
    objects: pd.DataFrame, output: Path
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "truth_forward": "#171717",
        "q0": "#3B6FB6",
        "smc_em1": "#D47A24",
        "q1": "#7A5195",
        "smc_em2": "#28846B",
    }
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for method in ("truth_forward", *POSTERIOR_METHODS):
        values = np.asarray(
            objects.loc[
                objects["method"] == method,
                "posterior_median_reduced_chi2",
            ],
            dtype=float,
        )
        values = np.sort(values[np.isfinite(values) & (values > 0.0)])
        probability = np.arange(1, len(values) + 1, dtype=float) / len(values)
        axis.plot(values, probability, lw=1.7, label=method, color=colors[method])
    axis.axvline(1.0, color="0.5", lw=1)
    axis.set_xscale("log")
    axis.set_xlabel("Per-object median predictive reduced chi-square")
    axis.set_ylabel("Empirical CDF")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    axis.set_title("Catalogue posterior-predictive fit quality")
    figure.tight_layout()
    png = output / "posterior_predictive_reduced_chi2_ecdf.png"
    pdf = output / "posterior_predictive_reduced_chi2_ecdf.pdf"
    figure.savefig(png, dpi=200)
    figure.savefig(pdf)
    plt.close(figure)
    return {"reduced_chi2_png": png, "reduced_chi2_pdf": pdf}


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
