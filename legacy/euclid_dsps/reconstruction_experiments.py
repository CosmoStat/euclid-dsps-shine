"""Reproducible Diffsky reconstruction experiment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir, load_row_indices, write_json


def build_reconstruction_rowsets(
    *,
    train_run: str | Path,
    infer_run: str | Path,
    out_dir: str | Path,
    worst_sizes: tuple[int, ...] = (500, 1000),
    metric: str = "median_abs_sigma",
    balanced_size: int = 20_000,
    balanced_seed: int = 42,
    redshift_bins: tuple[float, ...] | list[float] | None = None,
) -> dict[str, str]:
    """Write reference and worst-case row-index files from an inference run."""
    train_run = Path(train_run)
    infer_run = Path(infer_run)
    out = ensure_dir(out_dir)
    train_indices = _load_npy_int(train_run / "train_indices.npy")
    validation_indices = _load_npy_int(train_run / "validation_indices.npy")
    inference_indices = _load_npy_int(infer_run / "inference_indices.npy")
    reference_indices = np.sort(
        np.unique(np.concatenate([train_indices, validation_indices]))
    )
    inference_unique = np.sort(np.unique(inference_indices))
    if not np.array_equal(reference_indices, inference_unique):
        raise ValueError(
            "Reference train+validation indices do not match inference indices: "
            f"train+val={reference_indices.size} inference={inference_unique.size}"
        )

    _write_indices(out / "reference_train.txt", np.sort(train_indices))
    _write_indices(out / "reference_validation.txt", np.sort(validation_indices))
    _write_indices(out / "reference_20k.txt", reference_indices)

    residual = _read_first_table(
        infer_run,
        ("posterior_predictive_residual_summary",),
    )
    ranking = _object_residual_ranking(residual)
    if metric not in ranking:
        raise ValueError(
            f"metric {metric!r} is not available; choose one of "
            f"{sorted(ranking.columns)}"
        )
    ranked = ranking.sort_values(metric, ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=np.int64))
    ranked.to_csv(out / "worst_ranked.csv", index=False)
    ranked.to_parquet(out / "worst_ranked.parquet", index=False)

    rowsets: dict[str, str] = {
        "reference_train": "reference_train.txt",
        "reference_validation": "reference_validation.txt",
        "reference_20k": "reference_20k.txt",
    }
    balanced_summary: dict[str, Any] | None = None
    if int(balanced_size) > 0:
        truth = _read_optional_table(infer_run, ("inference_truth",))
        balanced, balanced_diagnostics, balanced_summary = _balanced_rowset(
            residual,
            truth,
            reference_indices=reference_indices,
            size=int(balanced_size),
            seed=int(balanced_seed),
            redshift_bins=redshift_bins,
        )
        balanced_name = "balanced20k" if int(balanced_size) == 20_000 else f"balanced_{int(balanced_size)}"
        _write_indices(out / f"{balanced_name}.txt", balanced)
        balanced_diagnostics.to_csv(
            out / f"{balanced_name}_diagnostics.csv",
            index=False,
        )
        balanced_diagnostics.to_parquet(
            out / f"{balanced_name}_diagnostics.parquet",
            index=False,
        )
        rowsets[balanced_name] = f"{balanced_name}.txt"
    for size in sorted({int(value) for value in worst_sizes if int(value) > 0}):
        selected = ranked.head(size)["row_index"].to_numpy(dtype=np.int64)
        name = f"worst_{size}"
        _write_indices(out / f"{name}.txt", selected)
        rowsets[name] = f"{name}.txt"

    manifest = {
        "train_run": str(train_run),
        "infer_run": str(infer_run),
        "metric": str(metric),
        "reference_rows": int(reference_indices.size),
        "train_rows": int(train_indices.size),
        "validation_rows": int(validation_indices.size),
        "balanced_rowset": balanced_summary,
        "rowsets": rowsets,
        "ranking": "worst_ranked.csv",
    }
    write_json(out / "rowsets_manifest.json", manifest)
    return {"manifest": str(out / "rowsets_manifest.json")}


def compare_reconstruction_runs(
    *,
    out_dir: str | Path,
    runs: list[tuple[str, Path]],
    rowset_path: str | Path | None = None,
) -> dict[str, str]:
    """Aggregate reconstruction residual summaries for multiple methods."""
    out = ensure_dir(out_dir)
    if not runs:
        raise ValueError("At least one run must be provided")
    rowset = set(load_row_indices(rowset_path)) if rowset_path is not None else None
    frames = []
    run_sources = []
    for label, path in runs:
        frame, source = _load_reconstruction_run(Path(path))
        frame = frame.copy()
        frame["method"] = str(label)
        frame["run_path"] = str(path)
        if rowset is not None and "row_index" in frame:
            frame = frame[frame["row_index"].isin(rowset)]
        frames.append(frame)
        run_sources.append({"method": str(label), "path": str(path), "source": source})
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        raise ValueError("No reconstruction residual rows were loaded")
    combined.to_parquet(out / "reconstruction_residual_summary.parquet", index=False)
    combined.to_csv(out / "reconstruction_residual_summary.csv", index=False)

    method_summary = _method_summary(combined)
    band_summary = _band_summary(combined)
    object_summary = _object_summary(combined)
    method_summary.to_csv(out / "reconstruction_method_summary.csv", index=False)
    band_summary.to_csv(out / "reconstruction_band_summary.csv", index=False)
    object_summary.to_csv(out / "reconstruction_object_summary.csv", index=False)
    method_summary.to_parquet(
        out / "reconstruction_method_summary.parquet",
        index=False,
    )
    band_summary.to_parquet(out / "reconstruction_band_summary.parquet", index=False)
    object_summary.to_parquet(
        out / "reconstruction_object_summary.parquet",
        index=False,
    )
    payload = {
        "runs": run_sources,
        "rowset_path": str(rowset_path) if rowset_path is not None else None,
        "rows": int(len(combined)),
        "methods": method_summary.to_dict(orient="records"),
    }
    write_json(out / "reconstruction_comparison_summary.json", payload)
    _write_report(out / "reconstruction_comparison.md", payload, method_summary)
    return {"report": str(out / "reconstruction_comparison.md")}


def _balanced_rowset(
    residual: pd.DataFrame,
    truth: pd.DataFrame | None,
    *,
    reference_indices: np.ndarray,
    size: int,
    seed: int,
    redshift_bins: tuple[float, ...] | list[float] | None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    diagnostics = _object_quality_diagnostics(residual, reference_indices)
    z_column = None
    if truth is not None and not truth.empty and "row_index" in truth:
        z_column = _first_existing_column(
            truth,
            ("redshift_true", "z_obs", "z_true_gal", "redshift", "z"),
        )
        keep = ["row_index"]
        if z_column:
            keep.append(z_column)
        diagnostics = diagnostics.merge(truth[keep], on="row_index", how="left")
    if z_column is not None:
        diagnostics["redshift_reference"] = pd.to_numeric(
            diagnostics[z_column],
            errors="coerce",
        )
    else:
        diagnostics["redshift_reference"] = np.nan
    diagnostics["redshift_bin"] = _redshift_bin_codes(
        diagnostics["redshift_reference"].to_numpy(dtype=float),
        redshift_bins,
    )
    diagnostics["quality_bin"] = _quantile_codes(
        np.log10(
            np.maximum(
                diagnostics["median_err_over_abs_flux"].to_numpy(dtype=float),
                1.0e-12,
            )
        ),
        n_bins=4,
    )
    diagnostics["balanced_group"] = (
        diagnostics["redshift_bin"].astype(str)
        + "_"
        + diagnostics["quality_bin"].astype(str)
    )
    selected = _balanced_sample_indices(
        diagnostics,
        group_column="balanced_group",
        size=min(int(size), int(len(diagnostics))),
        seed=int(seed),
    )
    diagnostics["selected_balanced"] = diagnostics["row_index"].isin(selected)
    selected_sorted = np.sort(np.asarray(selected, dtype=np.int64))
    summary = {
        "name": "balanced20k" if int(size) == 20_000 else f"balanced_{int(size)}",
        "requested_rows": int(size),
        "selected_rows": int(selected_sorted.size),
        "seed": int(seed),
        "redshift_column": z_column,
        "redshift_bins": list(redshift_bins) if redshift_bins is not None else None,
        "quality_proxy": "median obs_err_fnu_cgs / abs(obs_flux_fnu_cgs)",
        "quality_bins": 4,
        "diagnostics": (
            "balanced20k_diagnostics.csv"
            if int(size) == 20_000
            else f"balanced_{int(size)}_diagnostics.csv"
        ),
    }
    return selected_sorted, diagnostics, summary


def _object_quality_diagnostics(
    residual: pd.DataFrame,
    reference_indices: np.ndarray,
) -> pd.DataFrame:
    required = {"row_index", "obs_flux_fnu_cgs", "obs_err_fnu_cgs"}
    missing = required - set(residual.columns)
    if missing:
        raise ValueError(
            "balanced rowset requires residual summary columns: "
            f"{sorted(missing)}"
        )
    frame = residual[residual["row_index"].isin(reference_indices)].copy()
    frame["obs_flux_fnu_cgs"] = pd.to_numeric(
        frame["obs_flux_fnu_cgs"],
        errors="coerce",
    )
    frame["obs_err_fnu_cgs"] = pd.to_numeric(
        frame["obs_err_fnu_cgs"],
        errors="coerce",
    )
    abs_flux = np.maximum(np.abs(frame["obs_flux_fnu_cgs"].to_numpy(float)), 1.0e-300)
    err = frame["obs_err_fnu_cgs"].to_numpy(float)
    frame["err_over_abs_flux"] = err / abs_flux
    frame["snr_proxy"] = abs_flux / np.maximum(err, 1.0e-300)
    grouped = frame.groupby("row_index", sort=False)
    rows = grouped.agg(
        median_err_over_abs_flux=("err_over_abs_flux", "median"),
        max_err_over_abs_flux=("err_over_abs_flux", "max"),
        median_snr_proxy=("snr_proxy", "median"),
        min_snr_proxy=("snr_proxy", "min"),
        n_valid_bands=("err_over_abs_flux", "count"),
    ).reset_index()
    rows["row_index"] = rows["row_index"].astype(np.int64)
    return rows


def _balanced_sample_indices(
    frame: pd.DataFrame,
    *,
    group_column: str,
    size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    remaining = frame.copy()
    while len(selected) < int(size) and not remaining.empty:
        groups = list(remaining.groupby(group_column, sort=True))
        if not groups:
            break
        quota = max(1, int(np.ceil((int(size) - len(selected)) / len(groups))))
        used_positions = []
        for _group_name, group in groups:
            take = min(quota, len(group), int(size) - len(selected))
            if take <= 0:
                continue
            choices = rng.choice(group.index.to_numpy(), size=take, replace=False)
            used_positions.extend(int(value) for value in choices)
            selected.extend(
                int(value) for value in group.loc[choices, "row_index"].to_numpy()
            )
            if len(selected) >= int(size):
                break
        if not used_positions:
            break
        remaining = remaining.drop(index=used_positions)
    if len(selected) < int(size):
        unselected = frame[~frame["row_index"].isin(selected)]
        take = min(int(size) - len(selected), len(unselected))
        if take > 0:
            choices = rng.choice(unselected.index.to_numpy(), size=take, replace=False)
            selected.extend(
                int(value) for value in unselected.loc[choices, "row_index"].to_numpy()
            )
    return np.asarray(selected[: int(size)], dtype=np.int64)


def _redshift_bin_codes(
    redshift: np.ndarray,
    redshift_bins: tuple[float, ...] | list[float] | None,
) -> np.ndarray:
    if redshift_bins is None:
        return np.zeros(redshift.shape, dtype=np.int64)
    bins = np.asarray(redshift_bins, dtype=float)
    if bins.ndim != 1 or bins.size < 2 or np.any(np.diff(bins) <= 0.0):
        return np.zeros(redshift.shape, dtype=np.int64)
    codes = np.digitize(redshift, bins, right=False) - 1
    valid = np.isfinite(redshift) & (codes >= 0) & (codes < bins.size - 1)
    return np.where(valid, codes, -1).astype(np.int64)


def _quantile_codes(values: np.ndarray, *, n_bins: int) -> np.ndarray:
    finite = np.isfinite(values)
    codes = np.full(values.shape, -1, dtype=np.int64)
    unique = np.unique(values[finite])
    if unique.size <= 1:
        codes[finite] = 0
        return codes
    q = min(int(n_bins), int(unique.size))
    ranked = pd.qcut(values[finite], q=q, labels=False, duplicates="drop")
    codes[finite] = np.asarray(ranked, dtype=np.int64)
    return codes


def _first_existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame:
            return name
    return None


def _load_reconstruction_run(run: Path) -> tuple[pd.DataFrame, str]:
    try:
        frame = _read_first_table(run, ("posterior_predictive_residual_summary",))
        return (
            _normalize_residual_summary(frame),
            "posterior_predictive_residual_summary",
        )
    except FileNotFoundError:
        pass
    try:
        frame = _read_first_table(
            run,
            ("batch_posterior_predictive_flux_residual_summary",),
        )
        return (
            _normalize_residual_summary(frame),
            "batch_posterior_predictive_flux_residual_summary",
        )
    except FileNotFoundError:
        pass
    frame = _read_first_table(run, ("batch_fit_photometry_comparison",))
    return _normalize_map_comparison(frame), "batch_fit_photometry_comparison"


def _normalize_residual_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"band", "residual_sigma_median"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"residual summary missing required columns: {sorted(missing)}")
    out = frame.copy()
    if "valid" not in out:
        out["valid"] = np.isfinite(
            pd.to_numeric(out["residual_sigma_median"], errors="coerce")
        )
    if "abs_residual_sigma_median" not in out:
        out["abs_residual_sigma_median"] = pd.to_numeric(
            out["residual_sigma_median"],
            errors="coerce",
        ).abs()
    return out


def _normalize_map_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    obs_flux = pd.to_numeric(frame.get("observed_flux_fnu_cgs"), errors="coerce")
    model_flux = pd.to_numeric(frame.get("model_flux_fnu_cgs"), errors="coerce")
    obs_err = pd.to_numeric(frame.get("observed_flux_error_fnu_cgs"), errors="coerce")
    sigma_eff = pd.to_numeric(
        frame.get("likelihood_sigma", frame.get("observed_flux_error_fnu_cgs")),
        errors="coerce",
    )
    residual = obs_flux - model_flux
    abs_obs_flux = np.maximum(np.abs(obs_flux.to_numpy(dtype=float)), 1.0e-300)
    abs_flux_residual = np.abs(residual.to_numpy(dtype=float))
    chi_likelihood = pd.to_numeric(frame.get("chi_likelihood"), errors="coerce")
    raw_residual = residual / obs_err
    out = pd.DataFrame(
        {
            "object_id": (
                frame["object_id"] if "object_id" in frame else frame.get("row_index")
            ),
            "row_index": frame["row_index"] if "row_index" in frame else np.nan,
            "band": frame["band"],
            "obs_flux_fnu_cgs": obs_flux,
            "obs_err_fnu_cgs": obs_err,
            "model_flux_median": model_flux,
            "sigma_eff_median": sigma_eff,
            "snr_proxy": np.abs(obs_flux.to_numpy(dtype=float))
            / np.maximum(obs_err.to_numpy(dtype=float), 1.0e-300),
            "obs_err_over_abs_flux": obs_err.to_numpy(dtype=float) / abs_obs_flux,
            "flux_residual_obs_minus_model_median": residual,
            "abs_flux_residual_median": abs_flux_residual,
            "abs_flux_residual_over_abs_flux_median": (
                abs_flux_residual / abs_obs_flux
            ),
            "residual_sigma_median": chi_likelihood,
            "raw_residual_sigma_median": raw_residual,
            "valid": (
                frame["band_used_in_likelihood"].astype(bool)
                if "band_used_in_likelihood" in frame
                else np.isfinite(chi_likelihood)
            ),
        }
    )
    out["abs_residual_sigma_median"] = out["residual_sigma_median"].abs()
    return out


def _object_residual_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_residual_summary(frame)
    if "row_index" not in frame:
        raise ValueError("Residual summary must contain row_index")
    valid = frame[frame["valid"].astype(bool)].copy()
    valid["residual_sigma_median"] = pd.to_numeric(
        valid["residual_sigma_median"],
        errors="coerce",
    )
    valid["abs_residual_sigma_median"] = valid["residual_sigma_median"].abs()
    grouped = valid.groupby("row_index", sort=False)
    rows = grouped["abs_residual_sigma_median"].agg(
        median_abs_sigma="median",
        mean_abs_sigma="mean",
        max_abs_sigma="max",
        n_valid_bands="count",
    )
    signed = grouped["residual_sigma_median"].median().rename("median_sigma")
    rows = rows.join(signed)
    rows["frac_abs_gt3"] = grouped["abs_residual_sigma_median"].apply(
        lambda values: float(np.mean(np.asarray(values, dtype=float) > 3.0))
    )
    rows["frac_abs_gt5"] = grouped["abs_residual_sigma_median"].apply(
        lambda values: float(np.mean(np.asarray(values, dtype=float) > 5.0))
    )
    optional_aggs = {
        "snr_proxy": "median_snr_proxy",
        "obs_err_over_abs_flux": "median_err_over_abs_flux",
        "abs_flux_residual_median": "median_abs_flux_residual",
        "abs_flux_residual_over_abs_flux_median": "median_frac_flux_residual",
    }
    for column, output in optional_aggs.items():
        if column in valid:
            rows[output] = grouped[column].median()
    if "obs_err_over_abs_flux" in valid:
        rows["max_err_over_abs_flux"] = grouped["obs_err_over_abs_flux"].max()
    if "object_id" in valid:
        rows["object_id"] = grouped["object_id"].first()
    return rows.reset_index()


def _method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method", sort=False):
        valid = _valid_residual_rows(group)
        residual = valid["residual_sigma_median"].to_numpy(dtype=float)
        abs_residual = np.abs(residual)
        identity = "row_index" if "row_index" in valid else "object_id"
        rows.append(
            {
                "method": method,
                "n_band_rows": int(len(group)),
                "n_valid_band_rows": int(len(valid)),
                "n_objects": int(valid[identity].nunique(dropna=True)),
                "median_residual_sigma": _nanmedian(residual),
                "median_abs_residual_sigma": _nanmedian(abs_residual),
                "p90_abs_residual_sigma": _nanquantile(abs_residual, 0.90),
                "p95_abs_residual_sigma": _nanquantile(abs_residual, 0.95),
                "max_abs_residual_sigma": _nanmax(abs_residual),
                "frac_abs_gt3": _frac_abs_gt(abs_residual, 3.0),
                "frac_abs_gt5": _frac_abs_gt(abs_residual, 5.0),
            }
        )
    return pd.DataFrame(rows)


def _band_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, band), group in frame.groupby(["method", "band"], sort=False):
        valid = _valid_residual_rows(group)
        residual = valid["residual_sigma_median"].to_numpy(dtype=float)
        abs_residual = np.abs(residual)
        rows.append(
            {
                "method": method,
                "band": band,
                "n_valid_band_rows": int(len(valid)),
                "median_residual_sigma": _nanmedian(residual),
                "median_abs_residual_sigma": _nanmedian(abs_residual),
                "median_snr_proxy": _nanmedian(
                    valid["snr_proxy"].to_numpy(dtype=float)
                )
                if "snr_proxy" in valid
                else float("nan"),
                "median_err_over_abs_flux": _nanmedian(
                    valid["obs_err_over_abs_flux"].to_numpy(dtype=float)
                )
                if "obs_err_over_abs_flux" in valid
                else float("nan"),
                "median_abs_flux_residual": _nanmedian(
                    valid["abs_flux_residual_median"].to_numpy(dtype=float)
                )
                if "abs_flux_residual_median" in valid
                else float("nan"),
                "median_frac_flux_residual": _nanmedian(
                    valid[
                        "abs_flux_residual_over_abs_flux_median"
                    ].to_numpy(dtype=float)
                )
                if "abs_flux_residual_over_abs_flux_median" in valid
                else float("nan"),
                "frac_abs_gt3": _frac_abs_gt(abs_residual, 3.0),
                "frac_abs_gt5": _frac_abs_gt(abs_residual, 5.0),
            }
        )
    return pd.DataFrame(rows)


def _object_summary(frame: pd.DataFrame) -> pd.DataFrame:
    identity = "row_index" if "row_index" in frame else "object_id"
    valid = _valid_residual_rows(frame)
    rows = []
    for (method, object_value), group in valid.groupby(
        ["method", identity],
        sort=False,
    ):
        residual = group["residual_sigma_median"].to_numpy(dtype=float)
        abs_residual = np.abs(residual)
        row = {
            "method": method,
            identity: object_value,
            "n_valid_bands": int(len(group)),
            "median_residual_sigma": _nanmedian(residual),
            "median_abs_residual_sigma": _nanmedian(abs_residual),
            "max_abs_residual_sigma": _nanmax(abs_residual),
            "median_snr_proxy": _nanmedian(group["snr_proxy"].to_numpy(dtype=float))
            if "snr_proxy" in group
            else float("nan"),
            "median_err_over_abs_flux": _nanmedian(
                group["obs_err_over_abs_flux"].to_numpy(dtype=float)
            )
            if "obs_err_over_abs_flux" in group
            else float("nan"),
            "max_err_over_abs_flux": _nanmax(
                group["obs_err_over_abs_flux"].to_numpy(dtype=float)
            )
            if "obs_err_over_abs_flux" in group
            else float("nan"),
            "median_abs_flux_residual": _nanmedian(
                group["abs_flux_residual_median"].to_numpy(dtype=float)
            )
            if "abs_flux_residual_median" in group
            else float("nan"),
            "median_frac_flux_residual": _nanmedian(
                group["abs_flux_residual_over_abs_flux_median"].to_numpy(dtype=float)
            )
            if "abs_flux_residual_over_abs_flux_median" in group
            else float("nan"),
            "frac_abs_gt3": _frac_abs_gt(abs_residual, 3.0),
            "frac_abs_gt5": _frac_abs_gt(abs_residual, 5.0),
        }
        if identity != "object_id" and "object_id" in group:
            row["object_id"] = group["object_id"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _valid_residual_rows(frame: pd.DataFrame) -> pd.DataFrame:
    residual = pd.to_numeric(frame["residual_sigma_median"], errors="coerce")
    valid = frame[residual.notna()].copy()
    if "valid" in valid:
        valid = valid[valid["valid"].astype(bool)]
    valid["residual_sigma_median"] = pd.to_numeric(
        valid["residual_sigma_median"],
        errors="coerce",
    )
    return valid[np.isfinite(valid["residual_sigma_median"].to_numpy(dtype=float))]


def _read_first_table(run: Path, stems: tuple[str, ...]) -> pd.DataFrame:
    for stem in stems:
        parquet = run / f"{stem}.parquet"
        if parquet.exists():
            return pd.read_parquet(parquet)
        csv = run / f"{stem}.csv"
        if csv.exists():
            return pd.read_csv(csv)
    names = ", ".join(stems)
    raise FileNotFoundError(f"No table found in {run} for: {names}")


def _read_optional_table(run: Path, stems: tuple[str, ...]) -> pd.DataFrame | None:
    try:
        return _read_first_table(run, stems)
    except FileNotFoundError:
        return None


def _load_npy_int(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path), dtype=np.int64)


def _write_indices(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.int64)
    path.write_text(
        "\n".join(str(int(value)) for value in values.tolist()) + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, payload: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        "# Diffsky reconstruction comparison",
        "",
        f"Rows: {payload['rows']}",
        f"Rowset: {payload.get('rowset_path') or 'all loaded rows'}",
        "",
        "| method | objects | median abs sigma | p95 abs sigma | "
        "frac >3sigma | frac >5sigma |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            "| {method} | {n_objects} | {median_abs_residual_sigma:.4g} | "
            "{p95_abs_residual_sigma:.4g} | {frac_abs_gt3:.4g} | "
            "{frac_abs_gt5:.4g} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _nanmedian(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _nanquantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def _nanmax(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else float("nan")


def _frac_abs_gt(abs_values: np.ndarray, threshold: float) -> float:
    values = np.asarray(abs_values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values > float(threshold))) if values.size else float("nan")
