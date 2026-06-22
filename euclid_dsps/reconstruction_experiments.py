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
            "flux_residual_obs_minus_model_median": residual,
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
