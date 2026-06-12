"""Full Diffsky HLTDS physical-validation orchestration and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from euclid_dsps.config import load_config
from euclid_dsps.io import ensure_dir, write_json


def run_diffsky_full_validation(
    config: dict[str, Any],
    *,
    out_dir: str | Path,
    dataset_path: str | Path | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    n_samples: int | None = None,
    posterior_samples: int | None = None,
    prior_samples: int | None = None,
    seed: int | None = None,
    report_only: bool = False,
    runs: list[tuple[str, Path]] | None = None,
    closure_run: str | Path | None = None,
    verbose: bool = True,
    progress: bool = True,
) -> Path:
    """Run or aggregate the full Diffsky validation workflow."""
    out = ensure_dir(out_dir)
    full = dict(config.get("full_validation", {}) or {})
    dataset = Path(dataset_path or config.get("catalog_path") or _required(full, "dataset"))
    stage_outputs: dict[str, Any] = {
        "dataset_path": str(dataset),
        "report_only": bool(report_only),
    }
    inference_runs = list(runs or [])
    closure_dir = Path(closure_run) if closure_run else None

    if not report_only:
        prior_basic_dir, prior_extended_dir = _run_supervised_prior_stages(
            full,
            dataset,
            out,
            limit=limit,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            verbose=verbose,
            progress=progress,
        )
        stage_outputs["supervised_prior_basic"] = str(prior_basic_dir)
        stage_outputs["supervised_prior_extended"] = str(prior_extended_dir)

        closure_dir = out / "trueparam_forward_closure"
        _run_forward_closure_stage(
            full,
            dataset,
            closure_dir,
            limit=limit,
            batch_size=batch_size,
        )
        stage_outputs["forward_closure"] = str(closure_dir)

        inference_runs = _run_amortized_stages(
            full,
            dataset,
            out,
            supervised_prior_checkpoint=prior_basic_dir / "checkpoints" / "best.eqx",
            limit=limit,
            batch_size=batch_size,
            epochs=epochs,
            n_samples=n_samples,
            posterior_samples=posterior_samples,
            prior_samples=prior_samples,
            seed=seed,
            verbose=verbose,
            progress=progress,
        )
        stage_outputs["inference_runs"] = [
            {"label": label, "run_dir": str(path)} for label, path in inference_runs
        ]
        _run_redshift_and_population_reports(
            full,
            dataset,
            out,
            inference_runs,
        )
        stage_outputs["redshift_ablation"] = str(out / "redshift_ablation")
        stage_outputs["population_realism"] = str(out / "population_realism")
    stage_outputs.setdefault(
        "inference_runs",
        [{"label": label, "run_dir": str(path)} for label, path in inference_runs],
    )
    if closure_dir is not None:
        stage_outputs.setdefault("forward_closure", str(closure_dir))

    report = write_full_validation_report(
        out,
        dataset_path=dataset,
        inference_runs=inference_runs,
        closure_run=closure_dir,
        stage_outputs=stage_outputs,
    )
    return report


def write_full_validation_report(
    out_dir: str | Path,
    *,
    dataset_path: str | Path,
    inference_runs: list[tuple[str, Path]] | None = None,
    closure_run: str | Path | None = None,
    stage_outputs: dict[str, Any] | None = None,
) -> Path:
    """Write a compact final validation report from existing stage outputs."""
    out = ensure_dir(out_dir)
    inference_runs = list(inference_runs or [])
    alpha_rows = []
    mass_rows = []
    photoz_rows = []
    for label, run_dir in inference_runs:
        run = Path(run_dir)
        alpha = _read_alpha(run / "inference_summary.json")
        alpha_rows.append({"stage": label, "run_dir": str(run), **alpha})
        mass_rows.extend(_read_mass_metrics(label, run / "posterior_vs_truth_metrics.csv"))
        photoz = _read_first_row(run / "photoz_metrics.csv")
        if photoz:
            photoz_rows.append({"label": label, **photoz})
    if closure_run:
        closure = Path(closure_run)
        alpha_rows.append(
            {
                "stage": "trueparam_forward_closure",
                "run_dir": str(closure),
                **_read_alpha(closure / "forward_closure_summary.json"),
            }
        )
    summary = {
        "dataset_path": str(dataset_path),
        "stage_outputs": stage_outputs or {},
        "global_sed_scale": alpha_rows,
        "mass_recovery": mass_rows,
        "photoz_metrics": photoz_rows,
    }
    write_json(out / "full_validation_summary.json", summary)
    report = out / "full_validation_report.md"
    _write_report_markdown(report, summary)
    return report


def _run_supervised_prior_stages(
    full: dict[str, Any],
    dataset: Path,
    out: Path,
    *,
    limit: int | None,
    batch_size: int | None,
    epochs: int | None,
    seed: int | None,
    verbose: bool,
    progress: bool,
) -> tuple[Path, Path]:
    from euclid_dsps.prior_learning.train import (
        prior_learning_config,
        train_supervised_prior,
    )

    basic_cfg = _load_stage_config(full, "supervised_prior_basic_config")
    extended_cfg = _load_stage_config(full, "supervised_prior_extended_config")
    basic_dir = out / "supervised_prior_basic"
    extended_dir = out / "supervised_prior_extended"
    for cfg, target in ((basic_cfg, basic_dir), (extended_cfg, extended_dir)):
        prior_cfg = prior_learning_config(cfg)
        train_supervised_prior(
            cfg,
            target,
            dataset_path=dataset,
            schema_name=prior_cfg["schema"],
            limit=limit,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            verbose=verbose,
            progress=progress,
        )
    return basic_dir, extended_dir


def _run_forward_closure_stage(
    full: dict[str, Any],
    dataset: Path,
    out: Path,
    *,
    limit: int | None,
    batch_size: int | None,
) -> None:
    from euclid_dsps.diffsky_forward_closure import run_diffsky_forward_closure

    closure_cfg = _with_dataset_path(
        _load_stage_config(full, "forward_closure_config"),
        dataset,
    )
    run_diffsky_forward_closure(
        closure_cfg,
        dataset_path=dataset,
        out_dir=out,
        limit=limit,
        batch_size=int(batch_size or 64),
    )


def _run_amortized_stages(
    full: dict[str, Any],
    dataset: Path,
    out: Path,
    *,
    supervised_prior_checkpoint: Path,
    limit: int | None,
    batch_size: int | None,
    epochs: int | None,
    n_samples: int | None,
    posterior_samples: int | None,
    prior_samples: int | None,
    seed: int | None,
    verbose: bool,
    progress: bool,
) -> list[tuple[str, Path]]:
    from euclid_dsps.amortized.config import amortized_config
    from euclid_dsps.amortized.infer import infer_amortized_fs2
    from euclid_dsps.amortized.train import train_amortized_fs2

    stages = (
        ("standard_normal", "amortized_standard_normal_config"),
        ("supervised_prior", "amortized_supervised_prior_config"),
        ("joint_realnvp", "amortized_joint_realnvp_config"),
    )
    runs = []
    for label, key in stages:
        cfg = _with_dataset_path(_load_stage_config(full, key), dataset)
        if label == "supervised_prior":
            amortized = dict(cfg.get("amortized", {}) or {})
            prior = dict(amortized.get("prior", {}) or {})
            prior["checkpoint"] = str(supervised_prior_checkpoint)
            amortized["prior"] = prior
            cfg["amortized"] = amortized
        acfg = amortized_config(cfg)
        training = acfg["training"]
        inference = acfg["inference"]
        train_dir = out / f"amortized_{label}"
        infer_dir = out / f"amortized_{label}_infer"
        train_amortized_fs2(
            cfg,
            train_dir,
            limit=limit,
            batch_size=int(batch_size or training.get("batch_size", 32)),
            epochs=int(epochs or training.get("epochs", 10)),
            n_samples=int(n_samples or training.get("n_samples", 1)),
            seed=int(seed if seed is not None else training.get("seed", 42)),
            verbose=verbose,
            progress=progress,
            dataset_label=f"Diffsky HLTDS {label}",
        )
        infer_amortized_fs2(
            cfg,
            infer_dir,
            checkpoint=train_dir / "checkpoints" / "best.eqx",
            feature_stats_path=train_dir / "feature_stats.json",
            limit=limit,
            batch_size=int(batch_size or training.get("batch_size", 32)),
            posterior_samples=int(
                posterior_samples
                if posterior_samples is not None
                else inference.get("posterior_samples", 32)
            ),
            prior_samples=int(
                prior_samples
                if prior_samples is not None
                else inference.get("prior_samples", 8192)
            ),
            seed=int(seed if seed is not None else training.get("seed", 42)),
            decoder_sample_chunk_size=int(
                inference.get("decoder_sample_chunk_size", 1)
            ),
            dataset_label=f"Diffsky HLTDS {label}",
        )
        runs.append((label, infer_dir))
    return runs


def _run_redshift_and_population_reports(
    full: dict[str, Any],
    dataset: Path,
    out: Path,
    runs: list[tuple[str, Path]],
) -> None:
    from euclid_dsps.amortized.prior_overlap import write_diffsky_prior_overlap_report
    from euclid_dsps.diffsky_redshift_ablation import run_redshift_ablation

    run_redshift_ablation(
        dataset_path=dataset,
        runs=runs,
        out_dir=out / "redshift_ablation",
    )
    population_root = ensure_dir(out / "population_realism")
    for label, run_dir in runs:
        cfg_key = {
            "standard_normal": "amortized_standard_normal_config",
            "supervised_prior": "amortized_supervised_prior_config",
            "joint_realnvp": "amortized_joint_realnvp_config",
        }[label]
        cfg = _with_dataset_path(_load_stage_config(full, cfg_key), dataset)
        write_diffsky_prior_overlap_report(
            dataset_path=dataset,
            run_dir=run_dir,
            out_dir=population_root / label,
            config=cfg,
        )


def _read_alpha(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    alpha = payload.get("global_sed_scale", {})
    if not isinstance(alpha, dict):
        return {}
    keys = (
        "alpha_sed",
        "log_alpha_sed",
        "delta_mag_global",
        "alpha_prior_penalty",
        "large_scale_warning",
        "warning",
        "mode",
        "enabled",
        "trainable",
    )
    return {key: alpha[key] for key in keys if key in alpha}


def _read_mass_metrics(label: str, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if "metric_name" not in frame:
        return []
    keep = frame[frame["metric_name"].isin(["mass_bias_raw", "mass_bias_alpha_corrected"])]
    rows = []
    for _, row in keep.iterrows():
        rows.append(
            {
                "label": label,
                "metric_name": str(row["metric_name"]),
                "bias": _float_or_none(row.get("bias")),
                "median_bias": _float_or_none(row.get("median_bias")),
                "rmse": _float_or_none(row.get("rmse")),
                "sigma_mad": _float_or_none(row.get("sigma_mad")),
            }
        )
    return rows


def _read_first_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return {
        str(key): _jsonable(value)
        for key, value in frame.iloc[0].to_dict().items()
    }


def _write_report_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Diffsky Full Physical Validation",
        "",
        f"- dataset: `{summary['dataset_path']}`",
        "",
        "## Global SED scale calibration",
        "",
        _markdown_table(pd.DataFrame(summary["global_sed_scale"]))
        if summary["global_sed_scale"]
        else "_No alpha_sed rows found._",
        "",
        "## Mass recovery raw vs alpha-corrected",
        "",
        _markdown_table(pd.DataFrame(summary["mass_recovery"]))
        if summary["mass_recovery"]
        else "_No mass recovery metrics found._",
        "",
        "## Redshift calibration summary",
        "",
        _markdown_table(pd.DataFrame(summary["photoz_metrics"]))
        if summary["photoz_metrics"]
        else "_No redshift metrics found._",
        "",
        "## Stage outputs",
        "",
    ]
    for key, value in dict(summary.get("stage_outputs", {})).items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}" if pd.notna(value) else "nan")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _load_stage_config(full: dict[str, Any], key: str) -> dict[str, Any]:
    return load_config(_required(full, key))


def _with_dataset_path(config: dict[str, Any], dataset: Path) -> dict[str, Any]:
    config = dict(config)
    config["catalog_path"] = str(dataset)
    return config


def _required(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not value:
        raise ValueError(f"full_validation.{key} is required")
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _jsonable(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _float_or_none(value) -> float | None:
    value = _jsonable(value)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
