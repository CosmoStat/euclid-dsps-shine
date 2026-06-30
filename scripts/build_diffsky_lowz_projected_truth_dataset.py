"""Build the low-z Diffsky parquet augmented with DSPS projected truth.

This is intentionally an offline data-preparation script. The supervisor
notebook should load the resulting parquet directly instead of recomputing
Diffstar/Diffmah projections interactively.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "Data/diffsky/processed/"
    "hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet"
)
DEFAULT_OUTPUT = Path(
    "Data/diffsky/processed/"
    "hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr_projected_truth.parquet"
)

DIFFSKY_SFH_TRUTH_COLUMNS = [
    "diffstar_lgmcrit",
    "diffstar_lgy_at_mcrit",
    "diffstar_indx_lo",
    "diffstar_indx_hi",
    "diffstar_lg_qt",
    "diffstar_qlglgdt",
    "diffstar_lg_drop",
    "diffstar_lg_rejuv",
    "diffmah_logm0",
    "diffmah_logtc",
    "diffmah_early_index",
    "diffmah_late_index",
    "diffmah_t_peak",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Defaults to OUT with suffix .projected_truth_metadata.csv.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Defaults to OUT with suffix .summary.json.",
    )
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--n-sfh-bins", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} already exists; use --force to overwrite")
    if not args.input.exists():
        raise SystemExit(f"Missing input parquet: {args.input}")

    frame = pd.read_parquet(args.input)
    projected, metadata = build_projected_truth(frame, args.n_sfh_bins, args.chunk_size)
    augmented = frame.copy()
    if "row_index" not in augmented:
        augmented.insert(0, "row_index", projected["row_index"].to_numpy(int))
    for column in projected.columns:
        if column in {"row_index", "object_id"}:
            continue
        augmented[column] = projected[column].to_numpy()
    augmented["projected_truth_available"] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(args.out, index=False)

    metadata_out = args.metadata_out or args.out.with_suffix(
        ".projected_truth_metadata.csv"
    )
    metadata.to_csv(metadata_out, index=False)

    summary_out = args.summary_out or args.out.with_suffix(".summary.json")
    summary = {
        "input_path": str(args.input),
        "output_path": str(args.out),
        "row_count": int(len(augmented)),
        "column_count": int(len(augmented.columns)),
        "projected_truth_columns": [
            column
            for column in projected.columns
            if column not in {"row_index", "object_id"}
        ],
        "redshift_min": float(pd.to_numeric(augmented["redshift_true"]).min()),
        "redshift_max": float(pd.to_numeric(augmented["redshift_true"]).max()),
        "metadata_path": str(metadata_out),
    }
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_projected_truth(
    frame: pd.DataFrame,
    n_sfh_bins: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_index = (
        frame["row_index"].to_numpy(int)
        if "row_index" in frame
        else np.arange(len(frame), dtype=int)
    )
    object_id = (
        frame["object_id"].to_numpy(int)
        if "object_id" in frame
        else row_index.copy()
    )
    truth = pd.DataFrame({"row_index": row_index, "object_id": object_id})
    metadata_rows: list[dict[str, Any]] = []

    def assign(
        parameter: str,
        values: Any,
        *,
        source_kind: str,
        source_columns: str,
        formula: str,
        notes: str = "",
    ) -> None:
        series = pd.to_numeric(pd.Series(values), errors="coerce")
        arr = series.to_numpy(float)
        truth[parameter] = arr
        finite = float(np.isfinite(arr).mean()) if arr.size else float("nan")
        metadata_rows.append(
            {
                "parameter": parameter,
                "source_kind": source_kind,
                "source_columns": source_columns,
                "formula": formula,
                "finite_fraction": finite,
                "notes": notes,
            }
        )

    required_direct = {
        "redshift_true",
        "logsm_true",
        "logsfr_true",
        "dust_av",
        "dust_delta",
        *DIFFSKY_SFH_TRUTH_COLUMNS,
    }
    missing = sorted(required_direct - set(frame.columns))
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing))

    assign(
        "z_obs",
        frame["redshift_true"],
        source_kind="direct_truth",
        source_columns="redshift_true",
        formula="z_obs = redshift_true",
    )
    assign(
        "log10_stellar_mass",
        frame["logsm_true"],
        source_kind="direct_truth",
        source_columns="logsm_true",
        formula="log10_stellar_mass = logsm_true",
    )
    assign(
        "log10_sfr_at_obs",
        frame["logsfr_true"],
        source_kind="direct_truth",
        source_columns="logsfr_true",
        formula="log10_sfr_at_obs = logsfr_true",
    )
    if "logssfr_true" in frame:
        assign(
            "log10_ssfr_at_obs",
            frame["logssfr_true"],
            source_kind="direct_truth",
            source_columns="logssfr_true",
            formula="log10_ssfr_at_obs = logssfr_true",
        )
    else:
        assign(
            "log10_ssfr_at_obs",
            pd.to_numeric(frame["logsfr_true"], errors="coerce")
            - pd.to_numeric(frame["logsm_true"], errors="coerce"),
            source_kind="derived_truth",
            source_columns="logsfr_true,logsm_true",
            formula="log10_ssfr_at_obs = logsfr_true - logsm_true",
        )

    dlogs = project_diffsky_sfh_to_popcosmos_dlogs(
        frame,
        n_sfh_bins=n_sfh_bins,
        chunk_size=chunk_size,
    )
    for index in range(1, 7):
        assign(
            f"dlog10_sfr_{index}",
            dlogs[:, index - 1],
            source_kind="projected_generated_truth",
            source_columns=",".join(DIFFSKY_SFH_TRUTH_COLUMNS),
            formula=(
                "Diffstar/Diffmah generated SFH -> "
                "project_sfh_to_popcosmos_dlogsfr_jax"
            ),
            notes="Projected into PopCosmos adjacent lookback-bin log-SFR ratios.",
        )

    assign(
        "tau2",
        pd.to_numeric(frame["dust_av"], errors="coerce") / 1.086,
        source_kind="projected_generated_truth",
        source_columns="dust_av",
        formula="tau2 = dust_av / 1.086",
        notes="Matches diffsky_basic_dust_params_jax.",
    )
    assign(
        "dust_index_n",
        frame["dust_delta"],
        source_kind="projected_generated_truth",
        source_columns="dust_delta",
        formula="dust_index_n = dust_delta",
        notes="Matches diffsky_basic_dust_params_jax.",
    )
    for parameter, note in {
        "log10_stellar_metallicity": (
            "The active low-z Diffsky parquet has no object-level stellar "
            "metallicity truth column."
        ),
        "tau1_over_tau2": (
            "The active low-z Diffsky parquet has no object-level birth-cloud "
            "dust ratio."
        ),
    }.items():
        assign(
            parameter,
            np.full(len(frame), np.nan),
            source_kind="missing",
            source_columns="",
            formula="",
            notes=note,
        )

    return truth, pd.DataFrame(metadata_rows)


def project_diffsky_sfh_to_popcosmos_dlogs(
    frame: pd.DataFrame,
    *,
    n_sfh_bins: int,
    chunk_size: int,
) -> np.ndarray:
    try:
        import jax
        import jax.numpy as jnp
        from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z

        from euclid_dsps.model import (
            build_diffsky_basic_sfh_table_jax,
            project_sfh_to_popcosmos_dlogsfr_jax,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Projected truth requires jax, dsps, diffstar, and diffmah. "
            "Run this in the project environment, e.g. `conda activate shine`."
        ) from exc

    def project_one(z_obs, *values):
        params = {
            name: value for name, value in zip(DIFFSKY_SFH_TRUTH_COLUMNS, values)
        }
        t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
        gal_t_table = jnp.linspace(
            jnp.asarray(0.05, dtype=jnp.float32),
            jnp.maximum(t_obs, jnp.asarray(0.06, dtype=jnp.float32)),
            int(n_sfh_bins),
        )
        sfh = build_diffsky_basic_sfh_table_jax(gal_t_table, t_obs, params)
        return project_sfh_to_popcosmos_dlogsfr_jax(gal_t_table, sfh, t_obs)

    project_batch = jax.jit(jax.vmap(project_one))
    chunks = []
    for start in range(0, len(frame), int(chunk_size)):
        chunk = frame.iloc[start : start + int(chunk_size)]
        arrays = [jnp.asarray(chunk["redshift_true"].to_numpy(np.float32))]
        arrays.extend(
            jnp.asarray(chunk[column].to_numpy(np.float32))
            for column in DIFFSKY_SFH_TRUTH_COLUMNS
        )
        chunks.append(np.asarray(project_batch(*arrays), dtype=float))
        print(f"[projected-truth] rows {start + len(chunk):,}/{len(frame):,}")
    return np.vstack(chunks) if chunks else np.empty((0, 6), dtype=float)


if __name__ == "__main__":
    main()
