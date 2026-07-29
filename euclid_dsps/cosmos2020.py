"""COSMOS2020 Farmer v2.1 contracts used by the Pop-COSMOS comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

ESO_TAP_URL = "https://archive.eso.org/tap_cat"
ESO_TABLE = "COSMOS2020_FARMER_V1"
ESO_FARMER_V21_ID = "ADP.2022-06-21T19:13:38.112"
ESO_FARMER_V21_URL = (
    f"https://dataportal.eso.org/dataPortal/file/{ESO_FARMER_V21_ID}"
)
ESO_FARMER_V21_SIZE = 2_923_603_200
POPCOSMOS_COMMIT = "28690aab5ae1aeca01db1ceaf7bc7fe2a58378a7"
POPCOSMOS_URL = "https://github.com/Cosmo-Pop/pop-cosmos.git"
ZENODO_RECORD = "13820043"
SPECZ_COMPILATION_URL = (
    "https://media.githubusercontent.com/media/cosmosastro/speczcompilation/"
    "main/specz_compilation/specz_compilation_COSMOS_DR1.1_unique.fits"
)
SPECZ_COMPILATION_SIZE = 70_223_040
SPECZ_COMPILATION_SHA256 = (
    "6ffd1145ed9caeba6c16f8e4267415682562b1a37549ac07a070ba5eb6336e99"
)
SPECZ_MIN_CONFIDENCE = 50.0


@dataclass(frozen=True)
class CosmosBand:
    name: str
    farmer_prefix: str
    extinction_coefficient: float
    farmer_lephare_offset_mag: float
    svo_id: str

    @property
    def flux_column(self) -> str:
        return f"{self.farmer_prefix}_FLUX"

    @property
    def error_column(self) -> str:
        return f"{self.farmer_prefix}_FLUXERR"

    @property
    def valid_column(self) -> str:
        return f"{self.farmer_prefix}_VALID"


# Order is the public A24 Pop-COSMOS COSMOS_FILTERS order.
COSMOS_BANDS = (
    CosmosBand(
        "u_megaprime_sagem",
        "CFHT_ustar",
        4.674,
        -0.023,
        "CFHT/MegaCam.u_1",
    ),
    CosmosBand("hsc_g", "HSC_g", 3.690, 0.073, "Subaru/HSC.g"),
    CosmosBand("hsc_r", "HSC_r", 2.715, 0.101, "Subaru/HSC.r"),
    CosmosBand("hsc_i", "HSC_i", 2.000, 0.038, "Subaru/HSC.i"),
    CosmosBand("hsc_z", "HSC_z", 1.515, 0.036, "Subaru/HSC.z"),
    CosmosBand("hsc_y", "HSC_y", 1.298, 0.086, "Subaru/HSC.Y"),
    CosmosBand("uvista_y_cosmos", "UVISTA_Y", 1.213, 0.054, "Paranal/VISTA.Y"),
    CosmosBand("uvista_j_cosmos", "UVISTA_J", 0.874, 0.017, "Paranal/VISTA.J"),
    CosmosBand("uvista_h_cosmos", "UVISTA_H", 0.565, -0.045, "Paranal/VISTA.H"),
    CosmosBand("uvista_ks_cosmos", "UVISTA_Ks", 0.365, 0.000, "Paranal/VISTA.Ks"),
    CosmosBand("ia427_cosmos", "SC_IB427", 4.261, -0.104, "Subaru/Suprime.IB427"),
    CosmosBand("ia464_cosmos", "SC_IB464", 3.844, -0.044, "Subaru/Suprime.IB464"),
    CosmosBand("ia484_cosmos", "SC_IA484", 3.622, -0.021, "Subaru/Suprime.IB484"),
    CosmosBand("ia505_cosmos", "SC_IB505", 3.425, -0.018, "Subaru/Suprime.IB505"),
    CosmosBand("ia527_cosmos", "SC_IA527", 3.265, -0.045, "Subaru/Suprime.IB527"),
    CosmosBand("ia574_cosmos", "SC_IB574", 2.938, -0.084, "Subaru/Suprime.IB574"),
    CosmosBand("ia624_cosmos", "SC_IA624", 2.694, 0.005, "Subaru/Suprime.IB624"),
    CosmosBand("ia679_cosmos", "SC_IA679", 2.431, 0.166, "Subaru/Suprime.IB679"),
    CosmosBand("ia709_cosmos", "SC_IB709", 2.290, -0.023, "Subaru/Suprime.IB709"),
    CosmosBand("ia738_cosmos", "SC_IA738", 2.151, -0.034, "Subaru/Suprime.IB738"),
    CosmosBand("ia767_cosmos", "SC_IA767", 1.997, -0.032, "Subaru/Suprime.IB767"),
    CosmosBand("ia827_cosmos", "SC_IB827", 1.748, -0.069, "Subaru/Suprime.IB827"),
    CosmosBand("NB711.SuprimeCam", "SC_NB711", 2.268, -0.010, "Subaru/Suprime.NB711"),
    CosmosBand("NB816.SuprimeCam", "SC_NB816", 1.787, -0.064, "Subaru/Suprime.NB816"),
    CosmosBand("irac1_cosmos", "IRAC_CH1", 0.163, -0.212, "Spitzer/IRAC.I1"),
    CosmosBand("irac2_cosmos", "IRAC_CH2", 0.112, -0.219, "Spitzer/IRAC.I2"),
)

FARMER_METADATA_COLUMNS = (
    "ID",
    "ALPHA_J2000",
    "DELTA_J2000",
    "FLAG_COMBINED",
    "EBV_MW",
    "lp_type",
    "lp_zBEST",
)
DEFAULT_SUBSET_SIZES = (512, 5_000, 20_000, 40_000)
R_LIMIT_UJY = 10.0 ** ((23.9 - 25.0) / 2.5)


def farmer_columns() -> tuple[str, ...]:
    columns = list(FARMER_METADATA_COLUMNS)
    for band in COSMOS_BANDS:
        columns.extend((band.flux_column, band.error_column, band.valid_column))
    return tuple(columns)


def farmer_adql(max_rows: int | None = None) -> str:
    top = f"TOP {int(max_rows)} " if max_rows is not None else ""
    return f"SELECT {top}{','.join(farmer_columns())} FROM {ESO_TABLE}"


def read_farmer_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".fits", ".fit", ".fts"}:
        with fits.open(path, memmap=True) as hdus:
            data = hdus[1].data
            table = Table({name: data[name] for name in farmer_columns()})
        frame = table.to_pandas()
    elif path.suffix.lower() in {".vot", ".xml"}:
        table = Table.read(path)
        frame = table.to_pandas()
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    frame.columns = [str(column) for column in frame.columns]
    missing = sorted(set(farmer_columns()) - set(frame.columns))
    if missing:
        raise ValueError("Farmer input is missing columns: " + ", ".join(missing))
    return frame


def prepare_farmer_catalog(
    frame: pd.DataFrame,
    *,
    public_catalog_rows: np.ndarray | None = None,
    min_public_retention: float = 0.99,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the public A24 selection and produce generic DSPS columns.

    Fluxes receive the official readcat Milky Way and Farmer+LePhare
    corrections. Errors are left unchanged, matching the released notebook.
    Values remain in microJy in the parquet.
    """
    ebv = pd.to_numeric(frame["EBV_MW"], errors="coerce").to_numpy(float)
    output = pd.DataFrame(
        {
            "catalog_index": np.arange(len(frame), dtype=np.int64),
            "object_id": pd.to_numeric(frame["ID"], errors="raise").astype("int64"),
            "ra_deg": pd.to_numeric(frame["ALPHA_J2000"], errors="coerce"),
            "dec_deg": pd.to_numeric(frame["DELTA_J2000"], errors="coerce"),
            "ebv_mw": ebv,
            "lp_zbest": pd.to_numeric(frame["lp_zBEST"], errors="coerce"),
            "flag_combined": pd.to_numeric(
                frame["FLAG_COMBINED"], errors="coerce"
            ),
            "lp_type": pd.to_numeric(frame["lp_type"], errors="coerce"),
        }
    )
    for band in COSMOS_BANDS:
        mw_correction = np.power(
            10.0, 0.4 * band.extinction_coefficient * ebv
        )
        zero_point_correction = np.power(
            10.0, -0.4 * band.farmer_lephare_offset_mag
        )
        flux = pd.to_numeric(frame[band.flux_column], errors="coerce").to_numpy(float)
        error = pd.to_numeric(
            frame[band.error_column], errors="coerce"
        ).to_numpy(float)
        catalog_valid = (
            pd.to_numeric(frame[band.valid_column], errors="coerce")
            .fillna(0)
            .to_numpy(int)
            > 0
        )
        measurement_admitted = (
            catalog_valid
            if public_catalog_rows is None
            else np.ones(len(frame), dtype=bool)
        )
        corrected_flux = flux * mw_correction * zero_point_correction
        corrected_error = error
        valid = (
            measurement_admitted
            & np.isfinite(corrected_flux)
            & np.isfinite(corrected_error)
            & (corrected_error > 0.0)
        )
        output[f"flux_{band.name}"] = corrected_flux
        # The generic loader derives its mask from finite positive errors.
        output[f"fluxerr_{band.name}"] = np.where(
            measurement_admitted, corrected_error, np.nan
        )
        output[f"valid_{band.name}"] = valid

    public_cohort_audit = None
    if public_catalog_rows is None:
        selected = (
            (output["flag_combined"].to_numpy() == 0)
            & (output["lp_type"].to_numpy() == 0)
            & np.isfinite(output["ebv_mw"].to_numpy())
            & (
                output["flux_hsc_r"].to_numpy()
                > R_LIMIT_UJY
            )
        )
        selection = (
            "FLAG_COMBINED == 0 and lp_type == 0 and official-readcat HSC r < 25"
        )
    else:
        rows = np.asarray(public_catalog_rows, dtype=np.int64)
        if len(np.unique(rows)) != len(rows):
            raise ValueError("Public COSMOS cohort contains duplicate catalog rows")
        if len(rows) and (rows.min() < 0 or rows.max() >= len(frame)):
            raise IndexError("Public COSMOS cohort row is outside the Farmer catalog")
        requested = np.zeros(len(frame), dtype=bool)
        requested[rows] = True
        valid_metadata = (
            (output["flag_combined"].to_numpy() == 0)
            & (output["lp_type"].to_numpy() == 0)
            & np.isfinite(output["ebv_mw"].to_numpy())
        )
        valid_photometry = np.ones(len(frame), dtype=bool)
        invalid_photometry_by_band: dict[str, dict[str, int]] = {}
        for band in COSMOS_BANDS:
            flux = output[f"flux_{band.name}"].to_numpy(float)
            error = output[f"fluxerr_{band.name}"].to_numpy(float)
            valid_flux = np.isfinite(flux) & (flux > 0.0)
            valid_error = np.isfinite(error) & (error > 0.0)
            valid_photometry &= valid_flux & valid_error
            invalid_photometry_by_band[band.name] = {
                "flux": int(np.count_nonzero(requested & ~valid_flux)),
                "error": int(np.count_nonzero(requested & ~valid_error)),
            }
        selected = requested & valid_metadata & valid_photometry
        requested_count = int(np.count_nonzero(requested))
        selected_count = int(np.count_nonzero(selected))
        retention = selected_count / requested_count if requested_count else 1.0
        requested_frame = output.loc[requested]
        public_cohort_audit = {
            "requested_rows": requested_count,
            "retained_rows": selected_count,
            "excluded_rows": requested_count - selected_count,
            "retention_fraction": retention,
            "excluded_flag_combined": int(
                np.count_nonzero(
                    requested & (output["flag_combined"].to_numpy() != 0)
                )
            ),
            "excluded_lp_type": int(
                np.count_nonzero(requested & (output["lp_type"].to_numpy() != 0))
            ),
            "excluded_nonfinite_ebv": int(
                np.count_nonzero(
                    requested & ~np.isfinite(output["ebv_mw"].to_numpy())
                )
            ),
            "invalid_photometry_by_band": invalid_photometry_by_band,
            "flag_combined_counts": {
                str(key): int(value)
                for key, value in requested_frame["flag_combined"]
                .value_counts(dropna=False)
                .items()
            },
            "lp_type_counts": {
                str(key): int(value)
                for key, value in requested_frame["lp_type"]
                .value_counts(dropna=False)
                .items()
            },
        }
        if retention < min_public_retention:
            raise ValueError(
                "Usable Farmer intersection retains only "
                f"{selected_count}/{requested_count} public T24 objects "
                f"({retention:.3%}), below {min_public_retention:.1%}; "
                f"audit={json.dumps(public_cohort_audit, sort_keys=True)}"
            )
        selection = (
            "published T24 MAGCUT_r == Y and XRAY == N Farmer IDs intersected "
            "with usable Phase-3 Farmer photometry, FLAG_COMBINED == 0, "
            "and lp_type == 0"
        )
    selected_frame = output.loc[selected].copy()
    selected_frame["r_abmag_mw_corrected"] = 23.9 - 2.5 * np.log10(
        selected_frame["flux_hsc_r"].to_numpy(float)
    )
    selected_frame = selected_frame.sort_values("object_id").reset_index(drop=True)
    selected_frame.insert(0, "row_index", np.arange(len(selected_frame), dtype=np.int64))
    manifest = {
        "input_rows": int(len(frame)),
        "selected_rows": int(len(selected_frame)),
        "selection": selection,
        "r_limit_ujy": float(R_LIMIT_UJY),
        "mw_extinction_applied_to": ["flux"],
        "farmer_lephare_offsets_applied_to": ["flux"],
        "public_catalog_ids": public_catalog_rows is not None,
        "public_cohort_audit": public_cohort_audit,
        "catalog_valid_flags_applied": public_catalog_rows is None,
        "output_flux_units": "microjy",
        "n_bands": len(COSMOS_BANDS),
        "band_order": [band.name for band in COSMOS_BANDS],
        "selection_modelled_in_rws": False,
        "claim_limit": (
            "Initial RWS learns the selected-catalog distribution; intrinsic "
            "population claims require an explicit selection normalization."
        ),
    }
    return selected_frame, manifest


def read_spectroscopic_catalog(path: str | Path) -> pd.DataFrame:
    """Read the public COSMOS spectroscopic compilation columns we use."""
    columns = (
        "Id_specz",
        "specz",
        "Confidence_level",
        "survey",
        "compilation_year",
        "Id_COS20_Farmer",
    )
    with fits.open(path, memmap=True) as hdus:
        data = hdus[1].data
        missing = sorted(set(columns) - set(data.names))
        if missing:
            raise ValueError(
                "Spectroscopic compilation is missing columns: "
                + ", ".join(missing)
            )
        table = Table({name: data[name] for name in columns})
    frame = table.to_pandas()
    for column in ("survey",):
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: (
                    value.decode("utf-8").strip()
                    if isinstance(value, bytes)
                    else str(value).strip()
                )
            )
    return frame


def attach_spectroscopic_redshifts(
    selected: pd.DataFrame,
    spectroscopy: pd.DataFrame,
    *,
    public_summary: pd.DataFrame | None = None,
    min_confidence: float = SPECZ_MIN_CONFIDENCE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach reliable public spec-z values by exact COSMOS2020 Farmer ID."""
    spec = spectroscopy.copy()
    spec["object_id"] = pd.to_numeric(
        spec["Id_COS20_Farmer"], errors="coerce"
    )
    spec["redshift_spec"] = pd.to_numeric(spec["specz"], errors="coerce")
    spec["specz_confidence_level"] = pd.to_numeric(
        spec["Confidence_level"], errors="coerce"
    )
    valid = (
        np.isfinite(spec["object_id"])
        & (spec["object_id"] != -999)
        & np.isfinite(spec["redshift_spec"])
        & (spec["redshift_spec"] >= 0.0)
        & np.isfinite(spec["specz_confidence_level"])
        & (spec["specz_confidence_level"] >= float(min_confidence))
    )
    spec = spec.loc[valid].copy()
    spec["object_id"] = spec["object_id"].astype(np.int64)
    spec["specz_survey"] = spec.get("survey", "").astype(str)
    spec["specz_compilation_year"] = pd.to_numeric(
        spec.get("compilation_year"), errors="coerce"
    )
    spec["specz_id"] = pd.to_numeric(spec.get("Id_specz"), errors="coerce")
    duplicate_rows = int(spec.duplicated("object_id", keep=False).sum())
    spec = spec.sort_values(
        ["object_id", "specz_confidence_level", "specz_compilation_year", "specz_id"],
        ascending=[True, False, False, True],
        na_position="last",
    ).drop_duplicates("object_id", keep="first")

    columns = [
        "object_id",
        "redshift_spec",
        "specz_confidence_level",
        "specz_survey",
        "specz_compilation_year",
        "specz_id",
    ]
    result = selected.merge(spec[columns], on="object_id", how="left", validate="one_to_one")
    result["redshift_true"] = result["redshift_spec"]

    flagged_ids: set[int] = set()
    if public_summary is not None:
        summary = public_summary.copy()
        flag = summary["z_SPEC"].astype(str).str.upper().eq("Y")
        flagged = pd.to_numeric(
            summary.loc[flag, "INDEX_COSMOS"], errors="coerce"
        ).dropna()
        flagged_ids = set(flagged.astype(np.int64).tolist())
    result["t24_specz_flag"] = result["object_id"].isin(flagged_ids)

    matched = np.isfinite(result["redshift_spec"].to_numpy(float))
    flagged = result["t24_specz_flag"].to_numpy(bool)
    audit = {
        "source": "cosmosastro/speczcompilation COSMOS DR1.1 unique",
        "join": "object_id == Id_COS20_Farmer",
        "minimum_confidence_level": float(min_confidence),
        "input_rows": int(len(spectroscopy)),
        "eligible_unique_farmer_ids": int(len(spec)),
        "duplicate_candidate_rows": duplicate_rows,
        "selected_rows": int(len(result)),
        "matched_selected_rows": int(np.count_nonzero(matched)),
        "t24_specz_flagged_selected_rows": int(np.count_nonzero(flagged)),
        "matched_and_t24_flagged_rows": int(np.count_nonzero(matched & flagged)),
        "t24_flagged_without_public_value": int(np.count_nonzero(flagged & ~matched)),
        "redshift_true_semantics": (
            "public spectroscopic redshift where available; NaN otherwise"
        ),
    }
    return result, audit


def deterministic_nested_order(object_ids: pd.Series, seed: int) -> np.ndarray:
    def key(value: Any) -> bytes:
        return hashlib.sha256(f"{seed}:{int(value)}".encode("ascii")).digest()

    return np.asarray(
        sorted(range(len(object_ids)), key=lambda index: key(object_ids.iloc[index])),
        dtype=np.int64,
    )


def write_nested_subsets(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    sizes: tuple[int, ...] = DEFAULT_SUBSET_SIZES,
    seed: int = 260727,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    order = deterministic_nested_order(frame["object_id"], seed)
    full = frame.iloc[order].reset_index(drop=True)
    full["row_index"] = np.arange(len(full), dtype=np.int64)
    paths: dict[str, str] = {}
    effective_sizes = sorted({min(int(size), len(full)) for size in sizes if size > 0})
    for size in effective_sizes:
        path = output / f"farmer_a24_n{size}.parquet"
        full.iloc[:size].to_parquet(path, index=False)
        paths[str(size)] = str(path)
    full_path = output / "farmer_a24_full.parquet"
    full.to_parquet(full_path, index=False)
    paths["full"] = str(full_path)
    ids_path = output / "nested_order.csv"
    full[["row_index", "object_id"]].to_csv(ids_path, index=False)
    return {
        "seed": int(seed),
        "ordering": "sha256(seed:object_id)",
        "sizes": effective_sizes,
        "paths": paths,
        "nested_order": str(ids_path),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
