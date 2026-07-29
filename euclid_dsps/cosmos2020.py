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


@dataclass(frozen=True)
class CosmosBand:
    name: str
    farmer_prefix: str
    extinction_coefficient: float
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
    CosmosBand("u_megaprime_sagem", "CFHT_u", 4.807, "CFHT/MegaCam.u"),
    CosmosBand("hsc_g", "HSC_g", 3.690, "Subaru/HSC.g"),
    CosmosBand("hsc_r", "HSC_r", 2.715, "Subaru/HSC.r"),
    CosmosBand("hsc_i", "HSC_i", 2.000, "Subaru/HSC.i"),
    CosmosBand("hsc_z", "HSC_z", 1.515, "Subaru/HSC.z"),
    CosmosBand("hsc_y", "HSC_y", 1.298, "Subaru/HSC.Y"),
    CosmosBand("uvista_y_cosmos", "UVISTA_Y", 1.213, "Paranal/VISTA.Y"),
    CosmosBand("uvista_j_cosmos", "UVISTA_J", 0.874, "Paranal/VISTA.J"),
    CosmosBand("uvista_h_cosmos", "UVISTA_H", 0.565, "Paranal/VISTA.H"),
    CosmosBand("uvista_ks_cosmos", "UVISTA_Ks", 0.365, "Paranal/VISTA.Ks"),
    CosmosBand("ia427_cosmos", "SC_IB427", 4.261, "Subaru/Suprime.IB427"),
    CosmosBand("ia464_cosmos", "SC_IB464", 3.844, "Subaru/Suprime.IB464"),
    CosmosBand("ia484_cosmos", "SC_IA484", 3.622, "Subaru/Suprime.IB484"),
    CosmosBand("ia505_cosmos", "SC_IB505", 3.425, "Subaru/Suprime.IB505"),
    CosmosBand("ia527_cosmos", "SC_IA527", 3.265, "Subaru/Suprime.IB527"),
    CosmosBand("ia574_cosmos", "SC_IB574", 2.938, "Subaru/Suprime.IB574"),
    CosmosBand("ia624_cosmos", "SC_IA624", 2.694, "Subaru/Suprime.IB624"),
    CosmosBand("ia679_cosmos", "SC_IA679", 2.431, "Subaru/Suprime.IB679"),
    CosmosBand("ia709_cosmos", "SC_IB709", 2.290, "Subaru/Suprime.IB709"),
    CosmosBand("ia738_cosmos", "SC_IA738", 2.151, "Subaru/Suprime.IB738"),
    CosmosBand("ia767_cosmos", "SC_IA767", 1.997, "Subaru/Suprime.IB767"),
    CosmosBand("ia827_cosmos", "SC_IB827", 1.748, "Subaru/Suprime.IB827"),
    CosmosBand("NB711.SuprimeCam", "SC_NB711", 2.268, "Subaru/Suprime.NB711"),
    CosmosBand("NB816.SuprimeCam", "SC_NB816", 1.787, "Subaru/Suprime.NB816"),
    CosmosBand("irac1_cosmos", "IRAC_CH1", 0.163, "Spitzer/IRAC.I1"),
    CosmosBand("irac2_cosmos", "IRAC_CH2", 0.112, "Spitzer/IRAC.I2"),
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


def prepare_farmer_catalog(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the public A24 selection and produce generic DSPS columns.

    Fluxes and errors are corrected for Milky Way extinction. They remain in
    microJy in the parquet; the generic DSPS loader performs the final fnu_cgs
    conversion declared in the YAML configuration.
    """
    ebv = pd.to_numeric(frame["EBV_MW"], errors="coerce").to_numpy(float)
    output = pd.DataFrame(
        {
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
        correction = np.power(10.0, 0.4 * band.extinction_coefficient * ebv)
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
        corrected_flux = flux * correction
        corrected_error = error * correction
        valid = (
            catalog_valid
            & np.isfinite(corrected_flux)
            & np.isfinite(corrected_error)
            & (corrected_error > 0.0)
        )
        output[f"flux_{band.name}"] = corrected_flux
        # The generic loader derives its mask from finite positive errors.
        output[f"fluxerr_{band.name}"] = np.where(
            catalog_valid, corrected_error, np.nan
        )
        output[f"valid_{band.name}"] = valid

    selected = (
        (output["flag_combined"].to_numpy() == 0)
        & (output["lp_type"].to_numpy() == 0)
        & np.isfinite(output["ebv_mw"].to_numpy())
        & (
            output["flux_hsc_r"].to_numpy()
            > R_LIMIT_UJY
        )
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
        "selection": "FLAG_COMBINED == 0 and lp_type == 0 and corrected HSC r < 25",
        "r_limit_ujy": float(R_LIMIT_UJY),
        "mw_extinction_applied_to": ["flux", "flux_error"],
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
