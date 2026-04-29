"""Download helpers for small native DSPS smoke-test assets."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from .io import ensure_dir, write_json

DEFAULT_ASSETS = {
    "ssp_data_fsps_v3.2_lgmet_age.h5": "https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_fsps_v3.2_lgmet_age.h5",
    "lsst_g_transmission.h5": "https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/filters/lsst_g_transmission.h5",
}


def download_assets(out_dir: str | Path, overwrite: bool = False) -> list[dict[str, str | bool]]:
    """Download DSPS SSP/filter files used by the smoke-test config."""
    out = ensure_dir(out_dir)
    records = []
    for filename, url in DEFAULT_ASSETS.items():
        path = out / filename
        downloaded = False
        if overwrite or not path.exists():
            urlretrieve(url, path)
            downloaded = True
        records.append(
            {
                "filename": filename,
                "url": url,
                "path": str(path),
                "downloaded": downloaded,
            }
        )
    write_json(out / "downloaded_assets.json", records)
    return records
