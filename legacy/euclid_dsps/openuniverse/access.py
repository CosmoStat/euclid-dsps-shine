"""Path resolution for OpenUniverse HEALPix file groups."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path


def openuniverse_paths_for_hpix(
    hpix: int,
    root: str | Path,
) -> dict[str, str]:
    """Return expected main/flux/SED paths for one nside=32 HEALPix id.

    ``root`` may be a local directory, an S3 URI, or a format string containing
    ``{hpix}``, ``{kind}``, and/or ``{filename}``.
    """
    hpix_int = int(hpix)
    if hpix_int < 0:
        raise ValueError("hpix must be non-negative")
    root_text = str(root)
    if _is_s3_uri(root_text):
        _require_s3_support()
    filenames = {
        "main": f"galaxy_{hpix_int}.parquet",
        "flux": f"galaxy_flux_{hpix_int}.parquet",
        "sed": f"galaxy_sed_{hpix_int}.hdf5",
    }
    return {
        kind: _format_path(root_text, hpix=hpix_int, kind=kind, filename=filename)
        for kind, filename in filenames.items()
    }


def resolve_openuniverse_paths(
    hpix_ids: Sequence[int],
    root: str | Path,
) -> list[dict[str, str]]:
    """Resolve OpenUniverse file groups for a sequence of HEALPix ids."""
    return [openuniverse_paths_for_hpix(hpix, root) for hpix in hpix_ids]


def _format_path(root: str, *, hpix: int, kind: str, filename: str) -> str:
    if "{" in root and "}" in root:
        return root.format(hpix=hpix, kind=kind, filename=filename)
    if _is_uri(root):
        return root.rstrip("/") + "/" + filename
    return str(Path(root) / filename)


def _is_uri(value: str) -> bool:
    return "://" in value


def _is_s3_uri(value: str) -> bool:
    return value.lower().startswith("s3://")


def _require_s3_support() -> None:
    if importlib.util.find_spec("s3fs") is not None:
        return
    try:
        from pyarrow import fs

        if hasattr(fs, "S3FileSystem"):
            return
    except ImportError:
        pass
    raise ImportError(
        "OpenUniverse S3 paths require pandas/pyarrow S3 support. Install "
        "`s3fs` or a pyarrow build with S3 support, then retry."
    )
