"""Download helpers with explicit size limits."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

from .inventory import rank_candidate_files
from .remote_listing import RemoteFile, load_remote_listing


def download_file(
    url: str,
    output_path: Path,
    max_bytes: int | None = None,
    overwrite: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return output_path
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        size = int(response.headers.get("content-length") or 0)
        if max_bytes is not None and size and size > max_bytes:
            raise ValueError(f"{url} is {size} bytes, above max_bytes={max_bytes}")
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        total = 0
        with tmp.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise ValueError(f"{url} exceeded max_bytes={max_bytes}")
                stream.write(chunk)
        shutil.move(tmp, output_path)
    return output_path


def download_candidate_subset(
    listing_path: Path,
    output_dir: Path,
    max_files: int = 5,
    max_total_gb: float = 5.0,
    include_patterns: tuple[str, ...] = ("diffsky_gals", "meta", "schema", "readme"),
    overwrite: bool = False,
) -> list[Path]:
    files = load_remote_listing(listing_path)
    frame = rank_candidate_files(files)
    selected: list[RemoteFile] = []
    total = 0
    max_total = int(max_total_gb * 1024**3)
    patterns = tuple(pattern.lower() for pattern in include_patterns)
    for _, row in frame.iterrows():
        name = str(row["name"])
        if patterns and not any(pattern in name.lower() for pattern in patterns):
            continue
        size = int(row["size_bytes"]) if row["size_bytes"] == row["size_bytes"] else 0
        if size and total + size > max_total:
            continue
        selected.append(RemoteFile(**{k: row[k] for k in RemoteFile.__dataclass_fields__}))
        total += size
        if len(selected) >= max_files:
            break
    paths = []
    for item in selected:
        name = Path(urlparse(item.url).path).name
        paths.append(download_file(item.url, output_dir / name, overwrite=overwrite))
    manifest = {
        "listing_path": str(listing_path),
        "output_dir": str(output_dir),
        "max_files": int(max_files),
        "max_total_gb": float(max_total_gb),
        "downloaded": [str(path) for path in paths],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "download_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return paths
