"""Remote Apache/NERSC directory listing helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests


@dataclass(frozen=True)
class RemoteFile:
    url: str
    name: str
    size_bytes: int | None
    extension: str
    depth: int
    is_dir: bool = False
    modified: str | None = None


class _ApacheIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.links: list[str | None] = []
        self._row: list[str] = []
        self._row_href: str | None = None
        self._cell_text: list[str] = []
        self._in_td = False
        self._current_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
            self._row_href = None
        elif tag == "td":
            self._in_td = True
            self._cell_text = []
        elif tag == "a":
            self._current_href = dict(attrs).get("href")
            if self._row_href is None:
                self._row_href = self._current_href

    def handle_data(self, data: str) -> None:
        if self._in_td:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td":
            self._row.append("".join(self._cell_text).strip())
            self._in_td = False
        elif tag == "a":
            self._current_href = None
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self.links.append(self._row_href)


def list_remote_directory(
    url: str, *, depth: int = 0, timeout: int = 60
) -> list[RemoteFile]:
    """List one remote Apache-style directory without downloading data files."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    parser = _ApacheIndexParser()
    parser.feed(response.text)
    files: list[RemoteFile] = []
    for row, href in zip(parser.rows, parser.links, strict=False):
        if not href or len(row) < 4:
            continue
        name = row[1]
        if not name or name == "Parent Directory" or href.startswith("/cfs/"):
            continue
        is_dir = href.endswith("/")
        full_url = urljoin(url, href)
        parsed = urlparse(full_url)
        extension = "" if is_dir else Path(parsed.path).suffix.lower()
        files.append(
            RemoteFile(
                url=full_url,
                name=name.rstrip("/"),
                size_bytes=_parse_apache_size(row[3]),
                extension=extension,
                depth=depth,
                is_dir=is_dir,
                modified=_clean_text(row[2]) or None,
            )
        )
    return files


def crawl_remote_tree(
    root_url: str,
    max_depth: int = 2,
    allowed_extensions: tuple[str, ...] = (
        ".hdf5",
        ".h5",
        ".parquet",
        ".json",
        ".yaml",
        ".yml",
        ".txt",
        ".md",
    ),
) -> list[RemoteFile]:
    """Recursively crawl a remote listing to a bounded depth."""
    out: list[RemoteFile] = []

    def visit(url: str, depth: int) -> None:
        for item in list_remote_directory(url, depth=depth):
            if item.is_dir:
                if depth < max_depth:
                    visit(item.url, depth + 1)
                continue
            if not allowed_extensions or item.extension in allowed_extensions:
                out.append(item)

    visit(root_url, 0)
    return out


def write_remote_listing(files: list[RemoteFile], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([asdict(item) for item in files], indent=2), encoding="utf-8"
    )
    return output


def load_remote_listing(path: str | Path) -> list[RemoteFile]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RemoteFile(**row) for row in rows]


def _parse_apache_size(text: str) -> int | None:
    cleaned = _clean_text(text)
    if not cleaned or cleaned == "-":
        return None
    match = re.match(r"^([0-9.]+)\s*([KMGTPE]?)$", cleaned)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}.get(unit)
    return None if scale is None else int(value * scale)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
