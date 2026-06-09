"""Remote and local inventory helpers for Diffsky/OpenCosmo data."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .hdf5_reader import inspect_hdf5_file
from .remote_listing import RemoteFile, load_remote_listing
from .truth import detect_truth_columns


def classify_remote_file(remote_file: RemoteFile) -> dict[str, Any]:
    name = remote_file.name.lower()
    score = 0
    tags: list[str] = []
    if "diffsky_gals" in name:
        score += 5
        tags.append("diffsky_gals")
    if remote_file.extension in {".hdf5", ".h5"}:
        score += 4
        tags.append("hdf5")
    for keyword, value in {
        "truth": 3,
        "phot": 3,
        "flux": 3,
        "mag": 3,
        "sed": 2,
        "meta": 2,
        "schema": 2,
        "readme": 2,
        "param": 2,
        "transmission": 2,
        "ssp": 1,
    }.items():
        if keyword in name:
            score += value
            tags.append(keyword)
    size = remote_file.size_bytes
    if size is not None:
        if size <= 200 * 1024**2:
            score += 1
            tags.append("small_or_medium")
        elif size > 5 * 1024**3:
            score -= 5
            tags.append("large")
    return {**asdict(remote_file), "score": score, "tags": ",".join(sorted(set(tags)))}


def rank_candidate_files(files: list[RemoteFile]) -> pd.DataFrame:
    frame = pd.DataFrame([classify_remote_file(item) for item in files])
    if frame.empty:
        return frame
    return frame.sort_values(["score", "size_bytes"], ascending=[False, True], na_position="last")


def write_candidate_report(frame: pd.DataFrame, csv_path: str | Path) -> tuple[Path, Path]:
    csv_out = Path(csv_path)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_out, index=False)
    md_out = csv_out.with_suffix(".md")
    lines = ["# Diffsky Remote Candidate Files", ""]
    if frame.empty:
        lines.append("No candidate files found.")
    else:
        lines.extend(
            [
                "## Top Candidates",
                "",
                _markdown_table(frame.head(20)),
                "",
                "## Notes",
                "",
                "- `diffsky_gals` HDF5 shards are first-priority science files.",
                "- `transmission_curves` and `ssp_data` are supporting decoder/filter assets.",
                "- Download one or a few shards first; do not mirror the whole directory by default.",
            ]
        )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_out, md_out


def inventory_remote_listing(listing_path: str | Path, out_csv: str | Path) -> pd.DataFrame:
    files = load_remote_listing(listing_path)
    frame = rank_candidate_files(files)
    write_candidate_report(frame, out_csv)
    return frame


def inventory_local_hdf5(root: str | Path, out_path: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    files = sorted([*root_path.rglob("*.hdf5"), *root_path.rglob("*.h5")])
    reports = [inspect_hdf5_file(path) for path in files]
    all_columns = []
    for report in reports:
        all_columns.extend(item["name"] for item in report["datasets"])
    truth_report = detect_truth_columns(all_columns)
    output = {
        "root": str(root_path),
        "n_files": len(files),
        "files": reports,
        "truth_report": asdict(truth_report),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_local_inventory_markdown(output, out.with_suffix(".md"))
    return output


def _write_local_inventory_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Diffsky Local Inventory",
        "",
        f"- root: `{report['root']}`",
        f"- files inspected: {report['n_files']}",
        "",
        "## Truth/Photometry Detection",
        "",
        _markdown_table(pd.DataFrame([report["truth_report"]])),
        "",
        "## Files",
        "",
    ]
    for item in report["files"]:
        interesting = [
            d["name"]
            for d in item["datasets"]
            if d.get("interesting")
        ][:40]
        lines.extend(
            [
                f"### `{Path(item['path']).name}`",
                "",
                f"- size: {item['size_bytes']} bytes",
                f"- datasets: {len(item['datasets'])}",
                f"- groups: {len(item['groups'])}",
                f"- interesting columns sample: {', '.join(interesting) if interesting else 'none'}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(_markdown_cell(row[col]) for col in cols) + " |")
    return "\n".join(rows)


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")
