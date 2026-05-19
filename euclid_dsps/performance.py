"""Lightweight runtime and resource diagnostics."""

from __future__ import annotations

import resource
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .io import ensure_dir, write_json


class PerformanceRecorder:
    """Record coarse workflow timings without adding heavy dependencies."""

    def __init__(self, verbose: bool = False) -> None:
        self._start = time.perf_counter()
        self._last = self._start
        self._rows: list[dict[str, Any]] = []
        self._verbose = bool(verbose)

    def mark(self, stage: str, **metadata: Any) -> None:
        now = time.perf_counter()
        elapsed = now - self._last
        total = now - self._start
        row: dict[str, Any] = {
            "stage": stage,
            "elapsed_seconds": elapsed,
            "total_seconds": total,
            "rss_peak_mb": _rss_peak_mb(),
            "gpu_memory_used_mb": _gpu_memory_used_mb(),
        }
        row.update(metadata)
        self._rows.append(row)
        self._last = now
        if self._verbose:
            details = " ".join(
                f"{key}={value}" for key, value in metadata.items() if value is not None
            )
            print(
                f"[bench] {stage} elapsed={elapsed:.3f}s total={total:.3f}s "
                f"rss={row['rss_peak_mb']:.1f}MB gpu={row['gpu_memory_used_mb']}MB "
                f"{details}".rstrip(),
                flush=True,
            )

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)


def write_performance_outputs(
    rows: list[dict[str, Any]], out_dir: str | Path, label: str
) -> None:
    """Write performance CSV and summary JSON for a workflow."""
    if not rows:
        return
    out = ensure_dir(out_dir)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"{label}_performance_benchmark.csv", index=False)
    numeric = frame.select_dtypes(include="number")
    summary: dict[str, Any] = {
        "n_stages": int(len(frame)),
        "total_seconds": float(frame["total_seconds"].max()),
    }
    if "rss_peak_mb" in numeric:
        summary["rss_peak_mb_max"] = float(numeric["rss_peak_mb"].max())
    if "gpu_memory_used_mb" in numeric and numeric["gpu_memory_used_mb"].notna().any():
        summary["gpu_memory_used_mb_max"] = float(
            numeric["gpu_memory_used_mb"].dropna().max()
        )
    if "n_rows" in numeric:
        chunk_rows = frame["stage"].astype(str).str.contains("chunk", na=False)
        summary["n_rows_processed"] = int(
            numeric.loc[chunk_rows, "n_rows"].fillna(0).sum()
        )
    write_json(out / f"{label}_performance_summary.json", summary)


def _rss_peak_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes. This project runs on Linux/WSL in
    # normal use, but keep the conversion sane for local docs builds elsewhere.
    if usage > 10_000_000:
        return float(usage / (1024 * 1024))
    return float(usage / 1024)


def _gpu_memory_used_mb() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = []
    for line in result.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None
