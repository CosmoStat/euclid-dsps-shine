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
    by_batch = _performance_by_batch(frame)
    if not by_batch.empty:
        by_batch.to_csv(out / f"{label}_performance_by_batch.csv", index=False)
    numeric = frame.select_dtypes(include="number")
    total_seconds = float(frame["total_seconds"].max())
    device = _jax_device_metadata()
    n_galaxies = _processed_galaxies(frame)
    n_gpu = int(device.get("n_devices", 0)) if device.get("backend") == "gpu" else 0
    summary: dict[str, Any] = {
        "n_stages": int(len(frame)),
        "total_seconds": total_seconds,
        "n_galaxies_processed": int(n_galaxies),
        "batch_size_max": _max_batch_size(frame),
        "seconds_per_galaxy": (
            float(total_seconds / n_galaxies) if n_galaxies > 0 else None
        ),
        "galaxies_per_second": (
            float(n_galaxies / total_seconds) if total_seconds > 0 else None
        ),
        "device": device,
        "n_gpu": n_gpu,
        "gpu_hours_total": (
            float(total_seconds * n_gpu / 3600.0) if n_gpu > 0 else None
        ),
        "gpu_hours_per_galaxy": (
            float(total_seconds * n_gpu / 3600.0 / n_galaxies)
            if n_gpu > 0 and n_galaxies > 0
            else None
        ),
    }
    if "rss_peak_mb" in numeric:
        summary["rss_peak_mb_max"] = float(numeric["rss_peak_mb"].max())
    if "gpu_memory_used_mb" in numeric and numeric["gpu_memory_used_mb"].notna().any():
        summary["gpu_memory_used_mb_max"] = float(
            numeric["gpu_memory_used_mb"].dropna().max()
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


def _processed_galaxies(frame: pd.DataFrame) -> int:
    if "n_rows" not in frame:
        return 0
    stages = frame["stage"].astype(str)
    priority = [
        "fit_chunk",
        "forward_chunk",
        "fit_population_chunk",
    ]
    for stage in priority:
        mask = stages == stage
        if mask.any():
            return int(pd.to_numeric(frame.loc[mask, "n_rows"], errors="coerce").sum())
    if "chunk_index" in frame:
        chunked = frame[frame["chunk_index"].notna()].copy()
        if not chunked.empty:
            per_chunk = pd.to_numeric(chunked["n_rows"], errors="coerce").groupby(
                chunked["chunk_index"]
            )
            return int(per_chunk.max().fillna(0).sum())
    return 0


def _max_batch_size(frame: pd.DataFrame) -> int | None:
    if "n_rows" not in frame:
        return None
    values = pd.to_numeric(frame["n_rows"], errors="coerce").dropna()
    if values.empty:
        return None
    return int(values.max())


def _performance_by_batch(frame: pd.DataFrame) -> pd.DataFrame:
    if "chunk_index" not in frame:
        return pd.DataFrame()
    work = frame[frame["chunk_index"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    if "n_rows" in work:
        work["n_rows"] = pd.to_numeric(work["n_rows"], errors="coerce")
    else:
        work["n_rows"] = 0
    work["elapsed_seconds"] = pd.to_numeric(
        work["elapsed_seconds"], errors="coerce"
    ).fillna(0.0)
    rows = []
    for chunk_index, group in work.groupby("chunk_index"):
        elapsed = float(group["elapsed_seconds"].sum())
        n_rows = (
            int(group["n_rows"].dropna().max())
            if group["n_rows"].notna().any()
            else 0
        )
        rows.append(
            {
                "chunk_index": int(chunk_index),
                "n_rows": n_rows,
                "elapsed_seconds_sum": elapsed,
                "seconds_per_galaxy": elapsed / n_rows if n_rows > 0 else None,
                "galaxies_per_second": n_rows / elapsed if elapsed > 0 else None,
                "stages": ",".join(group["stage"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("chunk_index")


def _jax_device_metadata() -> dict[str, Any]:
    try:
        import jax

        devices = jax.devices()
        backend = (jax.default_backend() or "").lower()
    except Exception as exc:  # pragma: no cover - defensive runtime probe
        return {"backend": "unknown", "n_devices": 0, "error": str(exc)}
    names = []
    platforms = []
    for device in devices:
        names.append(str(getattr(device, "device_kind", "") or device))
        platforms.append(str(getattr(device, "platform", "") or backend))
    normalized_backend = "gpu" if backend in {"gpu", "cuda", "rocm"} else backend
    return {
        "backend": normalized_backend,
        "jax_backend": backend,
        "n_devices": len(devices),
        "device_kinds": sorted(set(names)),
        "platforms": sorted(set(platforms)),
    }
