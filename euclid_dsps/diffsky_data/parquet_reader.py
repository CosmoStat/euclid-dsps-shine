"""Parquet inspection helpers for future Diffsky table variants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def inspect_parquet_file(path: str | Path, sample_size: int = 1024) -> dict:
    file_path = Path(path)
    frame = pd.read_parquet(file_path).head(sample_size)
    return {
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "n_sample_rows": len(frame),
    }
