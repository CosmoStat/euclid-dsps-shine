"""Dataset-level diagnostic plots for prepared Diffsky tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_dataset_diagnostics(
    dataset_path: str | Path, out_dir: str | Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    frame = pd.read_parquet(dataset_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for column, filename in [
        ("redshift_true", "redshift_histogram.png"),
        ("logsm_true", "logsm_histogram.png"),
        ("logssfr_true", "logssfr_histogram.png"),
        ("logsfr_true", "logsfr_histogram.png"),
        ("logmp_true", "logmp_histogram.png"),
    ]:
        if column not in frame:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        frame[column].plot.hist(bins=60, ax=ax)
        ax.set_xlabel(column)
        ax.set_ylabel("N")
        fig.tight_layout()
        path = out / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)
    missing = pd.DataFrame(
        {
            "column": frame.columns,
            "missing_fraction": [
                float(frame[column].isna().mean()) for column in frame.columns
            ],
        }
    )
    missing_path = out / "missing_values_report.csv"
    missing.to_csv(missing_path, index=False)
    outputs.append(missing_path)
    summary = out / "dataset_summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Diffsky Dataset Summary",
                "",
                f"- rows: {len(frame)}",
                f"- columns: {len(frame.columns)}",
                f"- flux bands: {sum(column.startswith('flux_') for column in frame.columns)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(summary)
    return outputs
