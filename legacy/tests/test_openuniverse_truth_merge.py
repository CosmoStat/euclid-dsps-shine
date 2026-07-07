from __future__ import annotations

import json

import pandas as pd
import pytest

from euclid_dsps.openuniverse.cli import main as ou_cli_main
from euclid_dsps.openuniverse.truth_merge import merge_external_truth_table


def test_merge_external_truth_table_prefixes_generated_truth_columns(tmp_path) -> None:
    input_path = tmp_path / "ou.parquet"
    truth_path = tmp_path / "diffsky_truth.parquet"
    out = tmp_path / "merged.parquet"
    schema = tmp_path / "schema.json"
    _write_base(input_path)
    pd.DataFrame(
        {
            "galaxy_id": [1, 2],
            "diffstar_u_param": [0.1, 0.2],
            "halo_mass": [1.0e12, 2.0e12],
        }
    ).to_parquet(truth_path, index=False)

    payload = merge_external_truth_table(
        input_path=input_path,
        truth_path=truth_path,
        output_path=out,
        schema_path=schema,
        truth_columns=["diffstar_u_param"],
    )

    merged = pd.read_parquet(out)
    schema_payload = json.loads(schema.read_text())
    assert payload["truth_level"] == "generated_truth"
    assert "generated_truth_diffstar_u_param" in merged
    assert "generated_truth_halo_mass" not in merged
    assert schema_payload["matched_fraction"] == 1.0


def test_merge_external_truth_table_rejects_duplicate_ids(tmp_path) -> None:
    input_path = tmp_path / "ou.parquet"
    truth_path = tmp_path / "truth.parquet"
    out = tmp_path / "merged.parquet"
    _write_base(input_path)
    pd.DataFrame({"galaxy_id": [1, 1], "x": [0.1, 0.2]}).to_parquet(
        truth_path,
        index=False,
    )

    with pytest.raises(ValueError, match="not unique"):
        merge_external_truth_table(
            input_path=input_path,
            truth_path=truth_path,
            output_path=out,
        )


def test_merge_external_truth_cli(tmp_path) -> None:
    input_path = tmp_path / "ou.parquet"
    truth_path = tmp_path / "truth.csv"
    out = tmp_path / "merged.parquet"
    schema = tmp_path / "schema.json"
    _write_base(input_path)
    pd.DataFrame({"galaxy_id": [1, 2], "dust2": [0.3, 0.4]}).to_csv(
        truth_path,
        index=False,
    )

    ou_cli_main(
        [
            "merge-external-truth",
            "--input",
            str(input_path),
            "--truth",
            str(truth_path),
            "--out",
            str(out),
            "--schema-out",
            str(schema),
            "--truth-level",
            "proxy",
            "--prefix",
            "proxy_",
        ]
    )

    merged = pd.read_parquet(out)
    payload = json.loads(schema.read_text())
    assert merged["proxy_dust2"].tolist() == [0.3, 0.4]
    assert payload["truth_level"] == "proxy"


def _write_base(path) -> None:
    pd.DataFrame(
        {
            "galaxy_id": [1, 2],
            "redshift": [0.1, 0.2],
            "stellar_mass": [1.0e9, 2.0e9],
        }
    ).to_parquet(path, index=False)
