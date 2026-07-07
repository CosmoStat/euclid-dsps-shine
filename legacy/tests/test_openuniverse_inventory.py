from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd

from euclid_dsps.openuniverse.cli import main as ou_cli_main
from euclid_dsps.openuniverse.inventory import (
    classify_openuniverse_column,
    inventory_openuniverse_truth_fields,
    write_basic_truth_artifacts,
)
from euclid_dsps.openuniverse.schema import OU_FLUX_COLUMNS


def test_inventory_truth_fields_writes_json_markdown_and_schema(tmp_path) -> None:
    _write_mock_hpix(tmp_path, 10307)
    processed = tmp_path / "processed.parquet"
    _write_processed(processed)

    payload = inventory_openuniverse_truth_fields(
        output_dir=tmp_path / "report",
        processed_path=processed,
        input_root=tmp_path,
        hpix_ids=[10307],
        include_sed=True,
        sed_sample_limit=1,
    )

    report = tmp_path / "report" / "openuniverse_truth_inventory.json"
    markdown = tmp_path / "report" / "openuniverse_truth_inventory.md"
    schema = tmp_path / "report" / "truth_schema.json"
    assert report.exists()
    assert markdown.exists()
    assert schema.exists()
    assert payload["truth_schema"]["redshift_column"] == "redshift"
    levels = {
        row["quantity"]: row["truth_level"]
        for row in payload["physical_truth_summary"]
    }
    assert levels["redshift"] == "truth"
    assert levels["low_resolution_sed"] == "generated_truth"


def test_write_basic_truth_artifacts_extracts_direct_truth_columns(tmp_path) -> None:
    processed = tmp_path / "processed.parquet"
    out = tmp_path / "truth.parquet"
    schema_out = tmp_path / "truth_schema.json"
    _write_processed(processed)

    payload = write_basic_truth_artifacts(
        input_path=processed,
        output_path=out,
        schema_path=schema_out,
    )

    truth = pd.read_parquet(out)
    schema_payload = json.loads(schema_out.read_text())
    assert payload["number_of_rows"] == 2
    assert truth["redshift_truth"].tolist() == [0.1, 0.2]
    assert truth["stellar_mass_truth"].tolist() == [1.0e9, 2.0e9]
    assert schema_payload["truth_levels"]["stellar_mass_truth"] == "truth"


def test_openuniverse_standalone_cli_extract_truth(tmp_path) -> None:
    processed = tmp_path / "processed.parquet"
    out = tmp_path / "truth.parquet"
    schema_out = tmp_path / "truth_schema.json"
    _write_processed(processed)

    ou_cli_main(
        [
            "extract-truth",
            "--input",
            str(processed),
            "--out",
            str(out),
            "--schema-out",
            str(schema_out),
        ]
    )

    assert out.exists()
    assert schema_out.exists()


def test_openuniverse_standalone_cli_photoz_and_prior_overlap(tmp_path) -> None:
    truth = pd.DataFrame({"redshift_truth": [0.1, 0.2]})
    samples = pd.DataFrame({"z_sample_0": [0.09, 0.19], "z_sample_1": [0.11, 0.21]})
    prior = pd.DataFrame({"z": [0.05, 0.15, 0.25]})
    posterior = pd.DataFrame({"z": [0.09, 0.11, 0.19, 0.21]})
    truth_path = tmp_path / "truth.parquet"
    samples_path = tmp_path / "samples.parquet"
    prior_path = tmp_path / "prior.parquet"
    posterior_path = tmp_path / "posterior.parquet"
    truth.to_parquet(truth_path, index=False)
    samples.to_parquet(samples_path, index=False)
    prior.to_parquet(prior_path, index=False)
    posterior.to_parquet(posterior_path, index=False)

    metrics_out = tmp_path / "photoz_metrics.csv"
    ou_cli_main(
        [
            "photoz-metrics",
            "--samples",
            str(samples_path),
            "--truth",
            str(truth_path),
            "--out",
            str(metrics_out),
        ]
    )
    assert pd.read_csv(metrics_out)["n_objects"].tolist() == [2]

    overlap_out = tmp_path / "overlap.csv"
    ou_cli_main(
        [
            "prior-overlap",
            "--truth",
            str(truth_path),
            "--posterior",
            str(posterior_path),
            "--prior",
            str(prior_path),
            "--truth-column",
            "redshift_truth",
            "--posterior-column",
            "z",
            "--prior-column",
            "z",
            "--name",
            "redshift",
            "--out",
            str(overlap_out),
        ]
    )
    assert pd.read_csv(overlap_out)["parameter"].tolist() == ["redshift"]


def test_classify_openuniverse_column_groups_science_fields() -> None:
    assert classify_openuniverse_column("galaxy_id") == "identifier"
    assert classify_openuniverse_column("redshiftHubble") == "redshift"
    assert classify_openuniverse_column("um_source_galaxy_obs_sm") == "stellar_mass"
    assert classify_openuniverse_column("MW_av") == "dust_or_extinction"
    assert classify_openuniverse_column("shear_1") == "lensing"


def _write_processed(path) -> None:
    frame = pd.DataFrame(
        {
            "galaxy_id": [1, 2],
            "redshift": [0.1, 0.2],
            "redshiftHubble": [0.11, 0.21],
            "stellar_mass": [1.0e9, 2.0e9],
            "um_source_galaxy_obs_sm": [1.0e9, 2.0e9],
            "MW_av": [0.01, 0.02],
            "shear_1": [0.0, 0.1],
        }
    )
    for band in OU_FLUX_COLUMNS:
        frame[f"flux_truth_{band}"] = [10.0, 11.0]
        frame[f"flux_{band}"] = [10.1, 10.9]
        frame[f"fluxerr_{band}"] = [0.2, 0.2]
        frame[f"mask_{band}"] = [True, True]
    frame.to_parquet(path, index=False)


def _write_mock_hpix(root, hpix: int) -> None:
    main = pd.DataFrame(
        {
            "galaxy_id": [1, 2],
            "ra": [10.0, 11.0],
            "dec": [-1.0, -2.0],
            "redshift": [0.1, 0.2],
            "redshiftHubble": [0.11, 0.21],
            "um_source_galaxy_obs_sm": [1.0e9, 2.0e9],
        }
    )
    flux = pd.DataFrame({"galaxy_id": [1, 2]})
    for column in OU_FLUX_COLUMNS.values():
        flux[column] = [10.0, 11.0]
    main.to_parquet(root / f"galaxy_{hpix}.parquet", index=False)
    flux.to_parquet(root / f"galaxy_flux_{hpix}.parquet", index=False)
    with h5py.File(root / f"galaxy_sed_{hpix}.hdf5", "w") as handle:
        handle.create_dataset(
            "meta/wave_list",
            data=np.asarray([500.0, 700.0], dtype=np.float32),
        )
        group = handle.create_group("galaxy/103070000")
        group.create_dataset("10307000000001", data=np.ones((3, 2), dtype=np.float32))
