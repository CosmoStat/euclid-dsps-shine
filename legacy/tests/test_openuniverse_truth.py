from __future__ import annotations

import pandas as pd
import pytest

from euclid_dsps.openuniverse.diffsky_truth import (
    extract_basic_truth_table,
    extract_extended_diffsky_truth,
    infer_truth_schema,
)


def test_truth_schema_detects_public_openuniverse_truths() -> None:
    frame = pd.DataFrame(
        {
            "galaxy_id": [1],
            "redshift": [0.2],
            "redshiftHubble": [0.21],
            "um_source_galaxy_obs_sm": [1.0e10],
            "dust_av": [0.1],
            "halo_mass": [1.0e12],
        }
    )

    schema = infer_truth_schema(frame)
    basic = extract_basic_truth_table(frame)

    assert schema.redshift_column == "redshift"
    assert schema.stellar_mass_column == "um_source_galaxy_obs_sm"
    assert schema.dust_columns == ("dust_av",)
    assert schema.halo_columns == ("halo_mass",)
    assert basic["redshift_truth"].tolist() == [0.2]
    assert basic.attrs["truth_levels"]["stellar_mass_truth"] == "truth"


def test_extended_diffsky_truth_missing_optional_dependency_is_clear(tmp_path) -> None:
    out = tmp_path / "extended.parquet"

    try:
        frame = extract_extended_diffsky_truth(
            tmp_path,
            [9812],
            out,
            require_diffsky=True,
        )
    except ImportError as exc:
        assert "requires optional dependency" in str(exc)
        return
    except NotImplementedError as exc:
        assert "not wired" in str(exc)
        return

    assert frame.empty or frame["status"].tolist() == ["unavailable"]


def test_extended_diffsky_truth_no_require_writes_unavailable_when_absent(
    tmp_path,
) -> None:
    out = tmp_path / "extended.parquet"

    try:
        frame = extract_extended_diffsky_truth(
            tmp_path,
            [9812],
            out,
            require_diffsky=False,
        )
    except NotImplementedError:
        pytest.skip(
            "diffsky is installed but extraction is intentionally not wired yet"
        )

    assert out.exists()
    assert frame["status"].tolist() == ["unavailable"]
