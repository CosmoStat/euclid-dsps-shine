from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_script_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_feniks_individual_posteriors.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_feniks_individual_posteriors", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(path: Path) -> dict:
    parameters = (
        "z_obs",
        "log10_stellar_mass",
        "log10_stellar_metallicity",
        "dust_av",
        "dust_delta",
    )
    return {
        "catalog_path": str(path),
        "dataset": {"id_column": "object_id"},
        "fit": {"free_parameters": {name: {} for name in parameters}},
        "truth": {
            "parameter_columns": {
                name: {"column": name, "kind": "exact_spline_truth"}
                for name in parameters
            }
        },
    }


def test_representative_selection_is_unique_and_finds_extremes(tmp_path) -> None:
    module = _load_script_module()
    n_rows = 200
    frame = pd.DataFrame(
        {
            "object_id": np.arange(10_000, 10_000 + n_rows),
            "z_obs": np.linspace(0.01, 5.0, n_rows),
            "log10_stellar_mass": np.linspace(7.0, 12.0, n_rows),
            "log10_stellar_metallicity": np.sin(np.arange(n_rows) / 20),
            "dust_av": np.linspace(0.0, 3.0, n_rows),
            "dust_delta": np.cos(np.arange(n_rows) / 17),
            "logssfr_true": np.linspace(-14.0, -8.0, n_rows),
        }
    )
    dataset = tmp_path / "test.parquet"
    frame.to_parquet(dataset, index=False)

    selected = module.select_representative_rows(_config(dataset), dataset)

    assert selected["example_key"].tolist() == [
        "typical",
        "nearby",
        "high_z",
        "massive",
        "dusty",
        "quenched",
        "star_forming",
    ]
    assert selected["row_index"].is_unique
    by_key = selected.set_index("example_key")
    assert by_key.loc["quenched", "log10_ssfr_true"] == -14.0
    assert by_key.loc["star_forming", "log10_ssfr_true"] >= -8.1


def test_individual_corner_contains_truth_and_posterior_metadata(tmp_path) -> None:
    module = _load_script_module()
    rng = np.random.default_rng(7)
    columns = [
        "z_obs",
        "log10_stellar_mass",
        "log10_stellar_metallicity",
        "dust_av",
        "dust_delta",
    ]
    posterior = pd.DataFrame(rng.normal(size=(128, len(columns))), columns=columns)
    posterior.insert(0, "row_index", 3)
    posterior.insert(1, "object_id", 1003)
    prior = pd.DataFrame(rng.normal(size=(256, len(columns))), columns=columns)
    truth = pd.DataFrame(
        [[3, 1003, 0.2, 10.0, -1.0, 0.5, -0.2]],
        columns=[
            "row_index",
            "object_id",
            *columns,
        ],
    )
    selection = pd.DataFrame(
        [
            {
                "order": 1,
                "example_key": "typical",
                "object_id": 1003,
                "row_index": 3,
            }
        ]
    )

    records = module._write_corners(
        posterior,
        prior,
        truth,
        selection,
        tmp_path,
        _config(tmp_path / "unused.parquet"),
    )

    assert records[0]["posterior_rows"] == 128
    assert records[0]["truth_rows"] == 1
    corner = tmp_path / records[0]["corner"]
    assert corner.exists() and corner.stat().st_size > 0
    metadata = corner.with_name(f"{corner.stem}_columns.csv")
    columns_frame = pd.read_csv(metadata)
    assert columns_frame["truth_finite_rows"].eq(1).all()
