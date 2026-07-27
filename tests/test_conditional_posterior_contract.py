from __future__ import annotations

import json

import pandas as pd
import pytest

from euclid_dsps.prior_learning.spline15d import SPLINE15D_PARAMETER_NAMES
from scripts.validate_feniks_conditional_posterior_inputs import validate_inputs


def _write_inputs(tmp_path, *, integrity: str = "PASS"):
    catalog = tmp_path / "amortized"
    catalog.mkdir()
    frame = pd.DataFrame(
        {"object_id": [1, 2]} | {name: [0.1, 0.2] for name in SPLINE15D_PARAMETER_NAMES}
    )
    records = {}
    for split in ("train", "test"):
        path = catalog / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        records[split] = {"rows": 2}
    (catalog / "amortized_catalog_contract.json").write_text(
        json.dumps(
            {
                "version": 1,
                "truth_kind": "exact_spline15d",
                "join_key": "object_id",
                "parameter_names": list(SPLINE15D_PARAMETER_NAMES),
                "splits": records,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best.eqx"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.with_suffix(".eqx.json").write_text(
        json.dumps(
            {
                "version": 1,
                "parameter_names": list(SPLINE15D_PARAMETER_NAMES),
                "architecture": {"latent_dim": len(SPLINE15D_PARAMETER_NAMES)},
                "flow_integrity": {"status": integrity},
            }
        ),
        encoding="utf-8",
    )
    return catalog, checkpoint


def test_conditional_posterior_inputs_accept_exact_contract(tmp_path) -> None:
    catalog, checkpoint = _write_inputs(tmp_path)
    validate_inputs(catalog, checkpoint)


def test_conditional_posterior_inputs_reject_bad_prior(tmp_path) -> None:
    catalog, checkpoint = _write_inputs(tmp_path, integrity="FAIL")
    with pytest.raises(ValueError, match="integrity"):
        validate_inputs(catalog, checkpoint)
