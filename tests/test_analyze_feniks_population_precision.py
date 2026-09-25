import json

import pandas as pd
import pytest

from scripts import analyze_feniks_population_precision as analysis
from scripts.feniks_avi_experiments import sha, write


def make_mirror(root):
    write(root / "MANIFEST.json", {"settings": {"bootstraps": 1}})
    folders = ["cache", "decoder"]
    folders += [f"metric/{arm}" for arm in analysis.ARMS]
    folders += [
        f"bootstrap/repeat_000/{arm}_{ratio}"
        for arm in analysis.ARMS
        for ratio in analysis.RATIOS
    ]
    for folder in folders:
        (root / folder).mkdir(parents=True)
        write(root / folder / "FINAL.json", {"status": "COMPLETE", "artifacts": {}})
    artifact = root / "cache/small.json"
    write(artifact, {"value": 1})
    write(
        root / "cache/FINAL.json",
        {
            "status": "COMPLETE",
            "artifacts": {"small.json": sha(artifact), "large.npy": "not-synced"},
        },
    )


def test_lightweight_mirror_checks_present_artifacts_without_claiming_absent_ones(
    tmp_path,
):
    make_mirror(tmp_path)
    result = analysis.verify_mirror(tmp_path)
    assert result == {"verified": 1, "missing_not_verified": ["cache/large.npy"]}
    write(tmp_path / "cache/small.json", {"value": 2})
    with pytest.raises(ValueError, match="corrupted"):
        analysis.verify_mirror(tmp_path)


def test_missing_or_incomplete_cell_is_not_accepted(tmp_path):
    make_mirror(tmp_path)
    receipt = tmp_path / "bootstrap/repeat_000/noisy_photometry_exact/FINAL.json"
    write(receipt, {"status": "RUNNING", "artifacts": {}})
    with pytest.raises(ValueError, match="incomplete"):
        analysis.verify_mirror(tmp_path)
    receipt.unlink()
    with pytest.raises(FileNotFoundError):
        analysis.verify_mirror(tmp_path)


def test_same_integrator_is_not_evidence_of_numerical_improvement(tmp_path):
    (tmp_path / "decoder").mkdir()
    write(
        tmp_path / "MANIFEST.json",
        {
            "settings": {"contracts": {"decoder_p95_abs_sigma": 0.25}},
        },
    )
    write(
        tmp_path / "decoder/decoder_model.json",
        {
            "model": {"photometry_integrator": "merged_gauss4_v1"},
        },
    )
    write(
        tmp_path / "decoder/FINAL.json",
        {
            "baseline_p95_max": 1.4,
            "merged_p95_max": 1.4,
        },
    )
    frame = pd.DataFrame(
        {
            "row": [7, 7],
            "band": ["r", "r"],
            "variant": ["baseline", "merged"],
            "decoded_flux": [1.0, 1.0],
        }
    )
    frame.to_csv(tmp_path / "decoder/residuals.csv", index=False)
    result = analysis.decoder_comparison(tmp_path)
    assert not result["distinct_integrators"]
    assert not result["numerical_improvement_demonstrated"]
    assert not result["catalogue_compatibility_pass"]
    assert result["maximum_flux_difference"] == 0
    json.dumps(result, allow_nan=False)
    frame.loc[1, "row"] = 8
    frame.to_csv(tmp_path / "decoder/residuals.csv", index=False)
    with pytest.raises(ValueError, match="identical row-band"):
        analysis.decoder_comparison(tmp_path)
