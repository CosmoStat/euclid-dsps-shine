from __future__ import annotations

import json
from pathlib import Path

from scripts.finalize_feniks_sc_drws_epoch160_catalogue_calibration import (
    EXPECTED,
    finalize,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def test_catalogue_calibration_finalizer_preserves_scientific_contracts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "epoch160"
    root = tmp_path / "calibration"
    _write_json(
        source / "EPOCH160_EVALUATION_COMPLETE.json",
        {
            "status": "DIAGNOSTIC_COMPLETE",
            "epoch": 160,
            "training_frozen_before_truth": True,
            "truth_used_for_training_or_checkpoint_selection": False,
            "catalogue_objects": 4706,
        },
    )
    for variant, ess in (("raw", 5.95), ("ema", 5.88)):
        _write_json(
            source / "heldout" / f"{variant}_support_summary.json",
            {"status": "FAIL", "median_raw_ess": ess},
        )
    for (mode, diagnostic), (draws, models) in EXPECTED.items():
        out = root / mode / diagnostic
        _write_json(
            out / f"{diagnostic}_summary.json",
            {
                "status": "complete",
                "models": models,
                "num_objects": 4706,
                "num_posterior_samples": draws,
            },
        )
        (out / "DONE").touch()
        plot = "mira_scores.png" if diagnostic == "mira" else "tarp_coverage.png"
        (out / plot).write_bytes(b"plot")

    receipt = finalize(source_evaluation_root=source, root=root)

    assert receipt["status"] == "DIAGNOSTIC_COMPLETE"
    assert receipt["catalogue_objects_evaluated"] == 4706
    assert receipt["evaluations"]["common32"]["draws_per_object"] == 32
    assert receipt["evaluations"]["q256"]["draws_per_object"] == 256
    assert receipt["iw_support_warning"]["median_effective_samples"] == {
        "raw": 5.95,
        "ema": 5.88,
    }
    assert receipt["scientific_promotion"] is False
    assert receipt["truth_used_for_training_or_checkpoint_selection"] is False
    assert "not relabeled" in receipt["contracts"]["posterior_aggregate"]
    assert finalize(source_evaluation_root=source, root=root) == receipt


def test_catalogue_calibration_launcher_reuses_existing_inference() -> None:
    submit = (
        ROOT / "scripts/submit_feniks_sc_drws_epoch160_catalogue_calibration.sh"
    ).read_text()
    worker = (
        ROOT / "scripts/feniks_sc_drws_epoch160_catalogue_calibration_h100.slurm"
    ).read_text()

    assert "--array=0-3%4" in submit
    assert "EPOCH160_EVALUATION_COMPLETE.json" in submit
    assert "posterior_samples" in submit
    assert "amortized-infer-diffsky" not in submit
    assert "MODE=common32; DIAGNOSTIC=mira; DRAWS=32" in worker
    assert "MODE=q256; DIAGNOSTIC=tarp; DRAWS=256" in worker
    assert "raw_iw=$RAW_IW32" in worker
    assert "ema_iw=$EMA_IW32" in worker
    assert "--drop-nonfinite-truth" in worker
