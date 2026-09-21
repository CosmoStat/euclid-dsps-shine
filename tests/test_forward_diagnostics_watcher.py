import json
import subprocess
from pathlib import Path


def test_compact_watcher_hides_artifacts_and_shows_epochs(tmp_path):
    root = tmp_path
    (root / "baseline/report").mkdir(parents=True)
    (root / "baseline/report/FINAL.json").write_text(
        json.dumps(
            {
                "status": "FORWARD_POPULATION_REPORT_COMPLETE",
                "artifacts": {"huge.png": "secret_hash_must_not_be_printed"},
            }
        )
    )
    (root / "posterior/posterior").mkdir(parents=True)
    (root / "posterior/posterior/PROGRESS.json").write_text(
        json.dumps(
            {
                "epoch": 7,
                "validation_nll": 2.612615,
                "best_nll": 2.127367,
            }
        )
    )
    (root / "MANIFEST.json").write_text(
        json.dumps({"settings": {"posterior": {"epochs": 150}}})
    )
    result = subprocess.run(
        ["bash", "scripts/watch_feniks_forward_diagnostics.sh", str(root), "--once"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "secret_hash" not in result.stdout
    assert "huge.png" not in result.stdout
    assert "TERMINE" in result.stdout
    assert "7/150" in result.stdout
    assert "2.6126" in result.stdout
    assert "2.1274" in result.stdout
    assert len(result.stdout.splitlines()) < 15


def test_compact_watcher_tolerates_incomplete_json(tmp_path):
    (tmp_path / "posterior/posterior").mkdir(parents=True)
    (tmp_path / "posterior/posterior/PROGRESS.json").write_text('{"epoch":')
    result = subprocess.run(
        [
            "bash",
            str(Path("scripts/watch_feniks_forward_diagnostics.sh")),
            str(tmp_path),
            "--once",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "EN ATTENTE" in result.stdout
