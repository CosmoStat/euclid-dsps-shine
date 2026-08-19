from __future__ import annotations

import subprocess
from pathlib import Path


def test_proposal_expressivity_launchers_preserve_comma_separated_seed_env() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts/submit_popcosmos_proposal_expressivity.sh"
    recovery = root / "scripts/recover_popcosmos_proposal_expressivity.sh"
    for path in (launcher, recovery):
        text = path.read_text(encoding="utf-8")
        assert 'SMC_SEEDS_CSV="260817,260818"' in text
        assert "export SMC_VARIANTS_CSV SMC_SEEDS_CSV" in text
        assert 'SMC_SEEDS_CSV="$SMC_SEEDS_CSV"' not in text
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_recovery_reuses_completed_shards_and_recreates_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/recover_popcosmos_proposal_expressivity.sh").read_text(
        encoding="utf-8"
    )
    assert "if not done.is_file()" in text
    assert 'path.exists() and not (path / "DONE").is_file()' in text
    assert '--dependency="afterok:${SMC_DIAGNOSTIC_JOB}"' in text
    assert '--dependency="afterok:${SMC_DIAGNOSTIC_FINALIZER_JOB}"' in text
