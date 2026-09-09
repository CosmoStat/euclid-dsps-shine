import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_script_runs_without_checkout_imports(tmp_path):
    root = tmp_path / "source"
    folder = root / "cases/observed_000/amortized"
    folder.mkdir(parents=True)
    bank = folder / "direct_draws.npz"
    np.savez(
        bank,
        x=np.arange(8.0).reshape(4, 1, 2),
        logq=np.zeros((4, 1)),
        logprior=np.zeros((4, 1)),
        loglike=np.zeros((4, 1)),
    )
    digest = hashlib.sha256(bank.read_bytes()).hexdigest()
    (folder / "SUMMARY.json").write_text(json.dumps({"direct_draws_sha256": digest}))
    script = Path(__file__).resolve().parents[1] / "scripts/audit_feniks_saved_draws.py"
    out = tmp_path / "audit"
    result = subprocess.run(
        [sys.executable, "-I", str(script), str(root), "--out", str(out)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "AUDIT.json").read_text())
    assert report["distributions"] == 1
    assert report["decoder_calls"] == 0
    assert report["scientific_promotion"] is False
