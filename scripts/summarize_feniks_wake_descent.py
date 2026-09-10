"""Check descent invariants and show paired final wake support."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(root):
    def read(p):
        return json.loads(p.read_text())

    manifest = read(root / "RUN_MANIFEST.json")
    if manifest.get("adaptation_contract") != "wake_armijo_v1":
        raise ValueError("expected wake_armijo_v1")
    final = read(root / "FINAL.json")
    if final["status"] != "OBJECTIVE_PILOT_COMPLETE" or final["cases_complete"] != 16:
        raise ValueError("pilot not complete; inspect FINAL.json")
    rows = []
    for path in sorted(root.glob("cases/*/wake_*/optimization.csv")):
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if final["wake_history_hashes"].get(str(path.relative_to(root))) != actual:
            raise ValueError(f"wake history changed: {path}")
        history = pd.read_csv(path)
        accepted = history[history.update_applied.eq(True)]
        armijo = manifest["wake_backtracking"]["armijo"]
        valid = (accepted.wake_loss_after < accepted.wake_loss) & (
            accepted.wake_loss_after
            <= accepted.wake_loss
            + armijo * accepted.accepted_scale * accepted.directional_derivative
        )
        if not valid.all():
            raise ValueError(f"non-descending applied update: {path}")
        after = read(path.parent / "draws_04096/SUMMARY.json")
        start = path.parent.name.split("_")[-1]
        source = read(path.parent.parent / f"source_{start}/SUMMARY.json")
        rows.append(
            dict(
                case=path.parent.parent.name,
                start=start,
                accepted=len(accepted),
                attempts=len(history),
                median_scale=accepted.accepted_scale.median()
                if len(accepted)
                else np.nan,
                ess_ratio=after["raw_ess"]["fraction_median"]
                / source["raw_ess"]["fraction_median"],
                rms_before=source["residual_rms"],
                rms_after=after["residual_rms"],
            )
        )
    if len(rows) != 32:
        raise ValueError("expected 32 wake trajectories")
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "All applied wake updates satisfy finite-batch descent. Independent metrics remain descriptive."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
