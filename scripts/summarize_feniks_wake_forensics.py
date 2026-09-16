"""CPU-only readback of fixed wake counterfactuals, without selection."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def summarize(root):
    final = json.loads((root / "FINAL.json").read_text())
    print("Status:", final["status"])
    if final["status"] not in ("WAKE_FORENSICS_COMPLETE", "WAKE_REPLAY_MISMATCH"):
        return
    for name, expected in final.get("artifacts", {}).items():
        with (root / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"artifact changed: {name}")
    report = json.loads((root / "WAKE_FORENSICS.json").read_text())
    print(pd.DataFrame(report["replays"]).to_string(index=False))
    rows = []
    for path in sorted(root.glob("cases/*/wake_forensic_*/UPDATE_AUDIT.json")):
        audit = json.loads(path.read_text())
        print(
            path.parent.relative_to(root),
            "loss:",
            audit["before_loss"],
            "->",
            audit["after_loss"],
        )
        print(pd.DataFrame(audit["directional_stencils"]).to_string(index=False))
        for scale, summary in audit["evaluations"].items():
            rows.append(
                dict(
                    distribution=str(path.parent.relative_to(root)),
                    scale=scale,
                    **{k: summary.get(k) for k in ("negative_elbo", "residual_rms")},
                    ess_fraction=summary["raw_ess"]["fraction_median"],
                    max_weight=summary["maximum_raw_weight"]["median"],
                )
            )
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "Fixed scales; no selection. A replay mismatch blocks causal interpretation. No promotion."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
