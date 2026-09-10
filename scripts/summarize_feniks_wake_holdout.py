"""Compare learning-batch descent with two diagnostic-only fresh batches."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.summarize_feniks_wake_descent import summarize as check_descent


def summarize(root):
    check_descent(root)
    final = json.loads((root / "FINAL.json").read_text())
    contract = final.get("holdout_contract", {})
    if contract.get("version") != "independent_fixed_mixture_v1":
        raise ValueError("not an independent-batch diagnostic")
    if contract.get("used_for_acceptance") is not False:
        raise ValueError("holdout must not choose updates")
    hashes = final["holdout_hashes"]
    if len(hashes) != 32 * 16 * 2:
        raise ValueError("expected two saved batches for all 512 attempts")
    for name, expected in hashes.items():
        with (root / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"changed holdout artifact: {name}")
    rows = []
    for path in sorted(root.glob("cases/*/wake_*/optimization.csv")):
        history = pd.read_csv(path)
        for _, row in history.iterrows():
            for replica in range(2):
                prefix = f"holdout_{replica}_"
                rows.append(
                    dict(
                        case=path.parent.parent.name,
                        start=path.parent.name,
                        attempt=int(row.attempt),
                        applied=bool(row.update_applied),
                        replica=replica,
                        training_delta=row.wake_loss_after - row.wake_loss,
                        **{
                            name: row[prefix + name]
                            for name in (
                                "loss_delta",
                                "ess",
                                "max_weight",
                                "finite",
                                "informative",
                            )
                        },
                    )
                )
    frame = pd.DataFrame(rows)
    print("\nAccepted updates, two fresh batches each:")
    print(frame[frame.applied].to_string(index=False))
    print("\nCounts across all attempts (including rejected updates):")
    print(
        frame.groupby(["applied", "finite", "informative"], dropna=False)
        .size()
        .to_string()
    )
    eligible = frame[frame.applied & frame.informative]
    print(f"\nInformative accepted-update batch evaluations: {len(eligible)}")
    print(f"Loss decreases: {int((eligible.loss_delta < 0).sum())}")
    print(f"Loss increases: {int((eligible.loss_delta > 0).sum())}")
    print(
        "Two batches of the same update are not independent galaxies. Low-support batches are inconclusive, not a validation PASS. No selection or promotion."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
