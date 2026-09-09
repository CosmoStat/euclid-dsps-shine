"""Paired final-only readback, without selection or decoder calls."""

import argparse
import json
from pathlib import Path

from summarize_feniks_sc_drws_controlled_local_vi import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "RUN_MANIFEST.json").read_text())
    if manifest["method"] != "qualified_long_replay_v1":
        raise ValueError("expected long replay root")
    source = Path(manifest["support_probe"]["path"])
    old = collect(source)
    old = old[(old.step == 512) | (old.regime == "amortized")]
    new = collect(args.root)
    if new.empty:
        print("No completed replay evaluations")
        return
    metrics = [
        "ess_fraction",
        "max_weight",
        "bad_k",
        "evidence_delta",
        "negative_elbo",
        "residual_rms",
    ]
    paired = old[["case", "start", *metrics]].merge(
        new[["case", "start", *metrics]],
        on=["case", "start"],
        how="outer",
        validate="one_to_one",
        suffixes=("_K1024", "_K4096"),
        indicator=True,
    )
    print(paired.to_string(index=False))
    print(
        "Final checkpoints only; different K and independent draws. No selection or promotion."
    )


if __name__ == "__main__":
    main()
