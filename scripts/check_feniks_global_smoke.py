"""Fail closed unless the smoke exercised actual wake and population updates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def check(root):
    receipt = json.loads((root / "training_receipt.json").read_text())
    history = pd.read_csv(root / "sc_drws_training_log.csv")
    wake = history[history.update_kind.eq("wake")]
    if receipt.get("wake_updates", 0) <= 0 or receipt.get("prior_updates", 0) <= 0:
        raise ValueError("smoke must apply wake and prior updates")
    for phase in ("robust_warmup", "joint_gaussian"):
        rows = wake[wake.phase.eq(phase)]
        applied = rows[rows.q_update_applied.eq(True)]
        if applied.empty or not (applied.flow_gradient_multiplier > 0).any():
            raise ValueError(f"no applied trainable-flow wake in {phase}")
        if not np.isfinite(applied[["loss", "q_objective_after"]].to_numpy()).all():
            raise ValueError("nonfinite accepted wake objective")
        if not (applied.q_objective_after < applied.loss).all():
            raise ValueError("accepted wake does not decrease its objective")
    return {
        "status": "PASS",
        "scientific_promotion": False,
        "meaning": "execution coverage only, not posterior qualification",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    root = p.parse_args().root
    result = check(root)
    (root / "SMOKE_GATE.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
