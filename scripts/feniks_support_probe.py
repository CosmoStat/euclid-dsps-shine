"""Pinned checkpoint probes; no optimization and no posterior promotion."""

from pathlib import Path

import numpy as np

from euclid_dsps.amortized.population_vem import sha256_file
from scripts.run_feniks_sc_drws_balanced_npe import read


def pin_source(root, manifest):
    root = Path(root).resolve()
    old = read(root / "RUN_MANIFEST.json")
    final = read(root / "FINAL.json")
    if (
        old.get("method") != "qualified_controlled_local_vi_v1"
        or final.get("status") != "DIAGNOSTIC_COMPLETE"
        or final.get("prior_bitwise_unchanged") is not True
        or final.get("truth_used") is not False
        or final.get("scientific_promotion") is not False
        or final.get("cases_complete") != 2 * old["objects_per_group"]
    ):
        raise ValueError("completed frozen controlled source required")
    for key in (
        "qualified_night",
        "config_sha256",
        "rows_sha256",
        "seed",
        "objects_per_group",
        "dataset",
    ):
        if old[key] != manifest[key]:
            raise ValueError(f"probe/source mismatch: {key}")
    for name, key in (
        ("config.yaml", "config_sha256"),
        ("observed_rows.npy", "rows_sha256"),
    ):
        if sha256_file(root / name) != old[key]:
            raise ValueError(f"source artifact changed: {name}")
    names = [
        "RUN_MANIFEST.json",
        "FINAL.json",
        "SIMULATED_INPUTS.npz",
        "observed_rows.npy",
        "config.yaml",
    ]
    receipt = final["artifacts"]["SIMULATED_INPUTS.npz"]
    if sha256_file(root / "SIMULATED_INPUTS.npz") != receipt["sha256"]:
        raise ValueError("source simulated inputs changed")
    for group in ("observed", "simulated"):
        for i in range(old["objects_per_group"]):
            folder = f"cases/{group}_{i:03d}"
            record = read(root / folder / "COMPLETE.json")
            if record.get("prior_bitwise_unchanged") is not True:
                raise ValueError("source case not frozen")
            names.append(folder + "/COMPLETE.json")
            for start in (4, 5):
                name = f"{folder}/start_{start}/parameters.eqx"
                if (
                    sha256_file(root / name)
                    != record["local_starts"][start]["checkpoint_sha256"]
                ):
                    raise ValueError("source checkpoint changed")
                names.extend([name, f"{folder}/start_{start}/REGIME.json"])
                regime = read(root / folder / f"start_{start}/REGIME.json")
                if regime != dict(
                    name="slow_mc16",
                    learning_rate=0.0001,
                    gradient_draws=16,
                    initialization=start - 4,
                    scientific_promotion=False,
                ):
                    raise ValueError("expected prespecified slow_mc16 source")
    return dict(
        path=str(root),
        hashes={name: sha256_file(root / name) for name in names},
        contract="Hashes pinned at preparation; checkpoints checked against case receipts. No original complete-run inventory available.",
    )


def verify_source(reference):
    root = Path(reference["path"])
    for name, digest in reference["hashes"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"probe source changed: {name}")


def check_simulations(source, current):
    with (
        np.load(source, allow_pickle=False) as old,
        np.load(current, allow_pickle=False) as new,
    ):
        if set(old.files) != set(new.files) or any(
            not np.array_equal(old[k], new[k]) for k in old.files
        ):
            raise ValueError(
                "simulated contexts changed; do not reuse local checkpoints"
            )
