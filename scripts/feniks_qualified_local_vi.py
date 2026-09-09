"""Receipt-linked local VI follow-up of the precision night, never promotion."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import yaml

from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.config import load_config
from scripts.feniks_precision_night import require_qualification
from scripts.run_feniks_sc_drws_balanced_npe import read, write


def verify_night(root: Path, arm: str = "C") -> dict:
    root = Path(root).resolve()
    if arm not in {"B", "C"}:
        raise ValueError("local follow-up requires the prespecified B or C checkpoint")
    inventory = read(root / "NIGHT_ARTIFACTS.json")

    def checked(name):
        path = root / name
        if name not in inventory or sha256_file(path) != inventory[name]:
            raise ValueError(f"night artifact changed or unrecorded: {name}")
        return path

    final = read(checked("NIGHT_FINAL.json"))
    if (
        final.get("status") != "PRECISION_NIGHT_DIAGNOSTIC_COMPLETE"
        or final.get("truth_used") is not False
        or final.get("population_training_started") is not False
        or final.get("scientific_promotion") is not False
    ):
        raise ValueError("completed truth-free frozen-parent night required")
    require_qualification(read(checked("FULL_DECODER_QUALIFICATION.json")))
    manifest = read(checked("RUN_MANIFEST.json"))
    migration = read(checked("MIGRATION.json"))
    if migration.get("status") != "PASS":
        raise ValueError("checkpoint migration was not certified")
    receipt = read(checked(f"arms/{arm}/ARM_COMPLETE.json"))
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("prior_bitwise_unchanged") is not True
        or receipt.get("truth_used") is not False
    ):
        raise ValueError("source arm is not certified frozen and truth-free")
    checkpoint = checked(f"arms/{arm}/train/checkpoints/best.eqx")
    sidecar = checked(f"arms/{arm}/train/checkpoints/best.eqx.json")
    config = checked(f"configs/{arm}.yaml")
    stats = checked(f"arms/{arm}/train/feature_stats.json")
    if (
        sha256_file(checkpoint) != receipt["checkpoint_sha256"]
        or sha256_file(sidecar) != receipt["checkpoint_sidecar_sha256"]
        or sha256_file(stats) != receipt["feature_stats_sha256"]
    ):
        raise ValueError("arm receipt disagrees with inventory")
    checked("cohorts/tracking.npy")
    checked("cohorts/train.npy")
    checked("cache/precision64_sleep.npz.json")
    return dict(
        path=str(root),
        arm=arm,
        inventory_sha256=sha256_file(root / "NIGHT_ARTIFACTS.json"),
        manifest=manifest,
        frozen_array_sha256=migration["frozen_array_sha256"],
        source={
            "status": "COMPLETE",
            "prior_bitwise_unchanged": True,
            "truth_used_for_training_or_checkpoint_selection": False,
            **{
                key: str(value)
                for key, value in (
                    ("checkpoint", checkpoint),
                    ("config", config),
                    ("feature_stats", stats),
                    ("checkpoint_sidecar", sidecar),
                )
            },
            **{
                key + "_sha256": sha256_file(value)
                for key, value in (
                    ("checkpoint", checkpoint),
                    ("config", config),
                    ("feature_stats", stats),
                    ("checkpoint_sidecar", sidecar),
                )
            },
        },
    )


def prepare(
    root,
    source_root,
    night_root,
    objects=8,
    steps=64,
    draws=128,
    arm="C",
    controlled=False,
    support_probe_root=None,
):
    # Import the existing observed-only loader, not a new catalogue read path.
    from euclid_dsps.amortized.latent import latent_spec_from_config, latent_spec_hash
    from euclid_dsps.model import photometry_numerics
    from scripts.run_feniks_sc_drws_local_vi_diagnostic import (
        _representative_observed_rows,
        load_photometry_arrays_from_config,
        observed_only_config,
        required_catalog_columns,
    )

    root, night_root = Path(root).resolve(), Path(night_root).resolve()
    if root.exists():
        raise FileExistsError(f"preserve existing diagnostic: {root}")
    if not (2 <= objects <= 8 and 1 <= steps <= 64 and 32 <= draws <= 128):
        raise ValueError(
            "bounded follow-up: 2..8 cases/group, 1..64 steps, 32..128 draws"
        )
    reference = verify_night(night_root, arm)
    old = reference["manifest"]
    if Path(old["source_root"]).resolve() != Path(source_root).resolve():
        raise ValueError("night and balanced environment have different source roots")
    config = observed_only_config(load_config(reference["source"]["config"]))
    a = config["amortized"]
    if (
        config["model"].get("spline_precision") != "float64_v1"
        or a["latent"].get("arithmetic_precision") != "float64_v1"
        or a["likelihood"].get("arithmetic_precision") != "float64_v1"
        or a["prior"].get("train_jointly") is not False
        or a.get("population_vi", {}).get("enabled") is not False
    ):
        raise ValueError("qualified numerical/frozen target contract changed")
    config["catalog_path"] = old["dataset"]["path"]
    if sha256_file(Path(config["catalog_path"])) != old["dataset"]["sha256"]:
        raise ValueError("source dataset changed")
    pool = np.load(night_root / "cohorts/tracking.npy", allow_pickle=False)
    train = np.load(night_root / "cohorts/train.npy", allow_pickle=False)
    if len(np.unique(pool)) != len(pool) or np.intersect1d(pool, train).size:
        raise ValueError("duplicate tracking identities or training overlap")
    rows, audit = _representative_observed_rows(config, pool, count=objects)
    if len(np.unique(rows)) != objects or not np.isin(rows, pool).all():
        raise ValueError("invalid observed-only cohort")
    arrays = load_photometry_arrays_from_config(
        config, batch_size=objects, row_indices=rows
    )
    if not np.all(arrays.mask) or not np.all(
        np.isfinite(arrays.flux_err) & (arrays.flux_err > 0)
    ):
        raise ValueError("incomplete context: stop, never silently filter cases")
    bank = night_root / "cache/precision64_sleep.npz"
    sidecar = bank.with_suffix(".npz.json")
    metadata = read(sidecar)
    if (
        metadata.get("photometry_numerics") != photometry_numerics(config["model"])
        or metadata.get("latent_spec_hash")
        != latent_spec_hash(latent_spec_from_config(config))
        or metadata.get("stored_flux_dtype") != "float64"
        or metadata.get("catalogue_truth_used") is not False
        or metadata.get("generator") != "direct_frozen_parent"
    ):
        raise ValueError("cache does not describe the qualified simulator")
    cache = dict(
        path=str(bank), sha256=sha256_file(bank), sidecar_sha256=sha256_file(sidecar)
    )
    root.mkdir(parents=True)
    np.save(root / "observed_rows.npy", rows, allow_pickle=False)
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = dict(
        method="qualified_fixed_parent_same_family_local_vi_v1",
        mode="local_vi",
        status="PREPARED",
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_root=str(night_root),
        source=reference["source"],
        dataset=old["dataset"],
        qualified_night={
            k: reference[k]
            for k in ("path", "arm", "inventory_sha256", "frozen_array_sha256")
        },
        source_manifest_sha256=sha256_file(night_root / "RUN_MANIFEST.json"),
        cache=cache,
        config_sha256=sha256_file(root / "config.yaml"),
        rows_sha256=sha256_file(root / "observed_rows.npy"),
        cohort=audit,
        catalogue_columns=required_catalog_columns(config),
        objects_per_group=objects,
        steps=steps,
        evaluation_draws=draws,
        starts=2,
        gradient_draws=4,
        learning_rate=0.001,
        seed=260910,
        seconds=9900,
        maximum_decoder_evaluations=32000,
        maximum_gpus=1,
        maximum_nodes=1,
        allocation_gpu_hours=3,
        truth_used=False,
        scientific_promotion=False,
        population_training_started=False,
        interpretation="Prespecified final iterates, no best-start selection. Reused development cohort, not an independent test. Local reverse KL can miss modes; no teacher or population authorization.",
    )
    if controlled:
        manifest.update(
            method="qualified_controlled_local_vi_v1",
            optimization_regimes=[
                dict(name="original", learning_rate=0.001, gradient_draws=4),
                dict(name="slow", learning_rate=0.0001, gradient_draws=4),
                dict(name="slow_mc16", learning_rate=0.0001, gradient_draws=16),
            ],
            trajectory_steps=[s for s in (8, 16, 32) if s < steps],
            maximum_decoder_evaluations=180000,
            interpretation="Paired development experiment, not checkpoint selection. Three regimes, two identical initializations each; fresh evaluation draws shared across regimes and steps, never used by the optimizer. No scientific promotion.",
        )
    if support_probe_root is not None:
        from scripts.feniks_support_probe import pin_source

        if controlled or objects != 8 or steps != 64 or draws != 128:
            raise ValueError("support probe requires fixed 8/64/128 recipe")
        reference = pin_source(support_probe_root, manifest)
        manifest.update(
            method="qualified_support_probe_v1",
            support_probe=reference,
            steps=0,
            source_checkpoint_step=64,
            optimization_started=False,
            optimization_regimes=[
                dict(name="local_x1", factor=1.0, mixture=False),
                dict(name="local_x1p5", factor=1.5, mixture=False),
                dict(name="local_x2", factor=2.0, mixture=False),
                dict(name="mixture_x1p5", factor=1.5, mixture=True),
            ],
            maximum_decoder_evaluations=50000,
            interpretation="Fixed final slow_mc16 checkpoints, two starts, four direct-draw proposals. No optimization, winner selection, or scientific promotion.",
        )
    write(root / "RUN_MANIFEST.json", manifest)
    return manifest
