"""Upgrade ONLY a not-yet-built coherent reference from bounded Av to log Av."""

from __future__ import annotations

import argparse
import copy
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

from scripts.feniks_avi_experiments import sha, write
from scripts.feniks_coherent_inference import settings


def check(root: Path) -> dict:
    manifest, cfg, _ = settings(root)
    r = cfg["reference"]
    if r.get("positive") or r["bounds"].get("dust_av") != [0.000001, 6.0]:
        raise ValueError("Repair requires the original bounded-dust reference config")
    if (root / "REFERENCE_REPAIR.json").exists():
        raise ValueError(
            "A reference repair already exists; inspect it before resuming"
        )
    if (root / "reference/FINAL.json").exists() or (
        root / "report/FINAL.json"
    ).exists():
        raise ValueError("Cannot change coordinates after a completed stage")
    for name in ("banks", "oracle", "population", "posterior"):
        if any(p.is_file() for p in (root / name).rglob("*")):
            raise ValueError(
                f"Cannot mix new coordinates with existing {name} artifacts"
            )
    for name in ("CODE_DIR", "CODE_SHA256"):
        if not (root / name).is_file():
            raise ValueError(f"Missing original code provenance: {name}")
    return manifest


def repair(root: Path, code: Path, archive: Path) -> None:
    """The shell caller verifies that all prior SLURM jobs have stopped first."""
    manifest = check(root)
    if not (code / "scripts/feniks_coherent_inference.py").is_file():
        raise ValueError("Missing newly frozen pipeline")
    digest = sha(archive)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    backup = root / "recovery" / f"reference_support_{timestamp}"
    backup.mkdir(parents=True)
    originals = [
        root / n
        for n in ("MANIFEST.json", "experiment.yaml", "CODE_DIR", "CODE_SHA256")
    ]
    originals += sorted(root.glob("JOBS*.env"))
    if (root / "ROADMAP_STATUS.md").is_file():
        originals.append(root / "ROADMAP_STATUS.md")
    original_hashes = {p.name: sha(p) for p in originals}
    for path in originals:
        shutil.copy2(path, backup / path.name)
    receipt = dict(
        status="PREPARING",
        reason="Replace artificial open dust_av (1e-6,6) bound by positive log coordinate",
        old_manifest_sha256=sha(root / "MANIFEST.json"),
        old_code_dir=(root / "CODE_DIR").read_text().strip(),
        old_code_sha256=(root / "CODE_SHA256").read_text().strip(),
        new_code_dir=str(code),
        new_code_sha256=digest,
        new_code_archive=str(archive),
        backup=str(backup),
        original_hashes=original_hashes,
        dataset_modified=False,
        completed_training_reused=False,
    )
    write(root / "REFERENCE_REPAIR.json", receipt)
    # Archive partial reference/blocked report, never overwrite their provenance.
    for name in ("reference", "report"):
        if (root / name).exists():
            shutil.move(root / name, backup / name)
    updated = copy.deepcopy(manifest)
    updated["settings"]["reference"]["positive"] = ["dust_av"]
    del updated["settings"]["reference"]["bounds"]["dust_av"]
    config = root / "experiment.yaml"
    tmp = config.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(updated["settings"], sort_keys=False))
    tmp.replace(config)
    updated["frozen_files"]["experiment.yaml"] = sha(config)
    updated["reference_repair"] = dict(
        original_manifest=receipt["old_manifest_sha256"], backup=str(backup)
    )
    write(root / "MANIFEST.json", updated)
    for name, value in (("CODE_DIR", str(code)), ("CODE_SHA256", digest)):
        path = root / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(value + "\n")
        tmp.replace(path)
    receipt.update(status="COMPLETE", new_manifest_sha256=sha(root / "MANIFEST.json"))
    write(root / "REFERENCE_REPAIR.json", receipt)
    print(f"Reference support repaired; archived original state: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--code", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.check:
        check(args.root.resolve())
    else:
        if args.code is None or args.archive is None:
            parser.error("--code and --archive required for repair")
        repair(args.root.resolve(), args.code.resolve(), args.archive.resolve())


if __name__ == "__main__":
    main()
