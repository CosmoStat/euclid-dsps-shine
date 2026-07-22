#!/usr/bin/env python3
"""Fail-fast contract for the three learned-prior Student-t2 candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from validate_feniks_selfsup_rws_inputs import validate_selfsup_rws_inputs

from euclid_dsps.config import load_config


def validate_production_inputs(
    catalog_dir: Path,
    reference_checkpoint: Path,
    config_paths: list[Path],
) -> None:
    validate_selfsup_rws_inputs(catalog_dir, reference_checkpoint, config_paths)
    expected = (
        (1, "importance", 8),
        (2, "importance", 8),
        (2, "smc", 4),
    )
    if len(config_paths) != len(expected):
        raise ValueError("Production array requires exactly three configs")
    for path, (components, sampler, particles) in zip(
        config_paths, expected, strict=True
    ):
        config = load_config(path)
        amortized = dict(config.get("amortized", {}) or {})
        encoder = dict(amortized.get("encoder", {}) or {})
        prior = dict(amortized.get("prior", {}) or {})
        likelihood = dict(amortized.get("likelihood", {}) or {})
        wake = dict(((amortized.get("objective", {}) or {}).get("wake", {})) or {})
        actual = (
            int(encoder.get("base_components", 1)),
            str(wake.get("sampler", "importance")),
            int(wake.get("n_particles", 0)),
        )
        if actual != (components, sampler, particles):
            raise ValueError(
                f"{path}: expected posterior/wake {(components, sampler, particles)}, "
                f"got {actual}"
            )
        if str(prior.get("source")) != "joint_realnvp":
            raise ValueError(f"{path}: production candidates must learn the prior")
        if (
            str(likelihood.get("type")) != "student_t"
            or float(likelihood.get("student_t_dof", -1.0)) != 2.0
        ):
            raise ValueError(f"{path}: production likelihood must be Student-t2")
    print("[selfsup-production-contract] valid: exact three-candidate matrix")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    args = parser.parse_args()
    validate_production_inputs(
        args.catalog_dir,
        args.reference_checkpoint,
        args.config,
    )


if __name__ == "__main__":
    main()
