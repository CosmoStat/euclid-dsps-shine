"""Diffsky/FENIKS proposal generation and DSPS closure utilities."""

from __future__ import annotations

from .config import SyntheticDiffskyConfig, load_synthetic_diffsky_config

__all__ = [
    "SyntheticDiffskyConfig",
    "generate_dsps_closure_dataset",
    "load_synthetic_diffsky_config",
    "validate_dsps_closure_dataset",
]


def __getattr__(name: str):
    if name == "generate_dsps_closure_dataset":
        from .generation import generate_dsps_closure_dataset

        return generate_dsps_closure_dataset
    if name == "validate_dsps_closure_dataset":
        from .validation import validate_dsps_closure_dataset

        return validate_dsps_closure_dataset
    raise AttributeError(name)
