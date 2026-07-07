"""Truth/proxy labeling helpers for OpenUniverse validation tables."""

from __future__ import annotations

TRUTH = "truth"
GENERATED_TRUTH = "generated_truth"
PROXY = "proxy"
UNAVAILABLE = "unavailable"

TRUTH_LEVELS = (TRUTH, GENERATED_TRUTH, PROXY, UNAVAILABLE)


def validate_truth_level(level: str) -> str:
    """Validate a truth-level label used in reports and manifests."""
    normalized = str(level)
    if normalized not in TRUTH_LEVELS:
        raise ValueError(f"Unknown truth level {level!r}; expected {TRUTH_LEVELS}")
    return normalized
