"""Checkpoint loading for production spline-15D exact flow priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from euclid_dsps.amortized.config import require_amortized_dependencies

from .flows import RealNVPPrior, assert_flow_integrity
from .spline15d import SPLINE15D_PARAMETER_NAMES
from .train import PriorFlow, _prior_template_from_architecture


def load_spline15d_flow_checkpoint(
    path: str | Path,
) -> tuple[PriorFlow, dict[str, Any]]:
    """Load a spline-15D exact flow and embedded marginal transform contract."""
    eqx, _optax = require_amortized_dependencies()
    checkpoint = Path(path)
    sidecar = json.loads(
        checkpoint.with_suffix(checkpoint.suffix + ".json").read_text(encoding="utf-8")
    )
    architecture = sidecar["architecture"]
    if architecture.get("type") not in {"realnvp", "rq_spline_coupling"}:
        raise ValueError(
            "Spline-15D checkpoints must contain RealNVP or RQ-spline coupling"
        )
    if int(architecture.get("latent_dim", -1)) != len(SPLINE15D_PARAMETER_NAMES):
        raise ValueError("Spline-15D checkpoint has the wrong latent dimension")
    if tuple(sidecar.get("parameter_names", ())) != SPLINE15D_PARAMETER_NAMES:
        raise ValueError("Spline-15D checkpoint has the wrong parameter order")
    normalization_family = sidecar.get("normalization", {}).get("family")
    if normalization_family not in {
        "asinh",
        "shifted_asinh",
        "mixed_log_shifted_asinh",
    }:
        raise ValueError(
            "Spline-15D checkpoint is missing a supported marginal transform"
        )
    template = _prior_template_from_architecture(architecture)
    template = jax.tree_util.tree_map(
        lambda value: (
            value.astype(jnp.float32) if eqx.is_inexact_array(value) else value
        ),
        template,
    )
    prior = eqx.tree_deserialise_leaves(checkpoint, template)
    assert_flow_integrity(
        prior,
        context=f"spline15d exact-flow checkpoint load {checkpoint}",
        sample_count=64,
        roundtrip_fail_atol=float(sidecar.get("integrity_roundtrip_fail_atol", 1.0e-2)),
    )
    return prior, sidecar


def load_spline15d_realnvp_checkpoint(
    path: str | Path,
) -> tuple[RealNVPPrior, dict[str, Any]]:
    """Load a legacy spline-15D RealNVP checkpoint."""
    prior, sidecar = load_spline15d_flow_checkpoint(path)
    if not isinstance(prior, RealNVPPrior):
        raise ValueError("Spline-15D checkpoint does not contain a RealNVP")
    return prior, sidecar
