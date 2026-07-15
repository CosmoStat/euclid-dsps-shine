"""Checkpoint loading for the production spline-15D RealNVP prior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from euclid_dsps.amortized.config import require_amortized_dependencies

from .flows import RealNVPPrior, assert_flow_integrity
from .spline15d import SPLINE15D_PARAMETER_NAMES


def load_spline15d_realnvp_checkpoint(
    path: str | Path,
) -> tuple[RealNVPPrior, dict[str, Any]]:
    """Load a spline-15D RealNVP and its embedded marginal transform contract."""
    eqx, _optax = require_amortized_dependencies()
    checkpoint = Path(path)
    sidecar = json.loads(
        checkpoint.with_suffix(checkpoint.suffix + ".json").read_text(encoding="utf-8")
    )
    architecture = sidecar["architecture"]
    if architecture.get("type") != "realnvp":
        raise ValueError("Spline-15D checkpoints must contain a RealNVP")
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
    template = RealNVPPrior(
        jax.random.PRNGKey(0),
        latent_dim=int(architecture["latent_dim"]),
        n_layers=int(architecture["n_layers"]),
        hidden_size=int(architecture["hidden_size"]),
        scale_clamp=float(architecture["scale_clamp"]),
        shift_clamp=float(architecture["shift_clamp"]),
        permutation=str(architecture.get("permutation", "none")),
        init=str(architecture.get("init", "default")),
        init_scale=float(architecture.get("init_scale", 1.0)),
    )
    template = jax.tree_util.tree_map(
        lambda value: (
            value.astype(jnp.float32) if eqx.is_inexact_array(value) else value
        ),
        template,
    )
    prior = eqx.tree_deserialise_leaves(checkpoint, template)
    assert_flow_integrity(
        prior,
        context=f"spline15d RealNVP checkpoint load {checkpoint}",
        sample_count=64,
        roundtrip_fail_atol=float(sidecar.get("integrity_roundtrip_fail_atol", 1.0e-2)),
    )
    return prior, sidecar
