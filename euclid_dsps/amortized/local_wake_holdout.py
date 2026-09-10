"""Read-only fixed-mixture batch comparison; no optimizer or acceptance logic."""

import jax
import jax.numpy as jnp

from .local_wake_diagnostic import wake_loss


def compare_batch(encoder, before, after, context, x, logweights):
    """Keep pre-update proposal weights fixed when scoring both parameters."""
    weights = jax.nn.softmax(logweights, axis=0)
    ess = jnp.min(1 / jnp.sum(weights**2, axis=0))
    maximum = jnp.max(weights)
    old = wake_loss(encoder, before, context, x, logweights)
    new = wake_loss(encoder, after, context, x, logweights)
    finite = (
        jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(logweights))
        & jnp.isfinite(old)
        & jnp.isfinite(new)
        & jnp.isfinite(ess)
    )
    return dict(
        loss_before=old,
        loss_after=new,
        loss_delta=new - old,
        ess=ess,
        max_weight=maximum,
        finite=finite,
        informative=finite & (ess >= 16) & (maximum <= 0.2),
    )
