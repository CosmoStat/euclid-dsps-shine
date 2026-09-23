"""Normalized p(a) p(b|a) using the existing conditional flow implementation.

The two factors have disjoint parameters: fitting narrow SFH conditionals cannot
change the physical marginal. This is a density-capacity diagnostic, not a claim
that photometry identifies either factor. All 15 coordinates remain stochastic.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .forward_population import PHYSICAL
from .posterior import ConditionalFlowEncoder, posterior_log_prob, sample_posterior
from .proposal_expressivity import (
    IndependentFlowMixture,
    independent_mixture_log_prob,
    sample_independent_mixture,
)


class StandardNormal(eqx.Module):
    def log_prob(self, x):
        return -0.5 * jnp.sum(x**2 + jnp.log(2 * jnp.pi), axis=-1)


class DensityModel(eqx.Module):
    encoder: object
    prior: StandardNormal

    def __init__(self, encoder):
        self.encoder = encoder
        self.prior = StandardNormal()


def factor_template(settings, dimension: int, context: int, seed: int):
    """Same spline and residual trunk as production, adjusted input/output sizes."""
    a = settings["architecture"]

    def expert(key):
        return ConditionalFlowEncoder(
            key,
            input_dim=context,
            latent_dim=dimension,
            hidden_sizes=tuple(a.get("hidden_sizes", [256] * 3)),
            activation=a.get("activation", "gelu"),
            log_std_min=a.get("log_std_min", -5.0),
            log_std_max=a.get("log_std_max", 3.0),
            initial_log_std=a.get("initial_log_std", 0.0),
            family=a.get("flow_family", "rq_spline"),
            n_layers=a.get("flow_layers", 12),
            hidden_size=a.get("flow_hidden_size", 256),
            n_bins=a.get("flow_bins", 16),
            tail_bound=a.get("flow_tail_bound", 12.0),
            init_scale=a.get("flow_init_scale", 0.0),
            output_space="latent_x",
            context_encoder_type="residual_photometry",
            residual_trunk_width=a.get("residual_trunk_width", 512),
            residual_blocks=a.get("residual_blocks", 3),
            residual_representation_width=a.get("residual_representation_width", 256),
            residual_context_dim=a.get("residual_context_dim", 128),
            permutation=a.get("flow_permutation", "indexed_roll"),
            transport_float64=True,
        )

    count = int(settings["experts"])
    if count < 2:
        raise ValueError("comparison retains at least two independent experts")
    keys = jax.random.split(jax.random.PRNGKey(seed), count + 1)
    experts = tuple(expert(key) for key in keys[1:])
    mixture = IndependentFlowMixture(keys[0], experts[0], n_components=count)
    return eqx.tree_at(lambda m: m.experts, mixture, experts)


def factor_log_prob(factor, context, values):
    return independent_mixture_log_prob(
        DensityModel(factor.experts[0]), factor, context, values
    )


def factor_sample(factor, context, key):
    """One independent draw for each conditioning row."""
    return sample_independent_mixture(
        DensityModel(factor.experts[0]), factor, key, context, 1
    ).x[0]


class StructuredPopulation(eqx.Module):
    physical: IndependentFlowMixture
    conditional: IndependentFlowMixture
    physical_indices: tuple[int, ...] = eqx.field(static=True)
    nuisance_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, physical, conditional, names):
        if len(names) != 15 or not set(PHYSICAL).issubset(names):
            raise ValueError("expected the canonical 15D physical + SFH coordinates")
        self.physical, self.conditional = physical, conditional
        self.physical_indices = tuple(names.index(name) for name in PHYSICAL)
        self.nuisance_indices = tuple(
            i for i in range(15) if i not in self.physical_indices
        )

    def log_prob(self, x):
        a = x[..., jnp.asarray(self.physical_indices)]
        b = x[..., jnp.asarray(self.nuisance_indices)]
        return factor_log_prob(
            self.physical, jnp.zeros((len(x), 1)), a
        ) + factor_log_prob(self.conditional, a, b)

    def sample(self, key, count):
        ka, kb = jax.random.split(key)
        a = factor_sample(self.physical, jnp.zeros((count, 1)), ka)
        b = factor_sample(self.conditional, a, kb)
        x = jnp.zeros((count, 15), dtype=a.dtype)
        return (
            x.at[:, jnp.asarray(self.physical_indices)]
            .set(a)
            .at[:, jnp.asarray(self.nuisance_indices)]
            .set(b)
        )


def transport_audit(factor, context, key, tolerance=1e-5, values=None):
    """Check trained experts using independent forward and inverse calculations.

    Mixture sampling re-evaluates log_prob, which alone is a tautological test.
    Here each expert's forward sample density is compared to its inverse density,
    and its spline logdet is checked against an autodifferentiated Jacobian.
    These checks do not constitute a global numerical integration in 15D.
    """
    from .posterior import posterior_encoder_state

    rows = []
    for expert, k in zip(
        factor.experts, jax.random.split(key, factor.n_components), strict=True
    ):
        model = DensityModel(expert)
        state = posterior_encoder_state(model, context)
        base = state.mean + jnp.exp(state.log_std) * jax.random.normal(
            k, state.mean.shape, dtype=jnp.float64
        )
        x, ld = expert.forward(base, state.flow_context)
        back, ild = expert.inverse(x, state.flow_context)
        draws = sample_posterior(model, k, context, 1)
        inverse_lp = posterior_log_prob(model, context, draws.x)
        jac = jax.jacfwd(
            lambda z, e=expert, c=state.flow_context[0]: e.forward(z, c)[0]
        )(base[0])
        _, numerical_ld = jnp.linalg.slogdet(jac)
        rows.append(
            dict(
                inverse_error=float(jnp.max(jnp.abs(back - base))),
                logdet_cancellation=float(jnp.max(jnp.abs(ld + ild))),
                forward_inverse_logprob=float(
                    jnp.max(jnp.abs(draws.logq - inverse_lp))
                ),
                autodiff_logdet=float(jnp.abs(numerical_ld - ld[0])),
            )
        )
        if values is not None:
            latent, inverse_ld = expert.inverse(values, state.flow_context)
            restored, forward_ld = expert.forward(latent, state.flow_context)
            rows[-1].update(
                data_inverse_error=float(jnp.max(jnp.abs(restored - values))),
                data_logdet_cancellation=float(
                    jnp.max(jnp.abs(inverse_ld + forward_ld))
                ),
            )
    values = np.array([list(row.values()) for row in rows])
    return dict(
        passed=bool(np.isfinite(values).all() and np.max(values) <= tolerance),
        tolerance=tolerance,
        experts=rows,
    )
