from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.jacobian_lens import (
    autoencoder_jacobian_lens,
    decoder_jacobian_lens,
    lens_tables_for_object,
)
from euclid_dsps.amortized.latent import LatentSpec


def _latent_spec(dim: int) -> LatentSpec:
    return LatentSpec(
        names=tuple(f"theta_{index}" for index in range(dim)),
        lower=jnp.full((dim,), -5.0),
        upper=jnp.full((dim,), 5.0),
        raw_center=jnp.zeros((dim,)),
        raw_scale=jnp.ones((dim,)),
        normalization="identity",
    )


def test_decoder_jacobian_reports_exact_nullity_when_latent_exceeds_bands() -> None:
    weights = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    def decoder(x):
        return x @ weights

    result = decoder_jacobian_lens(
        decoder,
        jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=jnp.float32),
        _latent_spec(5),
        obs_flux=jnp.zeros((3,), dtype=jnp.float32),
        obs_err=jnp.ones((3,), dtype=jnp.float32),
        mask=jnp.ones((3,), dtype=bool),
        likelihood_type="gaussian",
    )

    assert result["j_flux_x"].shape == (3, 5)
    assert result["vt_full"].shape == (5, 5)
    assert result["singular_values"].shape == (3,)

    tables = lens_tables_for_object(
        object_id=1,
        row_index=7,
        decoder_flux_from_x=decoder,
        x0=jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=jnp.float32),
        latent_spec=_latent_spec(5),
        obs_flux=jnp.zeros((3,), dtype=jnp.float32),
        obs_err=jnp.ones((3,), dtype=jnp.float32),
        mask=jnp.ones((3,), dtype=bool),
        band_names=("b1", "b2", "b3"),
        likelihood_config={"type": "gaussian"},
        direction_top_k=2,
    )

    summary = tables.object_summary.iloc[0]
    assert summary["exact_nullity"] == 2
    assert summary["effective_rank_1e_3"] == 3
    assert set(tables.singular_values["direction_kind"]) >= {"exact_null"}
    assert len(tables.physical_direction_loadings) > 0


def test_autoencoder_lens_detects_identity_copy_like_modes() -> None:
    def autoencoder(flux):
        return flux

    result = autoencoder_jacobian_lens(
        autoencoder,
        obs_flux=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        obs_err=jnp.ones((3,), dtype=jnp.float32) * 0.1,
        mask=jnp.ones((3,), dtype=bool),
    )

    singular = np.asarray(result["singular_values"])
    assert singular.shape == (3,)
    assert np.allclose(singular, np.ones(3), atol=1.0e-5)
