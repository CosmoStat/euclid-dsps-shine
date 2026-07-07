from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.features import (
    compute_feature_stats,
    feature_stats_from_json,
    feature_stats_to_json,
    make_encoder_features,
)


def test_encoder_features_shape_and_negative_flux() -> None:
    flux = np.asarray([[1.0, -2.0] * 5, [3.0, -4.0] * 5], dtype=float)
    err = np.ones_like(flux) * 0.1
    stats = compute_feature_stats(flux, err)

    features = make_encoder_features(jnp.asarray(flux), jnp.asarray(err), stats)

    assert features.shape == (2, 20)
    assert jnp.all(jnp.isfinite(features))


def test_feature_stats_json_roundtrip() -> None:
    flux = np.ones((3, 10))
    err = np.ones((3, 10)) * 0.2
    stats = compute_feature_stats(
        flux, err, band_names=tuple(f"b{i}" for i in range(10))
    )

    payload = json.loads(json.dumps(feature_stats_to_json(stats)))
    loaded = feature_stats_from_json(payload)

    np.testing.assert_allclose(loaded.flux_scale, stats.flux_scale)
    np.testing.assert_allclose(loaded.err_scale, stats.err_scale)
    assert loaded.band_names == stats.band_names
    assert loaded.flux_transform == "asinh"


def test_feature_stats_use_mask() -> None:
    flux = np.asarray([[100.0, 1.0], [2.0, 2.0]], dtype=float)
    err = np.ones_like(flux)
    mask = np.asarray([[False, True], [True, True]])

    stats = compute_feature_stats(flux, err, mask=mask)

    assert stats.flux_scale[0] == 2.0


def test_asinh_flux_features_compress_bright_fluxes() -> None:
    flux = np.asarray([[1.0e-30] * 10, [300.0e-30] * 10], dtype=float)
    err = np.ones_like(flux) * 1.0e-31
    stats = compute_feature_stats(flux[:1], err[:1])

    features = np.asarray(make_encoder_features(jnp.asarray(flux), jnp.asarray(err), stats))

    assert stats.flux_transform == "asinh"
    assert features[1, 0] < 7.0
    assert features[1, 0] > features[0, 0]


def test_legacy_feature_stats_json_defaults_to_linear_transform() -> None:
    payload = {
        "flux_scale": [1.0] * 10,
        "err_scale": [1.0] * 10,
        "band_names": [f"b{i}" for i in range(10)],
    }

    loaded = feature_stats_from_json(payload)

    assert loaded.flux_transform == "linear"
