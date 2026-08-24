from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.latent import (
    LatentSpec,
    initial_theta_from_config,
    latent_center_theta_from_config,
    latent_prior_geometry_frame,
    latent_spec_from_config,
    latent_transform_provenance,
    theta_to_x,
    x_to_theta,
    x_to_theta_log_abs_det_jacobian,
)
from euclid_dsps.config import load_config
from euclid_dsps.parameters import (
    DIFFSKY_BASIC_PARAMETER_NAMES,
    POPCOSMOS_PARAMETER_NAMES,
)


def test_latent_spec_uses_popcosmos_order() -> None:
    spec = latent_spec_from_config(load_config("configs/fs2_gpu.yaml"))

    assert spec.names == POPCOSMOS_PARAMETER_NAMES
    assert spec.lower.shape == (16,)
    assert spec.upper.shape == (16,)


def test_x_theta_roundtrip_and_bounds() -> None:
    spec = latent_spec_from_config(load_config("configs/fs2_gpu.yaml"))
    x = jnp.linspace(-2.0, 2.0, 16)

    theta = x_to_theta(x, spec)
    recovered = theta_to_x(theta, spec)

    assert theta.shape == (16,)
    assert jnp.all(theta >= spec.lower)
    assert jnp.all(theta <= spec.upper)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(x), atol=2.0e-5)


def test_latent_transform_supports_rank_two_and_three() -> None:
    spec = latent_spec_from_config(load_config("configs/fs2_gpu.yaml"))

    assert x_to_theta(jnp.zeros((4, 16)), spec).shape == (4, 16)
    assert x_to_theta(jnp.zeros((2, 4, 16)), spec).shape == (2, 4, 16)


def test_feniks_latent_schema_uses_configured_free_parameters() -> None:
    config = load_config("configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml")
    spec = latent_spec_from_config(config)

    assert config["amortized"]["latent"]["schema"] == "diffsky_dsps_closure_full"
    assert spec.names == tuple(config["fit"]["free_parameters"])
    assert spec.names == DIFFSKY_BASIC_PARAMETER_NAMES
    assert spec.lower.shape == (18,)
    assert spec.upper.shape == (18,)
    assert spec.normalization == "identity"
    assert spec.raw_center is not None
    assert spec.raw_scale is not None


def test_initial_theta_uses_config_initial_with_midpoint_fallback() -> None:
    config = load_config("configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml")
    spec = latent_spec_from_config(config)

    theta = initial_theta_from_config(
        config,
        spec.names,
        np.asarray(spec.lower),
        np.asarray(spec.upper),
    )
    by_name = dict(zip(spec.names, theta, strict=True))

    assert by_name["z_obs"] == 0.8

    fallback_config = load_config(
        "configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml"
    )
    del fallback_config["fit"]["free_parameters"]["dust_av"]["initial"]
    fallback_theta = initial_theta_from_config(
        fallback_config,
        spec.names,
        np.asarray(spec.lower),
        np.asarray(spec.upper),
    )
    fallback_by_name = dict(zip(spec.names, fallback_theta, strict=True))

    assert fallback_by_name["dust_av"] == 2.5


def test_latent_center_can_be_decoupled_from_encoder_initialization() -> None:
    config = load_config("configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml")
    config["amortized"]["latent"]["center_source"] = "midpoint"
    config["amortized"]["latent"]["centers"] = {"dust_av": 1.0}
    spec = latent_spec_from_config(config)

    init_theta = initial_theta_from_config(
        config,
        spec.names,
        np.asarray(spec.lower),
        np.asarray(spec.upper),
    )
    center_theta = latent_center_theta_from_config(
        config,
        spec.names,
        np.asarray(spec.lower),
        np.asarray(spec.upper),
    )
    by_name_init = dict(zip(spec.names, init_theta, strict=True))
    by_name_center = dict(zip(spec.names, center_theta, strict=True))

    assert by_name_init["z_obs"] == 0.8
    np.testing.assert_allclose(by_name_center["z_obs"], 0.5 * (0.001 + 5.5))
    assert by_name_center["dust_av"] == 1.0


def test_latent_prior_geometry_reports_near_bound_mass() -> None:
    config = load_config("configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml")
    frame, payload = latent_prior_geometry_frame(config, n_samples=512, seed=1)

    assert set(frame["parameter"]) == set(config["fit"]["free_parameters"])
    assert payload["normalization"] == "identity"
    assert "z_obs" in set(frame["parameter"])
    assert "frac_within_either_5pct" in frame


def test_feniks_full_schema_bounds_match_configured_training_space() -> None:
    config = load_config("configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml")
    spec = latent_spec_from_config(config)

    assert config["amortized"]["encoder"]["latent_dim"] == len(
        DIFFSKY_BASIC_PARAMETER_NAMES
    )
    by_name = {
        name: (float(lower), float(upper))
        for name, lower, upper in zip(spec.names, spec.lower, spec.upper, strict=True)
    }
    np.testing.assert_allclose(by_name["z_obs"], (0.001, 5.5))
    np.testing.assert_allclose(by_name["log10_stellar_mass"], (6.0, 13.5))
    np.testing.assert_allclose(by_name["dust_av"], (0.0, 5.0))
    np.testing.assert_allclose(by_name["dust_delta"], (-2.5, 1.0))


def test_gas_metallicity_constraint_is_satisfied() -> None:
    spec = latent_spec_from_config(load_config("configs/fs2_gpu.yaml"))
    x = jnp.zeros((5, 16)).at[:, 12].set(-10.0)

    theta = x_to_theta(x, spec)

    stellar = theta[:, POPCOSMOS_PARAMETER_NAMES.index("log10_stellar_metallicity")]
    gas = theta[:, POPCOSMOS_PARAMETER_NAMES.index("log10_gas_metallicity")]
    assert jnp.all(gas >= stellar)


def test_spline15d_mixed_transform_roundtrip() -> None:
    spec = LatentSpec(
        names=("positive", "shifted"),
        lower=jnp.asarray([0.0, -10.0]),
        upper=jnp.asarray([10.0, 10.0]),
        raw_center=jnp.asarray([0.2, -0.4]),
        raw_scale=jnp.asarray([0.7, 1.3]),
        normalization="spline15d_mixed",
        transform_family=jnp.asarray([1, 0]),
        transform_location=jnp.asarray([0.0, 2.0]),
        transform_lambda=jnp.asarray([1.0, 0.5]),
    )
    x = jnp.asarray([[-1.0, 0.5], [0.4, -1.2]], dtype=jnp.float32)

    theta = x_to_theta(x, spec)
    recovered = theta_to_x(theta, spec)

    assert jnp.all(theta[:, 0] > 0.0)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(x), atol=2.0e-6)


def test_spline15d_config_has_exact_parameter_contract() -> None:
    config = load_config("configs/amortized_feniks_spline15d_18band_gpu.yaml")
    spec = latent_spec_from_config(config)

    assert config["model"]["sfh_model"] == "spline15d"
    assert config["amortized"]["prior"]["train_jointly"] is False
    assert len(spec.names) == 15
    assert spec.names[:5] == (
        "z_obs",
        "log10_stellar_mass",
        "log10_stellar_metallicity",
        "dust_av",
        "dust_delta",
    )
    assert spec.names[-1] == "sfh_dlog_sfr_10"


def _bounded_mixed_warp_config() -> dict:
    config = load_config("configs/amortized_feniks_spline15d_18band_gpu.yaml")
    config["truth"] = {"parameter_columns": {}}
    config["amortized"]["latent"] = {
        "schema": "feniks_spline15d",
        "include_redshift": True,
        "use_fit_bounds": True,
        "normalization": "bounded_mixed_warp",
        "warps": {
            "z_obs": {"family": "asinh", "center": "fit_initial", "lambda": 0.5},
            "log10_stellar_mass": {
                "family": "asinh",
                "center": "fit_initial",
                "lambda": 1.0,
            },
            "dust_av": {"family": "log1p", "lambda": 0.2},
        },
        "raw_scales": {name: 1.0 for name in config["fit"]["free_parameters"]},
    }
    return config


def test_bounded_mixed_warp_roundtrip_and_strict_bounds() -> None:
    config = _bounded_mixed_warp_config()
    spec = latent_spec_from_config(config)
    x = jnp.linspace(-4.0, 4.0, 45, dtype=jnp.float32).reshape(3, 15)

    theta = x_to_theta(x, spec)
    recovered = theta_to_x(theta, spec)

    assert spec.normalization == "bounded_mixed_warp"
    assert jnp.all(theta > spec.lower)
    assert jnp.all(theta < spec.upper)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(x), atol=3.0e-4)


def test_bounded_mixed_warp_jacobian_matches_autodiff() -> None:
    spec = latent_spec_from_config(_bounded_mixed_warp_config())
    x = jnp.linspace(-0.7, 0.7, 15, dtype=jnp.float32)

    jacobian = jax.jacrev(lambda value: x_to_theta(value, spec))(x)
    sign, numerical = jnp.linalg.slogdet(jacobian)
    analytic = x_to_theta_log_abs_det_jacobian(x, spec)
    gradient = jax.grad(lambda value: x_to_theta_log_abs_det_jacobian(value, spec))(x)

    assert sign > 0.0
    assert jnp.isfinite(analytic)
    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_allclose(float(analytic), float(numerical), atol=2.0e-5)


def test_bounded_mixed_warp_provenance_is_no_truth_and_hashed() -> None:
    config = _bounded_mixed_warp_config()
    payload = latent_transform_provenance(config)

    assert payload["truth_used"] is False
    assert payload["truth_columns_read"] == []
    assert payload["coordinate_information_source"] == (
        "fit_bounds_fit_initials_and_config_only"
    )
    assert len(payload["transform_hash"]) == 64
    assert len(payload["warps"]) == 15
    assert {row["family"] for row in payload["warps"]} == {"asinh", "log1p"}
