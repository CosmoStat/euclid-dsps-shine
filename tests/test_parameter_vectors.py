from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.observation_arrays import (
    photometry_arrays_from_dataframe,
    validate_fs2_band_contract,
)
from euclid_dsps.parameter_vectors import (
    model_mags_from_theta_matrix_jax,
    theta_vector_to_model_param_dict,
    theta_vector_to_param_dict,
)
from euclid_dsps.photometry import abmag_to_fnu_cgs


def test_theta_vector_to_param_dict_preserves_order() -> None:
    theta = jnp.asarray([1.0, 2.0, 3.0])
    params = theta_vector_to_param_dict(theta, ("a", "b", "c"))

    assert list(params) == ["a", "b", "c"]
    assert params["b"] == 2.0


def test_model_mags_from_theta_matrix_supports_shapes_and_gradients(
    monkeypatch,
) -> None:
    import euclid_dsps.parameter_vectors as vectors

    def fake_model_mags(context, model_args, params):
        del context, model_args
        return jnp.asarray([params["a"] + 2.0 * params["b"]], dtype=jnp.float32)

    monkeypatch.setattr(vectors, "model_mags_jax_dynamic", fake_model_mags)
    names = ("a", "b")

    single = model_mags_from_theta_matrix_jax(None, (), jnp.asarray([1.0, 2.0]), names)
    batch = model_mags_from_theta_matrix_jax(
        None,
        (),
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        names,
    )
    samples = model_mags_from_theta_matrix_jax(
        None,
        (),
        jnp.ones((2, 3, 2)),
        names,
    )

    assert single.shape == (1,)
    assert batch.shape == (2, 1)
    assert samples.shape == (2, 3, 1)
    grad = jax.grad(
        lambda theta: jnp.sum(model_mags_from_theta_matrix_jax(None, (), theta, names))
    )(jnp.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(grad), np.asarray([1.0, 2.0]))


def test_theta_vector_to_model_param_dict_merges_fixed_parameters() -> None:
    params = theta_vector_to_model_param_dict(
        jnp.asarray([1.5, 2.5]),
        ("free_a", "shared"),
        {
            "fixed_parameters": {
                "fixed_only": 4.0,
                "shared": -10.0,
                "not_scalar": [1.0, 2.0],
            }
        },
    )

    assert set(params) == {"fixed_only", "free_a", "shared"}
    np.testing.assert_allclose(np.asarray(params["fixed_only"]), 4.0)
    np.testing.assert_allclose(np.asarray(params["free_a"]), 1.5)
    np.testing.assert_allclose(np.asarray(params["shared"]), 2.5)


def test_truth_basic_parameters_map_to_popcosmos_decoder_inputs() -> None:
    params = theta_vector_to_model_param_dict(
        jnp.asarray([0.7, 10.0, -9.4, 0.543, -0.2], dtype=jnp.float32),
        (
            "z_obs",
            "log10_stellar_mass",
            "log10_ssfr_at_obs",
            "dust_av",
            "dust_delta",
        ),
        {
            "sfh_model": "popcosmos_bins",
            "truth_basic_ssfr_reference": -10.0,
            "fixed_parameters": {
                "dlog10_sfr_1": 0.0,
                "tau2": 0.3,
                "dust_index_n": -0.7,
            },
        },
    )

    expected_sfr_slope = (-9.4 - -10.0) / 6.0
    for index in range(1, 7):
        np.testing.assert_allclose(
            np.asarray(params[f"dlog10_sfr_{index}"]),
            expected_sfr_slope,
            rtol=1.0e-5,
        )
    np.testing.assert_allclose(np.asarray(params["tau2"]), 0.5, rtol=1.0e-5)
    np.testing.assert_allclose(np.asarray(params["dust_index_n"]), -0.2, rtol=1.0e-5)


def test_model_mags_from_theta_matrix_supplies_fixed_parameters(monkeypatch) -> None:
    import euclid_dsps.parameter_vectors as vectors

    def fake_model_mags(context, model_args, params):
        del context, model_args
        return jnp.asarray([params["free_a"] + params["fixed_only"]], dtype=jnp.float32)

    monkeypatch.setattr(vectors, "model_mags_jax_dynamic", fake_model_mags)
    context = SimpleNamespace(model_config={"fixed_parameters": {"fixed_only": 3.0}})

    mags = model_mags_from_theta_matrix_jax(
        context,
        (),
        jnp.asarray([2.0]),
        ("free_a",),
    )

    np.testing.assert_allclose(np.asarray(mags), np.asarray([5.0], dtype=np.float32))


def test_photometry_arrays_fs2_synthetic_batch() -> None:
    bands = _fs2_bands()
    validate_fs2_band_contract(bands)
    frame = pd.DataFrame({band["column"]: [1.0e-29, -2.0e-29] for band in bands})
    for band in bands:
        frame[band["error_column"]] = [1.0e-30, 2.0e-30]

    arrays = photometry_arrays_from_dataframe(frame, bands)

    assert arrays.flux.shape == (2, 10)
    assert arrays.flux_err.shape == (2, 10)
    assert arrays.mask.shape == (2, 10)
    assert arrays.mask.all()
    assert arrays.flux[1, 0] < 0.0


def test_photometry_arrays_accept_vectorized_ab_magnitudes() -> None:
    frame = pd.DataFrame({"object_id": [11, 12], "mag_u": [20.0, 21.0]})
    arrays = photometry_arrays_from_dataframe(
        frame,
        [
            {
                "name": "u",
                "column": "mag_u",
                "units": "abmag",
                "sigma_mag": 0.1,
            }
        ],
        object_id_column="object_id",
    )

    assert arrays.object_id.tolist() == [11, 12]
    assert arrays.flux.shape == (2, 1)
    np.testing.assert_allclose(
        arrays.flux[:, 0],
        np.asarray(abmag_to_fnu_cgs([20.0, 21.0]), dtype=np.float32),
        rtol=1.0e-6,
    )
    assert arrays.flux_err.shape == (2, 1)
    assert np.all(arrays.flux_err > 0.0)
    assert arrays.mask.all()


def _fs2_bands():
    names = (
        "lsst_u",
        "lsst_g",
        "lsst_r",
        "lsst_i",
        "lsst_z",
        "lsst_y",
        "euclid_vis",
        "euclid_nisp_y",
        "euclid_nisp_j",
        "euclid_nisp_h",
    )
    return [
        {
            "name": name,
            "column": name,
            "units": "fnu_cgs",
            "error_column": f"{name}_err",
            "error_units": "fnu_cgs",
            "sigma_mag": 0.05,
        }
        for name in names
    ]
