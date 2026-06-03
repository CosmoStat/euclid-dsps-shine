from __future__ import annotations

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
    theta_vector_to_param_dict,
)


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
