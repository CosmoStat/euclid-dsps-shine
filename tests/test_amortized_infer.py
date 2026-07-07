from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

HAS_DEPS = (
    importlib.util.find_spec("equinox") is not None
    and importlib.util.find_spec("optax") is not None
)
pytestmark = pytest.mark.skipif(
    not HAS_DEPS,
    reason="Equinox/Optax optional dependencies are not installed",
)

if HAS_DEPS:
    from euclid_dsps.amortized import infer as infer_mod
    from euclid_dsps.amortized.latent import LatentSpec


def test_model_flux_from_x_sample_chunks_splits_sample_axis(monkeypatch) -> None:
    calls = []

    def fake_model_flux_from_x(x, latent_spec, context, model_args, parameter_names):
        calls.append(tuple(x.shape))
        return jnp.ones(x.shape[:-1] + (10,), dtype=x.dtype) * len(calls)

    monkeypatch.setattr(infer_mod, "model_flux_from_x", fake_model_flux_from_x)
    spec = LatentSpec(
        names=tuple(f"p{i}" for i in range(16)),
        lower=jnp.zeros(16),
        upper=jnp.ones(16),
    )
    x = jnp.zeros((5, 2, 16), dtype=jnp.float32)

    flux = infer_mod._model_flux_from_x_sample_chunks(
        x,
        spec,
        None,
        None,
        spec.names,
        sample_chunk_size=2,
    )

    assert flux.shape == (5, 2, 10)
    assert calls == [(2, 2, 16), (2, 2, 16), (1, 2, 16)]
    assert jnp.all(flux[:2] == 1.0)
    assert jnp.all(flux[2:4] == 2.0)
    assert jnp.all(flux[4:] == 3.0)


def test_model_flux_from_x_sample_chunks_rejects_nonpositive_chunk() -> None:
    spec = LatentSpec(
        names=tuple(f"p{i}" for i in range(16)),
        lower=jnp.zeros(16),
        upper=jnp.ones(16),
    )
    with pytest.raises(ValueError, match="sample_chunk_size must be positive"):
        infer_mod._model_flux_from_x_sample_chunks(
            jnp.zeros((2, 1, 16)),
            spec,
            None,
            None,
            spec.names,
            sample_chunk_size=0,
        )


def test_posterior_predictive_chi2_is_finite_for_tiny_fluxes() -> None:
    batch = SimpleNamespace(
        flux=jnp.asarray([[2.0e-41, 3.0e-40]], dtype=jnp.float32),
        flux_err=jnp.asarray([[2.5e-31, 7.0e-32]], dtype=jnp.float32),
        mask=jnp.asarray([[True, True]]),
    )
    model_flux = jnp.asarray([[[3.0e-35, 2.0e-31]]], dtype=jnp.float32)

    chi2 = infer_mod._posterior_predictive_chi2(
        batch,
        model_flux,
        {"error_floor_frac": 0.02, "error_jitter": 0.0},
    )

    assert chi2.shape == (1, 1)
    assert np.isfinite(chi2).all()


def test_combine_inference_shard_tables_writes_dense_outputs(tmp_path) -> None:
    out = tmp_path
    for directory in infer_mod._inference_shard_dirs(out).values():
        directory.mkdir(parents=True, exist_ok=True)

    for batch in (1, 2):
        paths = infer_mod._inference_shard_paths(out, batch)
        pd.DataFrame(
            {
                "object_id": [f"obj-{batch}"],
                "row_index": [batch - 1],
                "redshift_median": [0.5 + batch],
            }
        ).to_parquet(paths["summary"], index=False)
        pd.DataFrame({"object_id": [f"obj-{batch}"], "sample_id": [0]}).to_parquet(
            paths["samples"],
            index=False,
        )
        pd.DataFrame({"object_id": [f"obj-{batch}"], "residual_rms": [0.1]}).to_parquet(
            paths["residual_summary"],
            index=False,
        )
        pd.DataFrame({"object_id": [f"obj-{batch}"], "finite_flux": [True]}).to_parquet(
            paths["features"],
            index=False,
        )
        paths["metadata"].write_text(
            f'{{"batch": {batch}, "counts": {{"summary_rows": 1}}}}',
            encoding="utf-8",
        )

    records = infer_mod._discover_shard_records(out)
    frames = infer_mod._combine_inference_shard_tables(
        out,
        records,
        combine_sample_shards=False,
    )

    assert [record["batch"] for record in records] == [1, 2]
    assert all(record["complete"] for record in records)
    assert len(frames["summary"]) == 2
    assert len(frames["features"]) == 2
    assert len(frames["residual_summary"]) == 2
    assert not (out / "posterior_samples.parquet").exists()
    assert pd.read_parquet(out / "posterior_summary.parquet")["row_index"].tolist() == [
        0,
        1,
    ]
