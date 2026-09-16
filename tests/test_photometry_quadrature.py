"""Numerical photometry controls without a catalogue or real SSP assets."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from euclid_dsps.amortized import photometry_reference as reference
from euclid_dsps.amortized.local_vi_diagnostic import Budget
from euclid_dsps.amortized.redshift_decomposition import floating_trace_dtypes
from euclid_dsps.model import fixed_spectrum_projection_jax
from euclid_dsps.photometry_quadrature import (
    filter_integral_numpy,
    interpolation_knot_audit,
    linear_product_integral_numpy,
    merged_integral_jax,
)


def test_smooth_closed_form_and_gradient():
    wave = jnp.array([1.0, 3.0, 10.0], dtype=jnp.float64)
    fw = jnp.array([4.0, 5.0, 7.0], dtype=jnp.float64)
    ft = jnp.ones_like(fw)
    z = jnp.array(0.4, dtype=jnp.float64)

    def f(zz):
        return merged_integral_jax(wave, wave, fw, ft, zz)

    assert float(f(z)) == pytest.approx(3 / 1.4, rel=1e-10)
    assert float(jax.grad(f)(z)) == pytest.approx(-3 / 1.4**2, rel=1e-9)
    assert linear_product_integral_numpy(wave, wave, fw, ft, 0.4) == pytest.approx(
        3 / 1.4, rel=1e-12
    )
    assert filter_integral_numpy(fw, ft) == pytest.approx(np.log(7 / 4), rel=1e-12)


def test_narrow_line_resolved_without_filter_resampling():
    wave = np.array([4000.0, 4999.0, 5000.0, 5001.0, 6000.0])
    lum = np.array([0.001, 0.001, 1.001, 0.001, 0.001])
    fw = np.array([4500.0, 4800.0, 5200.0, 5500.0])
    ft = np.array([0.3, 0.5, 0.8, 0.9])
    z = 0.013
    ref = linear_product_integral_numpy(wave, lum, fw, ft, z)
    knots = np.unique(np.clip(np.r_[wave * (1 + z), fw], fw[0], fw[-1]))
    independent, _ = quad(
        lambda w: np.interp(w, fw, ft) * np.interp(w, wave * (1 + z), lum) / w,
        fw[0],
        fw[-1],
        points=knots[1:-1],
        epsabs=1e-13,
        limit=100,
    )
    assert ref == pytest.approx(independent, rel=1e-10)
    f = jax.jit(
        lambda zz: merged_integral_jax(*map(jnp.asarray, (wave, lum, fw, ft)), zz)
    )
    assert float(f(z)) == pytest.approx(ref, rel=1e-10)
    fd = (
        linear_product_integral_numpy(wave, lum, fw, ft, z + 1e-6)
        - linear_product_integral_numpy(wave, lum, fw, ft, z - 1e-6)
    ) / 2e-6
    assert float(jax.grad(f)(z)) == pytest.approx(fd, rel=1e-6, abs=1e-10)
    from dsps.photometry.photometry_kernels import _obs_flux_ssp

    old = float(_obs_flux_ssp(*map(jnp.asarray, (wave, lum, fw, ft)), z))
    assert abs(old - ref) / ref > 0.3


def test_support_boundary_and_duplicate_knots():
    wave = jnp.array([2.0, 3.0, 4.0], dtype=jnp.float64)
    fw = jnp.array([3.0, 4.0, 5.0], dtype=jnp.float64)

    def f(z):
        return merged_integral_jax(wave, jnp.ones_like(wave), fw, jnp.ones_like(fw), z)

    # At z=0.1 the overlap is [3,4.4], with a moving upper boundary.
    assert float(f(0.1)) == pytest.approx(np.log(4.4 / 3), rel=1e-11)
    assert float(jax.grad(f)(0.1)) == pytest.approx(1 / 1.1, rel=1e-10)
    assert float(f(0.0)) == pytest.approx(np.log(4 / 3), rel=1e-10)
    assert float(f(3.0)) == 0.0
    assert np.isfinite(float(jax.grad(f)(3.0)))


def test_constant_spectrum_ab_normalization_and_dtype():
    wave = jnp.array([1000.0, 4000.0, 10000.0], dtype=jnp.float64)
    fw = jnp.array([4500.0, 4800.0, 5200.0, 5500.0], dtype=jnp.float64)
    ft = jnp.array([0.1, 1.0, 0.8, 0.05], dtype=jnp.float64)
    from dsps.photometry.photometry_kernels import AB0

    from euclid_dsps.model import fixed_spectrum_dimming_factor_jax

    def fn(z):
        return fixed_spectrum_projection_jax(
            wave, jnp.ones_like(wave) * AB0, fw, ft, z, method="merged"
        )

    assert float(fn(0.2)) == pytest.approx(
        float(fixed_spectrum_dimming_factor_jax(0.2)), rel=1e-11
    )
    assert floating_trace_dtypes(
        jax.make_jaxpr(fn)(jnp.array(0.2, dtype=jnp.float64))
    ) == ["float64"]


def test_knot_crossings_and_plateau_independence():
    audit = interpolation_knot_audit([1.0, 2.0, 3.0], [2.1, 2.5], 0.1, [0.01, 0.2])
    assert audit["nearest_knot_distance_z"] == pytest.approx(0.05)
    assert audit["crossings"][0]["switched_filter_samples"] == 0
    assert audit["crossings"][1]["switched_filter_samples"] > 0
    a = reference.reference_plateau([[1.0], [1.0], [1.0]], np.array([1.0]))
    b = reference.reference_plateau([[1.0], [1.0], [1.0]], np.array([3.0]))
    assert a[0]["reference_step_index"] == b[0]["reference_step_index"] == 2
    assert a[0]["status"] == "PASS" and b[0]["status"] == "FAIL"


def test_snapshot_analysis_cpu(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(reference, "STEPS", (0.0001, 0.00005, 0.000025))
    arrays = dict(
        wave=np.array([1000.0, 4000.0, 10000.0]),
        spectra_jit=np.ones((1, 3)),
        spectra_eager=np.ones((1, 3)),
        z=np.array([0.2]),
        step_scale=np.array([1.0]),
        band_names=np.array(["wide"]),
        flux_err=np.array([1e-10]),
        model_flux_jit=np.array([[1e-9]]),
        filter_wave_00=np.array([4500.0, 4800.0, 5200.0, 5500.0]),
        transmission_00=np.array([0.1, 1.0, 0.8, 0.05]),
    )
    result, rows = reference.analyze_snapshot(
        arrays, Budget(120, 1000), progress=lambda *args: None
    )
    assert result["status"] == "PHOTOMETRY_REFERENCE_COMPLETE"
    assert result["numerical_reference_checks"] == "PASS"
    assert result["production_decoder_changed"] is False
    assert len(rows) == 12
    assert result["npe_training_started"] is False

    from scripts import analyze_feniks_sc_drws_photometry_reference as reader

    source = tmp_path / "export"
    source.mkdir()
    archive = source / "FIXED_SPECTRA.npz"
    np.savez_compressed(archive, **arrays)
    (source / "SNAPSHOT.json").write_text(
        json.dumps(dict(sha256=reader.digest(archive), code_commit="test-source"))
    )
    output = tmp_path / "replay"
    reader.replay(source, output)
    reader.summarize(output, 0)
    assert "numerical_checks= PASS" in capsys.readouterr().out
    replay_report = json.loads((output / "PHOTOMETRY_REFERENCE.json").read_text())
    assert replay_report["source_code_commit"] == "test-source"
    assert len(replay_report["replay_source_sha256"]) == 3
    with pytest.raises(FileExistsError):
        reader.replay(source, output)
    (output / "photometry_reference.csv").write_text("tampered")
    with pytest.raises(ValueError, match="changed artifact"):
        reader.summarize(output, 0)
    archive.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed spectrum export"):
        reader.replay(source, tmp_path / "bad-replay")
