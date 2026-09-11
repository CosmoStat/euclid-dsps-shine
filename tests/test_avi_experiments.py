import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from euclid_dsps.amortized.avi_experiments import (
    ARMS,
    enumerated_elbo,
    log_prob,
    normalized_weights,
    stratified_proposal,
    weighted_nll,
)
from euclid_dsps.amortized.elbo import AmortizedModel
from euclid_dsps.amortized.flows import StandardNormalPrior
from euclid_dsps.amortized.posterior import ConditionalFlowEncoder, posterior_log_prob
from euclid_dsps.amortized.proposal_expressivity import IndependentFlowMixture
from euclid_dsps.calibration import GlobalSedScaleState
from scripts.feniks_avi_experiments import (
    enable_avi_selection,
    frozen_selection_normalization,
    phase_at,
    read,
    runtime_asset_paths,
    save_state,
    sha,
    source_config_text,
    validate_rows,
)


def test_avi_restores_inherited_selection_without_changing_target():
    from euclid_dsps.amortized.latent import latent_spec_from_config, latent_spec_hash
    from euclid_dsps.config import load_config

    config = load_config("configs/experiments/feniks_sc_drws_r29_frozen_parent_sleep_npe.yaml")
    objective = config["amortized"]["objective"]
    original = dict(objective["selection_correction"])
    latent = latent_spec_hash(latent_spec_from_config(config))
    assert original["enabled"] is False
    enable_avi_selection(config)
    assert objective["selection_correction"] == {**original, "enabled": True}
    assert objective["sleep"]["selection"]["enabled"]
    assert latent_spec_hash(latent_spec_from_config(config)) == latent
    objective["selection_correction"]["max_mag_ab"] = 25.0
    with pytest.raises(ValueError, match="inherited observed"):
        enable_avi_selection(config)


def test_frozen_selection_receipt_requires_enabled_correction(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from euclid_dsps.amortized import adaptive_smc_trainer

    runtime = SimpleNamespace(selection_objective_config={
        "selection_correction": {"enabled": True}, "prior_train_jointly": False,
    })
    monkeypatch.setattr(adaptive_smc_trainer, "_make_selection_log_alpha_fn",
                        lambda rt: lambda model, key: (
                            jnp.log(0.5), {"selection/alpha": jnp.array(0.5)}))
    value = frozen_selection_normalization(model(), runtime, {"seed": 3}, tmp_path)
    assert value == pytest.approx(np.log(0.5))
    receipt = read(tmp_path / "SELECTION.json")
    assert receipt["enabled"] and receipt["constant_wrt_encoder"]
    assert receipt["selection_in_object_weights"] is False
    runtime.selection_objective_config["selection_correction"]["enabled"] = False
    with pytest.raises(ValueError, match="must remain enabled"):
        frozen_selection_normalization(model(), runtime, {"seed": 3}, tmp_path)


def test_snapshot_contains_filter_contents_and_data_link(tmp_path):
    script = Path("scripts/submit_feniks_avi_experiments.sh").read_text()
    setup = script.split("JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python", 1)[0]
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("euclid_dsps", "scripts", "configs", "Data"):
        (repo / name).mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    curves = tmp_path / "site_filters"
    curves.mkdir()
    curve = curves / "Euclid_VIS.vis.dat"
    curve.write_text("4000 0\n5000 1\n6000 0\n")
    (repo / "filters").symlink_to(curves, target_is_directory=True)
    base = tmp_path / "base"
    base.mkdir()
    result = subprocess.run(
        ["bash", "-c", setup + '\nprintf "%s" "$AVI_CODE"', "snapshot",
         str(base), str(base / "new_run"), "7"],
        cwd=repo, env=dict(os.environ, SCRATCH=str(tmp_path / "scratch")),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    snapshot = Path(result.stdout)
    assert (snapshot / "Data").resolve() == repo / "Data"
    assert not (snapshot / "filters").is_symlink()
    copied = snapshot / "filters" / curve.name
    assert copied.read_text() == curve.read_text()
    curve.write_text("changed source\n")
    assert copied.read_text() == "4000 0\n5000 1\n6000 0\n"


def test_runtime_assets_checked_and_parsed_before_submission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "filters").mkdir()
    curve = tmp_path / "filters/vis.dat"
    curve.write_text("4000 0\n5000 1\n6000 0\n")
    ssp = tmp_path / "ssp.h5"
    ssp.write_bytes(b"SSP fixture: existence only; no decoder load")
    config = {"ssp_path": "ssp.h5", "bands": [{
        "name": "euclid_vis", "filter": {"kind": "ascii", "path": "filters/vis.dat"},
    }]}
    assert set(runtime_asset_paths(config)) == {ssp, curve}
    curve.unlink()
    with pytest.raises(FileNotFoundError, match="filter:euclid_vis"):
        runtime_asset_paths(config)
    curve.write_text("not a filter\n")
    with pytest.raises(ValueError):
        runtime_asset_paths(config)
    ssp.unlink()
    with pytest.raises(FileNotFoundError, match="ssp_path"):
        runtime_asset_paths(config)


def test_source_config_preserves_checkpoint_coordinates(tmp_path):
    import json

    import yaml

    from euclid_dsps.amortized.latent import latent_spec_from_config, latent_spec_hash

    config = {
        "fit": {"free_parameters": {
            "z_obs": {"bounds": [0.01, 5.0], "init": 1.0},
            "dust_av": {"bounds": [0.0, 6.0], "init": 0.2},
        }},
        "amortized": {
            "latent": {"schema": "config_free_parameters", "normalization": "identity"},
            "prior": {"source": "standard_normal"},
        },
    }
    expected = latent_spec_hash(latent_spec_from_config(config))
    checkpoint = tmp_path / "source.eqx"
    checkpoint.with_suffix(".eqx.json").write_text(json.dumps({"latent_spec_hash": expected}))
    restored = yaml.safe_load(source_config_text(config, checkpoint))
    assert list(restored["fit"]["free_parameters"]) == ["z_obs", "dust_av"]
    assert latent_spec_hash(latent_spec_from_config(restored)) == expected
    # The old default sorted dump corrupts the coordinate contract and must fail.
    reordered = yaml.safe_load(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="checkpoint/config latent hash mismatch"):
        source_config_text(reordered, checkpoint)


def test_slurm_enables_cuda_plugin_discovery(tmp_path):
    script = Path("scripts/feniks_avi_experiments.slurm").read_text()
    # Execute actual environment setup with only site-specific commands stubbed.
    setup = script.split("python -m scripts.feniks_avi_experiments", 1)[0]
    conda = tmp_path / "miniconda3/etc/profile.d/conda.sh"
    conda.parent.mkdir(parents=True)
    conda.write_text("conda() { :; }\n")
    env = dict(os.environ, WORK=str(tmp_path), SCRATCH=str(tmp_path),
               AVI_CODE=str(Path.cwd()), SLURM_JOB_ID="test",
               EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD="1")
    probe = """
import os
import jax._src.xla_bridge as bridge
discover = bridge.discover_pjrt_plugins
from euclid_dsps.jax_runtime import configure_jax_runtime
configure_jax_runtime()
assert os.environ['JAX_PLATFORMS'] == 'cuda'
assert bridge.discover_pjrt_plugins is discover
"""
    result = subprocess.run(
        ["bash", "-c", "module() { :; }\n" + setup +
         '\n"' + sys.executable + '" - <<\'PY\'\n' + probe + "\nPY\n"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def model():
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(1),
        input_dim=3,
        latent_dim=2,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
        family="realnvp",
        n_layers=2,
        hidden_size=8,
        init_scale=0.1,
        output_space="latent_x",
        transport_float64=True,
    )
    return AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=2),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.array(0.0)),
    )


def test_exact_mis_denominator_ignores_learned_gate():
    m = model()
    c = IndependentFlowMixture(jax.random.PRNGKey(4), m.encoder, n_components=4)
    c = eqx.tree_at(lambda c: c.experts, c, (m.encoder,) * 4)
    f = jnp.ones((3, 3), jnp.float32)
    x, r = stratified_proposal(m, c, f, jax.random.PRNGKey(3))
    expected = jnp.logaddexp(
        posterior_log_prob(m, f, x) + jnp.log(7 / 8),
        m.prior.log_prob(x) + jnp.log(1 / 8),
    )
    assert x.shape == (128, 3, 2)
    assert x.dtype == jnp.float64
    np.testing.assert_allclose(r, expected, atol=1e-10)
    assert np.isfinite(log_prob(m, c, f, x)).all()


def test_weights_nonfinite_and_zero_mass_are_safe():
    w, valid, ess = normalized_weights(jnp.array([[0.0, -jnp.inf], [0.0, -jnp.inf]]))
    np.testing.assert_array_equal(valid, [True, False])
    np.testing.assert_allclose(ess, [2, 0])
    assert weighted_nll(jnp.array([[-2.0, -jnp.inf], [-2.0, -jnp.inf]]), w, valid) == 2
    _, valid, _ = normalized_weights(jnp.array([[0.0], [jnp.inf]]))
    assert not valid[0]


def test_sleep_generator_supports_exact_mixture_callback(monkeypatch):
    from euclid_dsps.amortized import train as training

    m = model()
    c = IndependentFlowMixture(jax.random.PRNGKey(4), m.encoder, n_components=4)
    f = jnp.ones((2, 3), jnp.float32)
    x = jnp.zeros((2, 2), jnp.float64)
    b = training.LossBatch(
        jnp.zeros((2, 1)),
        jnp.ones((2, 1)),
        jnp.ones((2, 1), bool),
        f,
        jnp.zeros((2, 0)),
        x,
        jnp.zeros((2, 1)),
        jnp.ones(2, bool),
    )
    monkeypatch.setattr(training, "_sleep_flux_error", lambda *a: jnp.ones((2, 1)))
    monkeypatch.setattr(
        training, "_sample_sleep_noise", lambda *a, **k: (jnp.zeros((2, 1)), "gaussian")
    )
    monkeypatch.setattr(training, "_sleep_encoder_features", lambda *a: f)
    value, metrics = training._model_generated_sleep_loss(
        m,
        b,
        None,
        None,
        None,
        ("a", "b"),
        jax.random.PRNGKey(5),
        {"type": "gaussian"},
        {},
        {"sleep": {"enabled": True}},
        log_prob_fn=lambda f, x: log_prob(m, c, f, x),
    )
    np.testing.assert_allclose(value, -jnp.mean(log_prob(m, c, f, x)))
    assert metrics["finite_fraction"] == 1.0


def test_enumerated_elbo_has_gate_gradient():
    m = model()
    c = IndependentFlowMixture(
        jax.random.PRNGKey(4), m.encoder, n_components=4, mean_offset=0.8
    )
    f = jnp.ones((2, 3), jnp.float32)

    def fn(c):
        return enumerated_elbo(
            m,
            c,
            f,
            jax.random.PRNGKey(5),
            lambda x: -0.5 * jnp.sum((x - 2.0) ** 2, axis=-1),
        )

    value, grad = eqx.filter_value_and_grad(fn)(c)
    assert np.isfinite(value)
    assert np.linalg.norm(grad.gate.layers[-1].bias) > 1e-5
    eps = 0.005
    direction = jnp.array([1.0, -1.0, 0.0, 0.0], jnp.float32)
    b = c.gate.layers[-1].bias
    plus = eqx.tree_at(lambda a: a.gate.layers[-1].bias, c, b + eps * direction)
    minus = eqx.tree_at(lambda a: a.gate.layers[-1].bias, c, b - eps * direction)
    np.testing.assert_allclose(
        (fn(plus) - fn(minus)) / (2 * eps),
        jnp.dot(grad.gate.layers[-1].bias, direction),
        rtol=0.005,
    )


def test_schedule_and_split():
    m = dict(bootstrap_epochs=6, cycle=["sleep", "sleep", "wake"])
    assert [phase_at(i, ARMS[0], m) for i in range(3)] == m["cycle"]
    assert phase_at(8, ARMS[2], m) == "wake"
    validate_rows(np.arange(10), np.arange(5), False, 10)
    with pytest.raises(ValueError, match="overlap"):
        validate_rows(np.arange(10), np.arange(5), True, 10)


def test_atomic_full_optimizer_resume(tmp_path):
    c = jnp.array([1.0, 2.0])
    opt = optax.adam(1e-3)
    state = opt.init(c)
    updates, state = opt.update(jnp.ones_like(c), state, c)
    c = optax.apply_updates(c, updates)
    save_state(tmp_path, c, state, 1, "test")
    receipt = read(tmp_path / "RESUME.json")
    assert receipt["next_step"] == 1
    assert sha(tmp_path / receipt["path"]) == receipt["sha256"]
    restored, os = eqx.tree_deserialise_leaves(tmp_path / receipt["path"], (c, state))
    u1, _ = opt.update(jnp.ones_like(c), state, c)
    u2, _ = opt.update(jnp.ones_like(c), os, restored)
    np.testing.assert_array_equal(u1, u2)


def test_teacher_import_preserves_joint_draws_and_target_provenance(tmp_path):
    import pandas as pd

    from scripts.feniks_avi_experiments import build_teachers, write

    root = tmp_path / "nuts"
    root.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    checkpoint = tmp_path / "source.eqx"
    checkpoint.write_bytes(b"frozen")
    write(
        root / "MANIFEST.json",
        {"reference": str(reference), "inputs": {str(checkpoint): sha(checkpoint)}},
    )
    pd.DataFrame([{"case": "observed_000", "source_row": 3}]).to_csv(
        root / "OBSERVED_COHORT.csv", index=False
    )
    group = root / "nuts/observed_000/B_dense_depth6"
    group.mkdir(parents=True)
    write(
        group / "FINAL.json", {"status": "SAMPLING_COMPLETE", "diagnostics_pass": False}
    )
    for chain in range(8):
        chunks = group / f"chain_{chain}/chunks"
        chunks.mkdir(parents=True)
        x = np.array([chain, chain + 0.5], dtype=np.float64)
        pd.DataFrame({"x_00": x, "x_01": -x}).to_parquet(chunks / "part_000000.parquet")
    (root / "observed_000").mkdir()
    np.savez(
        root / "observed_000/observation.npz",
        flux=np.ones((1, 1)),
        flux_err=np.ones((1, 1)),
        mask=np.ones((1, 1), bool),
    )
    bank, meta, _ = build_teachers(
        root, np.arange(10), {"checkpoint": str(checkpoint)}, reference
    )
    assert bank["x"].shape == (1, 16, 2)
    np.testing.assert_array_equal(bank["x"][..., 0], -bank["x"][..., 1])
    assert meta[0]["final"]["diagnostics_pass"] is False
    with pytest.raises(ValueError, match="not a training identity"):
        build_teachers(root, np.arange(2), {"checkpoint": str(checkpoint)}, reference)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        build_teachers(root, np.arange(10), {"checkpoint": str(checkpoint)}, reference)


def test_four_device_accumulation_and_nan_rollback():
    code = """
import jax, jax.numpy as j, equinox as eqx, optax, numpy as np
from euclid_dsps.amortized.avi_experiments import make_parallel_steps
from euclid_dsps.amortized.adaptive_smc_trainer import _replicate_model_for_pmap
devices=tuple(jax.local_devices())
assert len(devices)==4
def loss(c, x, k):
    return j.mean((c-x)**2), j.zeros(4,j.float64)
opt=optax.sgd(.1)
c=j.array(0.,j.float64)
s=opt.init(c)
cr=_replicate_model_for_pmap(c,devices)
sr=_replicate_model_for_pmap(s,devices)
x=j.arange(32,dtype=j.float64).reshape(4,2,4)
step=make_parallel_steps(loss,opt,devices=devices)
cr,sr,_,_,ok=step(cr,sr,x,jax.random.split(jax.random.PRNGKey(0),4))
np.testing.assert_allclose(np.asarray(cr), .2*np.mean(np.arange(32)))
assert np.all(ok)
old=cr
cr,sr,_,_,ok=step(cr,sr,j.full_like(x,j.nan),jax.random.split(jax.random.PRNGKey(1),4))
assert not np.any(ok)
np.testing.assert_array_equal(cr,old)
"""
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_seven_arm_preflight_and_resume_on_mock_physics(tmp_path):
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": ".",
    }
    result = subprocess.run(
        [sys.executable, "tests/avi_mock_worker.py", str(tmp_path / "run")],
        env=env,
        text=True,
        capture_output=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
