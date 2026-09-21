"""Small stage-integration test; only DSPS assets/catalogue loading are replaced."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_forward_stages_freeze_parent_and_train_15d(tmp_path, monkeypatch):
    from pathlib import Path

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import yaml

    from euclid_dsps.amortized import forward_population_runtime as adapter
    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.features import FeatureStats, make_encoder_features
    from euclid_dsps.amortized.flows import StandardNormalPrior
    from euclid_dsps.amortized.forward_population import PHYSICAL, PhysicalBasis
    from euclid_dsps.amortized.latent import LatentSpec, x_to_theta
    from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
    from euclid_dsps.amortized.proposal_expressivity import IndependentFlowMixture
    from scripts import feniks_forward_population as run

    cfg = yaml.safe_load(
        Path("configs/experiments/feniks_forward_population_r29.yaml").read_text()
    )
    cfg.update(
        reference_simulations=4096,
        reference_shards=1,
        posterior_simulations=1024,
        posterior_shards=1,
        evaluation_simulations=1024,
        simulation_batch=64,
        bank_checkpoint_batches=16,
        report_objects=128,
        report_draws=32,
        population_report_draws=256,
        cut=29.0,
    )
    cfg["selection_support"] = dict(min_selected=5, min_alpha=0.001)
    cfg["classifier"].update(
        width=16,
        depth=1,
        batch_size=128,
        epochs=12,
        learning_rate=0.01,
        validation_limit=512,
    )
    cfg["posterior"].update(
        batch_size=64, epochs=2, learning_rate=0.001, validation_limit=128
    )
    names = (*PHYSICAL, *(f"sfh_dlog_sfr_{i:02d}" for i in range(1, 11)))
    spec = LatentSpec(names, jnp.full(15, -10.0), jnp.full(15, 10.0))
    basis = PhysicalBasis.create(names, 4, seed=9)
    root = tmp_path / "run"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    for name in ("reference", "population", "posterior_bank", "posterior", "report"):
        (root / name).mkdir()
    np.savez(
        root / "basis.npz", names=names, centers=basis.centers, scales=basis.scales
    )
    stats = FeatureStats(
        np.full(5, 1e-30), np.full(5, 1e-31), ("lsst_r", "g", "i", "z", "y")
    )

    def simulate(x, key):
        errors = jnp.full((len(x), 5), 1e-31)
        flux = 1e-30 * jnp.exp(x[:, :5] * 0.3) + errors * jax.random.normal(
            key, errors.shape
        )
        mask = jnp.ones_like(flux, dtype=bool)
        return dict(
            x=x,
            theta=x_to_theta(x, spec),
            flux=flux,
            errors=errors,
            mask=mask,
            valid=jnp.ones(len(x), dtype=bool),
            selected=flux[:, 0] > 10 ** (-0.4 * (29 + 48.6)),
            features=make_encoder_features(flux, errors, stats, mask),
        )

    sim = eqx.filter_jit(simulate)
    x, _ = basis.sample(np.random.default_rng(12), 512, weights=[0.1, 0.2, 0.3, 0.4])
    observed = {
        k: np.asarray(v) for k, v in sim(jnp.asarray(x), jax.random.PRNGKey(3)).items()
    }
    sel = np.flatnonzero(observed["selected"])
    arrays = SimpleNamespace(
        flux=observed["flux"][sel],
        flux_err=observed["errors"][sel],
        mask=observed["mask"][sel],
        row_index=sel,
    )
    catalog = source / "catalog.parquet"
    pd.DataFrame(observed["theta"], columns=names).to_parquet(catalog, index=False)
    np.save(source / "train.npy", sel)
    run.write(
        source / "MANIFEST.json",
        dict(source={"feature_stats": "mock"}, validation_catalog=str(catalog)),
    )
    run.write(
        root / "MANIFEST.json",
        dict(
            settings=cfg,
            hashes={},
            source=str(source),
            blind_truth_parent=None,
            blind_truth_selected=None,
            baseline=None,
        ),
    )
    enc = ConditionalFlowEncoder(
        jax.random.PRNGKey(21),
        input_dim=10,
        latent_dim=15,
        hidden_sizes=(8,),
        family="rq_spline",
        n_layers=1,
        hidden_size=8,
        n_bins=4,
        output_space="latent_x",
        activation="gelu",
        log_std_min=-5.0,
        log_std_max=3.0,
        initial_log_std=0.0,
    )
    model = AmortizedModel(enc, StandardNormalPrior(latent_dim=15), None)
    candidate = IndependentFlowMixture(jax.random.PRNGKey(22), enc, n_components=2)
    runtime = SimpleNamespace(
        latent_spec=spec, validation_arrays=arrays, feature_stats=stats
    )
    monkeypatch.setattr(
        adapter, "load_forward_runtime", lambda *a: (model, runtime, {}, {})
    )
    monkeypatch.setattr(adapter, "simulator", lambda *a: sim)
    monkeypatch.setattr(adapter, "posterior_template", lambda *a: candidate)
    monkeypatch.setattr("euclid_dsps.config.load_config", lambda *a: {})
    monkeypatch.setattr(
        "euclid_dsps.amortized.features.read_feature_stats", lambda *a: stats
    )
    monkeypatch.setattr(
        "euclid_dsps.amortized.data.load_photometry_arrays_from_config",
        lambda *a, **kw: arrays,
    )
    run.bank(root, 0, "reference")
    fit_weights = run.fit_selected_weights

    def failed_fit(*args, **kwargs):
        raise RuntimeError("injected population solver failure")

    monkeypatch.setattr(run, "fit_selected_weights", failed_fit)
    with pytest.raises(RuntimeError, match="injected population solver failure"):
        run.population(root)
    saved_classifier = run.sha(root / "population/best.eqx")
    training_log = (root / "population/training.jsonl").read_bytes()
    assert (root / "population/observed_ratios.npz").is_file()
    assert not (root / "population/FINAL.json").exists()
    monkeypatch.setattr(run, "fit_selected_weights", fit_weights)
    run.population(root)
    assert run.sha(root / "population/best.eqx") == saved_classifier
    assert (root / "population/training.jsonl").read_bytes() == training_log
    parent_hash = run.sha(root / "population/parent.json")
    run.bank(root, 0, "posterior")
    run.bank(root, 1, "posterior")
    run.train_posterior(root)
    assert run.sha(root / "population/parent.json") == parent_hash
    assert run.read(root / "posterior/FINAL.json")["rws_updates"] == 0
    # Skip expensive repeated corner rendering, retaining real report statistics,
    # raw joint draws, calibration, PPC and other figure generation.
    monkeypatch.setattr("scripts.feniks_avi_overnight._corner", lambda *a: None)
    monkeypatch.setattr("scripts.feniks_avi_overnight._marginals", lambda *a: None)
    from scripts.report_feniks_forward_population import report

    evaluation_path = root / "evaluation_indices.npy"
    np.save(evaluation_path, sel)
    manifest = run.read(root / "MANIFEST.json")
    manifest["evaluation_indices"] = str(evaluation_path)
    run.write(root / "MANIFEST.json", manifest)
    report(root)
    assert run.read(root / "report/FINAL.json")["posterior_dimensions"] == 15
    cal = pd.read_csv(root / "report/simulation_calibration.csv")
    assert len(cal) == 15
    assert (cal.group == "physical").sum() == 5
    # Exercise both capacity targets with the real 15D mixture, without DSPS.
    from scripts import feniks_forward_diagnostics as diag

    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    run.write(
        diagnostics / "MANIFEST.json",
        dict(
            settings=dict(
                seed=51,
                capacity=dict(
                    epochs=2,
                    batch_size=32,
                    learning_rate=0.001,
                    validation_limit=32,
                    training_draws=64,
                    test_draws=64,
                    fixed_validation=True,
                ),
            )
        ),
    )
    for task, name in enumerate(("capacity_analytic", "capacity_truth")):
        branch = diagnostics / name
        (branch / "report").mkdir(parents=True)
        manifest = run.read(root / "MANIFEST.json")
        manifest["blind_truth_parent"] = str(catalog)
        run.write(branch / "MANIFEST.json", manifest)
        (branch / "basis.npz").symlink_to(root / "basis.npz")
        (branch / "population").symlink_to(
            root / "population", target_is_directory=True
        )
        diag.capacity(diagnostics, task)
        final = run.read(branch / "report/FINAL.json")
        assert final["dimensions"] == 15
        assert not final["production_prior_modified"]
        if task == 0:
            assert np.isfinite(final["kl_target_flow"])
        else:
            split = np.load(branch / "report/truth_split.npz")
            assert not set(split["train"]) & set(split["test"])
            assert not set(split["validation"]) & set(split["test"])
