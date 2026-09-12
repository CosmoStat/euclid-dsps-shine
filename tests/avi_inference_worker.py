"""Exercise real residual mixture inference; mock only physical runtime and IO."""

import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import equinox as eqx
import jax
import numpy as np
import pandas as pd
from avi_mock_worker import build, target

from euclid_dsps.amortized.avi_experiments import ARMS
from euclid_dsps.amortized.features import FeatureStats
from scripts import feniks_avi_experiments as runner
from scripts.feniks_avi_inference import infer


def main(root):
    root.mkdir()
    training = root / "training"
    training.mkdir()
    np.save(training / "validation.npy", np.arange(5))
    model = build({}, jax.random.PRNGKey(1))
    m = dict(
        source=dict(checkpoint="none", feature_stats="none"),
        seed=3,
        validation_catalog="none",
    )
    contract = dict(
        training=str(training),
        arms=["source", *[a.name for a in ARMS]],
        hashes={},
        particles=32,
        replicas=2,
        seed=21,
        saved_draws=16,
    )
    runner.write(root / "MANIFEST.json", contract)
    arrays = SimpleNamespace(
        flux=np.ones((5, 1), np.float32),
        flux_err=np.ones((5, 1), np.float32),
        mask=np.ones((5, 1), bool),
        row_index=np.arange(5),
    )
    rt = SimpleNamespace(
        latent_spec=SimpleNamespace(names=("a", "b")),
        context=None,
        model_args=None,
        parameter_names=("a", "b"),
        validation_arrays=arrays,
        feature_stats=FeatureStats(np.ones(1), np.ones(1), ("f",), append_mask=True),
        likelihood_config={"type": "gaussian"},
        calibration_config={},
        selection_objective_config={"selection_correction": {"enabled": True}},
    )
    with ExitStack() as stack:
        for name, value in {
            "scripts.feniks_avi_experiments.check_inputs": lambda p: m,
            "euclid_dsps.config.load_config": lambda p: {"amortized": {"encoder": {}}},
            "euclid_dsps.amortized.train.load_checkpoint": lambda *a: model,
            "euclid_dsps.amortized.train.build_amortized_model": build,
            "euclid_dsps.amortized.adaptive_smc_trainer.prepare_adaptive_training_runtime": lambda *a, **kw: (
                rt
            ),
            "euclid_dsps.amortized.posterior_target.posterior_log_target": target,
            "euclid_dsps.amortized.latent.x_to_theta": lambda x, spec: x,
        }.items():
            stack.enter_context(patch(name, value))
        for task in (0, 1, 2):
            if task:
                arm = ARMS[task - 1]
                d = training / "arms" / arm.name
                d.mkdir(parents=True)
                candidate = runner.initialize_candidate(
                    model, {"amortized": {"encoder": {}}}, None, arm, 3
                )
                eqx.tree_serialise_leaves(d / "encoder.eqx", candidate)
                runner.write(
                    d / "FINAL.json",
                    dict(
                        prior_frozen_sha256=runner.tree_digest(
                            (model.prior, model.sed_scale, model.band_calibration)
                        )
                    ),
                )
            infer(root, task, platform="cpu")
            dest = root / "arms" / contract["arms"][task]
            assert runner.read(dest / "FINAL.json")["objects"] == 5
            assert len(pd.read_csv(dest / "metrics.csv")) == 10
            for replica in range(2):
                raw = pd.read_parquet(dest / f"raw_{replica}.parquet")
                weighted = pd.read_parquet(dest / f"is_{replica}.parquet")
                assert len(raw) == len(weighted) == 80
                assert not raw.duplicated(["object_id", "sample_id"]).any()
                assert raw.object_id.nunique() == 5
                bank = pd.concat(
                    pd.read_parquet(p)
                    for p in (dest / f"bank_{replica}").glob("*.parquet")
                )
                np.testing.assert_allclose(bank.groupby("object_id").weight.sum(), 1.0)
    print("INFERENCE_TEST_PASS")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
