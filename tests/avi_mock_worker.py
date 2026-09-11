"""Four-CPU-device integration check; only physical decoder/I/O are mocked."""

import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.avi_experiments import ARMS
from euclid_dsps.amortized.elbo import AmortizedModel
from euclid_dsps.amortized.features import FeatureStats
from euclid_dsps.amortized.flows import StandardNormalPrior
from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
from euclid_dsps.amortized.posterior_target import PosteriorTargetValues
from euclid_dsps.calibration import GlobalSedScaleState
from scripts import feniks_avi_experiments as runner


def build(config, key, latent_spec=None):
    return AmortizedModel(
        encoder=ConditionalFlowEncoder(
            key,
            input_dim=3,
            latent_dim=2,
            hidden_sizes=(4,),
            context_encoder_type="residual_photometry",
            residual_trunk_width=4,
            residual_blocks=1,
            residual_representation_width=4,
            residual_context_dim=4,
            activation="gelu",
            log_std_min=-6.0,
            log_std_max=2.0,
            initial_log_std=-1.0,
            family="realnvp",
            n_layers=1,
            hidden_size=4,
            init_scale=0.1,
            output_space="latent_x",
            transport_float64=True,
        ),
        prior=StandardNormalPrior(latent_dim=2),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.array(0.0)),
    )


def target(model, x, b, *args):
    flux = x[..., :1]
    ll = -0.5 * jnp.sum((flux - b.flux) ** 2, axis=-1)
    prior = model.prior.log_prob(x)
    return PosteriorTargetValues(
        ll + prior, ll, prior, jnp.ones(ll.shape, bool), flux, flux
    )


def sleep(
    model,
    b,
    spec,
    context,
    model_args,
    names,
    key,
    likelihood,
    calibration,
    objective,
    log_prob_fn,
):
    x = jax.lax.stop_gradient(model.prior.sample(key, len(b.flux)))
    return -jnp.mean(log_prob_fn(b.features, x)), {"finite_fraction": jnp.array(1.0)}


def main(root):
    root.mkdir()
    m = dict(
        gpus=4,
        train_rows=256,
        validation_rows=8,
        global_batch=256,
        epochs=9,
        learning_rate=2e-5,
        warmup_fraction=0.05,
        seed=3,
        particles=32,
        validation_particles=16,
        validation_catalog="none",
        bootstrap_epochs=6,
        cycle=["sleep", "sleep", "wake"],
        source={"checkpoint": "none", "feature_stats": "none"},
    )
    runner.write(root / "MANIFEST.json", m)
    np.save(root / "train.npy", np.arange(256))
    np.save(root / "validation.npy", np.arange(8))
    np.savez(
        root / "teachers.npz",
        x=np.random.default_rng(3).normal(size=(2, 256, 2)),
        flux=np.ones((2, 1), np.float32),
        flux_err=np.ones((2, 1), np.float32),
        mask=np.ones((2, 1), bool),
        rows=np.array([0, 1]),
    )
    arrays = SimpleNamespace(
        flux=np.ones((256, 1), np.float32),
        flux_err=np.ones((256, 1), np.float32),
        mask=np.ones((256, 1), bool),
        row_index=np.arange(256),
    )
    val = SimpleNamespace(
        flux=arrays.flux[:8], flux_err=arrays.flux_err[:8], mask=arrays.mask[:8]
    )
    rt = SimpleNamespace(
        latent_spec=None,
        context=None,
        model_args=None,
        parameter_names=("a", "b"),
        train_arrays=arrays,
        validation_arrays=val,
        feature_stats=FeatureStats(np.ones(1), np.ones(1), ("f",), append_mask=True),
        likelihood_config={"type": "gaussian"},
        calibration_config={},
        sleep_objective_config={},
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner, "frozen_selection_normalization", lambda *a: -0.2))
        stack.enter_context(patch.object(runner, "check_inputs", lambda root: m))
        stack.enter_context(
            patch(
                "euclid_dsps.config.load_config",
                lambda path: {"amortized": {"encoder": {}}},
            )
        )
        stack.enter_context(
            patch(
                "euclid_dsps.amortized.train.load_checkpoint",
                lambda *args: build({}, jax.random.PRNGKey(1)),
            )
        )
        stack.enter_context(
            patch("euclid_dsps.amortized.train.build_amortized_model", build)
        )
        stack.enter_context(
            patch("euclid_dsps.amortized.train._model_generated_sleep_loss", sleep)
        )
        stack.enter_context(
            patch(
                "euclid_dsps.amortized.adaptive_smc_trainer.prepare_adaptive_training_runtime",
                lambda *a, **k: rt,
            )
        )
        stack.enter_context(
            patch("euclid_dsps.amortized.posterior_target.posterior_log_target", target)
        )
        for task in range(7):
            args = SimpleNamespace(
                root=root, task=task, mode="preflight", max_hours=1.0
            )
            runner.run(args, required_platform="cpu")
            assert (
                runner.read(root / "preflight" / ARMS[task].name / "FINAL.json")[
                    "status"
                ]
                == "PREFLIGHT_PASS"
            )
        args = SimpleNamespace(root=root, task=0, mode="train", max_hours=-1.0)
        runner.run(args, required_platform="cpu")
        assert (root / "arms" / ARMS[0].name / "RESUME.json").is_file()
        args.max_hours = 1.0
        runner.run(args, required_platform="cpu")
        assert runner.read(root / "arms" / ARMS[0].name / "FINAL.json")["steps"] == 9
        import pandas as pd

        validation = pd.read_csv(root / "arms" / ARMS[0].name / "validation_final.csv")
        np.testing.assert_allclose(validation.selection_log_alpha, -0.2)
        np.testing.assert_allclose(
            validation.negative_elbo - validation.negative_elbo_unselected, -0.2,
        )
        runner.summarize(root)
        assert (root / "avi_comparison.png").is_file()
        from PIL import Image

        assert np.asarray(Image.open(root / "avi_comparison.png")).std() > 10


if __name__ == "__main__":
    main(Path(sys.argv[1]))
