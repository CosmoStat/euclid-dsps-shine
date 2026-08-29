from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.amortized.features import FeatureStats
from euclid_dsps.amortized.selection_correction import (
    estimate_log_alpha_reparameterized,
    estimate_log_alpha_score_function,
    estimate_log_alpha_score_function_diagnostic,
    observed_flux_selection_beta,
    observed_flux_selection_log_beta,
    observed_flux_selection_log_beta_gaussian_m5,
    observed_magnitude_flux_limit_jax,
)
from euclid_dsps.amortized.train import _selection_correction_runtime_config
from euclid_dsps.config import load_config
from euclid_dsps.photometric_uncertainty import (
    flux_error_from_model,
    m5_depth_flux_error_jax,
)
from euclid_dsps.photometry import abmag_to_fnu_cgs

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"


def test_m5_depth_flux_error_jax_matches_numpy() -> None:
    flux = np.asarray([0.0, -2.0e-31, 3.0e-30, 1.0e-29], dtype=np.float32)
    model = {
        "type": "m5_depth",
        "m5": {"lsst_r": 27.5},
        "gamma": {"lsst_r": 0.039},
        "sigma_sys_mag": 0.005,
        "min_sigma_fnu_cgs": 1.0e-40,
    }
    expected = flux_error_from_model(flux, model, band_name="lsst_r")
    actual = m5_depth_flux_error_jax(
        jnp.asarray(flux),
        27.5,
        0.039,
        sigma_sys_mag=0.005,
        min_sigma_fnu_cgs=1.0e-40,
    )
    compiled = jax.jit(
        lambda value: m5_depth_flux_error_jax(
            value,
            27.5,
            0.039,
            sigma_sys_mag=0.005,
            min_sigma_fnu_cgs=1.0e-40,
        )
    )(jnp.asarray(flux))
    assert np.allclose(np.asarray(actual), expected, rtol=2.0e-6, atol=0.0)
    assert np.allclose(np.asarray(compiled), expected, rtol=2.0e-6, atol=0.0)


def test_observed_flux_selection_beta_analytic_limits() -> None:
    flux_limit = jnp.asarray(3.0)
    sigma = jnp.asarray([0.2, 0.2, 0.2])
    model_flux = jnp.asarray([3.0, 5.0, 1.0])
    beta = observed_flux_selection_beta(model_flux, sigma, flux_limit)
    log_beta = observed_flux_selection_log_beta(model_flux, sigma, flux_limit)
    assert np.isclose(float(beta[0]), 0.5, atol=1.0e-7)
    assert float(beta[1]) > 1.0 - 1.0e-7
    assert float(beta[2]) < 1.0e-20
    assert np.all(np.isfinite(np.asarray(log_beta)))


def test_observed_flux_selection_beta_matches_gaussian_monte_carlo() -> None:
    model_flux = jnp.asarray(0.7)
    sigma = jnp.asarray(0.4)
    flux_limit = jnp.asarray(1.0)
    analytic = observed_flux_selection_beta(model_flux, sigma, flux_limit)
    noise = sigma * jax.random.normal(jax.random.PRNGKey(5), (200_000,))
    empirical = jnp.mean((model_flux + noise > flux_limit).astype(jnp.float32))
    assert np.isclose(float(empirical), float(analytic), atol=3.0e-3)


def test_fused_gaussian_m5_selection_matches_separate_photoerr_path() -> None:
    flux = jnp.asarray(
        [0.0, 1.0e-31, 1.0e-30, 3.631e-30, 1.0e-29],
        dtype=jnp.float32,
    )
    limit = observed_magnitude_flux_limit_jax(25.0)
    error = m5_depth_flux_error_jax(
        flux,
        27.5,
        0.039,
        sigma_sys_mag=0.005,
        min_sigma_fnu_cgs=1.0e-40,
    )
    separate = observed_flux_selection_log_beta(flux, error, limit)
    fused = observed_flux_selection_log_beta_gaussian_m5(
        flux,
        limit,
        27.5,
        0.039,
        sigma_sys_mag=0.005,
        min_sigma_fnu_cgs=1.0e-40,
    )
    assert np.allclose(np.asarray(fused), np.asarray(separate), rtol=2.0e-5)


def test_fused_gaussian_m5_selection_has_finite_cgs_scale_gradient() -> None:
    limit = observed_magnitude_flux_limit_jax(25.0)

    def objective(log_flux_ratio):
        flux = limit * jnp.exp(log_flux_ratio)
        return observed_flux_selection_log_beta_gaussian_m5(
            flux,
            limit,
            27.5,
            0.039,
            sigma_sys_mag=0.005,
            min_sigma_fnu_cgs=1.0e-40,
        )

    points = jnp.asarray([-30.0, -10.0, -2.0, 0.0, 2.0, 10.0])
    values = jax.vmap(objective)(points)
    gradients = jax.vmap(jax.grad(objective))(points)
    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(jnp.isfinite(gradients))


def test_fused_gaussian_m5_selection_handles_extreme_finite_flux() -> None:
    limit = observed_magnitude_flux_limit_jax(25.0)
    flux = jnp.asarray([0.0, 1.0e-35, limit, 1.0e-20, 1.0e30])
    values = observed_flux_selection_log_beta_gaussian_m5(
        flux,
        limit,
        27.5,
        0.039,
        sigma_sys_mag=0.005,
        min_sigma_fnu_cgs=1.0e-40,
    )
    gradient = jax.grad(
        lambda shift: jnp.sum(
            observed_flux_selection_log_beta_gaussian_m5(
                flux * jnp.exp(shift),
                limit,
                27.5,
                0.039,
                sigma_sys_mag=0.005,
                min_sigma_fnu_cgs=1.0e-40,
            )
        )
    )(jnp.asarray(0.0))
    assert jnp.all(jnp.isfinite(values))
    assert jnp.isfinite(gradient)


def test_fused_gaussian_m5_log_alpha_has_finite_nonzero_shift_gradient() -> None:
    limit = observed_magnitude_flux_limit_jax(25.0)
    base = jnp.linspace(-4.0, 4.0, 512)

    def log_alpha(shift):
        flux = limit * jnp.exp(base + shift)
        log_beta = observed_flux_selection_log_beta_gaussian_m5(
            flux,
            limit,
            27.5,
            0.039,
            sigma_sys_mag=0.005,
        )
        return jax.scipy.special.logsumexp(log_beta) - jnp.log(base.size)

    value, gradient = jax.value_and_grad(log_alpha)(jnp.asarray(0.0))
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    assert gradient > 0.0


def test_log_alpha_gradient_is_finite_nonzero_and_matches_finite_difference() -> None:
    class ShiftPrior:
        latent_dim = 1

        def __init__(self, shift):
            self.shift = shift

        def forward(self, base):
            value = base + self.shift
            return value, jnp.zeros(value.shape[:-1], dtype=value.dtype)

    key = jax.random.PRNGKey(11)

    def objective(shift):
        prior = ShiftPrior(shift)
        log_alpha, _metrics = estimate_log_alpha_reparameterized(
            prior,
            key,
            n_prior_samples=32_768,
            log_beta_fn=lambda value: observed_flux_selection_log_beta(
                value[:, 0],
                jnp.asarray(0.7),
                jnp.asarray(0.2),
            ),
        )
        return log_alpha

    shift = jnp.asarray(0.1)
    gradient = jax.grad(objective)(shift)
    step = 2.0e-3
    finite_difference = (objective(shift + step) - objective(shift - step)) / (
        2.0 * step
    )
    assert np.isfinite(float(gradient))
    assert abs(float(gradient)) > 1.0e-3
    assert np.isclose(float(gradient), float(finite_difference), rtol=2.0e-2)


def test_selection_pathwise_score_function_and_finite_difference_agree() -> None:
    class ShiftNormalPrior:
        latent_dim = 1

        def __init__(self, shift):
            self.shift = shift

        def forward(self, base):
            value = base + self.shift
            return value, jnp.zeros(value.shape[:-1], dtype=value.dtype)

        def log_prob(self, value):
            residual = value - self.shift
            return -0.5 * (jnp.square(residual[:, 0]) + jnp.log(2.0 * jnp.pi))

    key = jax.random.PRNGKey(211)
    samples = 65_536

    def pathwise(shift):
        value, _ = estimate_log_alpha_reparameterized(
            ShiftNormalPrior(shift),
            key,
            n_prior_samples=samples,
            prior_sample_batch_size=samples,
            log_beta_fn=lambda latent: observed_flux_selection_log_beta(
                latent[:, 0], jnp.asarray(0.7), jnp.asarray(0.2)
            ),
        )
        return value

    shift = jnp.asarray(0.1)
    pathwise_gradient = jax.grad(pathwise)(shift)
    step = 2.0e-3
    finite_difference = (pathwise(shift + step) - pathwise(shift - step)) / (2.0 * step)

    def score_surrogate(value):
        _score_value, surrogate, _metrics = (
            estimate_log_alpha_score_function_diagnostic(
                ShiftNormalPrior(value),
                key,
                n_prior_samples=samples,
                log_beta_fn=lambda latent: observed_flux_selection_log_beta(
                    latent[:, 0], jnp.asarray(0.7), jnp.asarray(0.2)
                ),
            )
        )
        return surrogate

    score_shift_gradient = jax.grad(score_surrogate)(shift)

    assert jnp.isfinite(pathwise_gradient)
    assert jnp.isfinite(score_shift_gradient)
    assert np.isclose(float(pathwise_gradient), float(finite_difference), rtol=2.0e-2)
    assert np.isclose(
        float(score_shift_gradient), float(pathwise_gradient), rtol=3.0e-2
    )


def test_score_function_objective_preserves_log_alpha_value_and_gradient() -> None:
    class ShiftNormalPrior:
        latent_dim = 1

        def __init__(self, shift):
            self.shift = shift

        def forward(self, base):
            value = base + self.shift
            return value, jnp.zeros(value.shape[:-1], dtype=value.dtype)

        def log_prob(self, value):
            residual = value - self.shift
            return -0.5 * (jnp.square(residual[:, 0]) + jnp.log(2.0 * jnp.pi))

    key = jax.random.PRNGKey(221)
    samples = 65_536

    def pathwise(shift):
        value, _ = estimate_log_alpha_reparameterized(
            ShiftNormalPrior(shift),
            key,
            n_prior_samples=samples,
            prior_sample_batch_size=samples,
            log_beta_fn=lambda latent: observed_flux_selection_log_beta(
                latent[:, 0], jnp.asarray(0.7), jnp.asarray(0.2)
            ),
        )
        return value

    def score(shift):
        return estimate_log_alpha_score_function(
            ShiftNormalPrior(shift),
            key,
            n_prior_samples=samples,
            log_beta_fn=lambda latent: observed_flux_selection_log_beta(
                latent[:, 0], jnp.asarray(0.7), jnp.asarray(0.2)
            ),
        )

    point = jnp.asarray(0.1)
    score_value, metrics = score(point)
    score_gradient = jax.grad(lambda shift: score(shift)[0])(point)
    pathwise_value = pathwise(point)
    pathwise_gradient = jax.grad(pathwise)(point)

    assert float(metrics["selection/gradient_estimator_score_function"]) == 1.0
    assert np.isclose(float(score_value), float(pathwise_value), rtol=1.0e-6)
    assert np.isfinite(float(score_gradient))
    assert np.isclose(float(score_gradient), float(pathwise_gradient), rtol=3.0e-2)


def test_identity_realnvp_selection_gradient_matches_finite_difference() -> None:
    import equinox as eqx

    from euclid_dsps.amortized.flows import RealNVPPrior

    prior = RealNVPPrior(
        jax.random.PRNGKey(301),
        latent_dim=2,
        n_layers=2,
        hidden_size=8,
        permutation="roll",
        init="identity",
    )
    key = jax.random.PRNGKey(302)
    flux_limit = observed_magnitude_flux_limit_jax(25.0)

    def objective(raw_shift):
        bias = prior.layers[0].shift_net.layers[-1].bias.at[0].set(raw_shift)
        candidate = eqx.tree_at(
            lambda value: value.layers[0].shift_net.layers[-1].bias,
            prior,
            bias,
        )
        value, _ = estimate_log_alpha_reparameterized(
            candidate,
            key,
            n_prior_samples=32_768,
            prior_sample_batch_size=32_768,
            log_beta_fn=lambda latent: observed_flux_selection_log_beta_gaussian_m5(
                flux_limit * jnp.exp(latent[:, 1]),
                flux_limit,
                27.5,
                0.039,
                sigma_sys_mag=0.005,
                min_sigma_fnu_cgs=1.0e-40,
            ),
        )
        return value

    def score_objective(raw_shift):
        bias = prior.layers[0].shift_net.layers[-1].bias.at[0].set(raw_shift)
        candidate = eqx.tree_at(
            lambda value: value.layers[0].shift_net.layers[-1].bias,
            prior,
            bias,
        )
        value, _ = estimate_log_alpha_score_function(
            candidate,
            key,
            n_prior_samples=32_768,
            prior_sample_batch_size=2048,
            log_beta_fn=lambda latent: observed_flux_selection_log_beta_gaussian_m5(
                flux_limit * jnp.exp(latent[:, 1]),
                flux_limit,
                27.5,
                0.039,
                sigma_sys_mag=0.005,
                min_sigma_fnu_cgs=1.0e-40,
            ),
        )
        return value

    point = jnp.asarray(0.03)
    gradient = jax.grad(objective)(point)
    score_gradient = jax.grad(score_objective)(point)
    step = 2.0e-3
    finite_difference = (objective(point + step) - objective(point - step)) / (
        2.0 * step
    )
    assert jnp.isfinite(gradient)
    assert jnp.isfinite(score_gradient)
    assert np.isclose(float(gradient), float(finite_difference), rtol=3.0e-2)
    assert np.isclose(float(score_gradient), float(gradient), rtol=5.0e-2)


def test_chunked_log_alpha_matches_single_batch_and_preserves_sample_count() -> None:
    class IdentityPrior:
        latent_dim = 2

        @staticmethod
        def forward(base):
            return base, jnp.zeros(base.shape[:-1], dtype=base.dtype)

    prior = IdentityPrior()
    key = jax.random.PRNGKey(13)
    kwargs = {
        "prior": prior,
        "key": key,
        "n_prior_samples": 1024,
        "log_beta_fn": lambda value: observed_flux_selection_log_beta(
            value[:, 0],
            jnp.asarray(0.8),
            jnp.asarray(0.1),
        ),
    }
    single, _single_metrics = estimate_log_alpha_reparameterized(
        **kwargs,
        prior_sample_batch_size=1024,
    )
    chunked, chunked_metrics = estimate_log_alpha_reparameterized(
        **kwargs,
        prior_sample_batch_size=64,
    )
    assert jnp.allclose(chunked, single, atol=1.0e-7)
    assert chunked_metrics["selection/n_prior_samples"] == 1024
    assert chunked_metrics["selection/prior_sample_batch_size"] == 64
    padded, padded_metrics = estimate_log_alpha_reparameterized(
        **{**kwargs, "n_prior_samples": 1000},
        prior_sample_batch_size=64,
    )
    assert jnp.isfinite(padded)
    assert padded_metrics["selection/n_prior_samples"] == 1000


def test_score_function_caps_working_batch_and_handles_padding() -> None:
    class ShiftNormalPrior:
        latent_dim = 2

        def __init__(self, shift):
            self.shift = shift

        def forward(self, base):
            value = base + self.shift
            return value, jnp.zeros(value.shape[:-1], dtype=value.dtype)

        def log_prob(self, value):
            residual = value - self.shift
            return -0.5 * jnp.sum(jnp.square(residual), axis=-1)

    key = jax.random.PRNGKey(1401)

    def evaluate(shift, requested_batch_size):
        return estimate_log_alpha_score_function(
            ShiftNormalPrior(shift),
            key,
            n_prior_samples=130,
            prior_sample_batch_size=requested_batch_size,
            log_beta_fn=lambda value: observed_flux_selection_log_beta(
                value[:, 0],
                jnp.asarray(0.7),
                jnp.asarray(0.2),
            ),
        )

    point = jnp.asarray(0.1)
    capped_value, capped_metrics = evaluate(point, 4096)
    explicit_value, explicit_metrics = evaluate(point, 64)
    capped_gradient = jax.grad(lambda shift: evaluate(shift, 4096)[0])(point)
    explicit_gradient = jax.grad(lambda shift: evaluate(shift, 64)[0])(point)

    assert capped_metrics["selection/n_prior_samples"] == 130
    assert capped_metrics["selection/prior_sample_batch_size"] == 64
    assert explicit_metrics["selection/prior_sample_batch_size"] == 64
    assert jnp.allclose(capped_value, explicit_value, atol=1.0e-7)
    assert jnp.allclose(capped_gradient, explicit_gradient, atol=1.0e-7)
    assert jnp.isfinite(capped_gradient)


def test_observed_magnitude_limit_uses_shared_ab_flux_convention() -> None:
    expected = float(np.asarray(abmag_to_fnu_cgs(25.0)))
    actual = float(np.asarray(observed_magnitude_flux_limit_jax(25.0)))
    assert np.isclose(actual, expected, rtol=1.0e-7)


def test_selection_correction_defaults_to_disabled() -> None:
    cfg = amortized_config({})
    selection = cfg["objective"]["selection_correction"]
    assert selection["enabled"] is False
    assert selection["kind"] == "observed_magnitude_limit"
    assert selection["n_prior_samples"] == 4096
    assert selection["prior_sample_batch_size"] == 64
    assert selection["gradient_preflight_samples"] == 0


def test_selection_aware_feniks_config_separates_likelihood_and_survey_noise() -> None:
    config = load_config(CONFIG_DIR / "feniks_selfsup_selection_r25_rws_mix2_t2.yaml")
    cfg = amortized_config(config)
    selection = cfg["objective"]["selection_correction"]
    assert cfg["likelihood"]["type"] == "student_t"
    assert cfg["objective"]["sleep"]["noise_family"] == "gaussian"
    assert cfg["objective"]["sleep"]["selection"]["band"] == "lsst_r"
    assert cfg["objective"]["wake"]["train_prior"] is True
    assert cfg["encoder"]["base_components"] == 2
    assert selection["enabled"] is True
    assert selection["survey_noise"] == "gaussian_m5"
    assert selection["prior_sample_batch_size"] == 64
    assert config["synthetic_diffsky"]["selection"]["observed_magnitude_limits"] == {
        "lsst_r": 25.0
    }


def test_adaptive_smc_runtime_preserves_score_function_estimator() -> None:
    config = load_config(
        CONFIG_DIR / "feniks_selfsup_adaptive_smcwake_parentprior_selection_r25.yaml"
    )
    stats = FeatureStats(
        flux_scale=np.ones(1),
        err_scale=np.ones(1),
        band_names=("lsst_r",),
    )
    runtime = _selection_correction_runtime_config(config, stats)

    assert runtime["gradient_estimator"] == "score_function"


def test_parentprior_sleepnpe_config_has_disjoint_q_and_prior_updates() -> None:
    config = load_config(
        CONFIG_DIR
        / "feniks_selfsup_parentprior_sleepnpe_defensivewake_selection_r25.yaml"
    )
    cfg = amortized_config(config)
    sleep = cfg["objective"]["sleep"]
    wake = cfg["objective"]["wake"]
    assert cfg["encoder"]["flow_family"] == "realnvp"
    assert cfg["encoder"]["flow_output_space"] == "latent_x"
    assert cfg["encoder"]["log_std_min"] == -4.0
    assert sleep["error_model"] == "observed_catalog"
    assert sleep["selection"]["max_mag_ab"] == 25.0
    assert wake["start_encoder_epoch"] == 25
    assert wake["every_encoder_epochs"] == 8
    assert wake["train_encoder"] is False
    assert wake["train_prior"] is True
    assert wake["support_gate_enabled"] is True
    assert [item["source"] for item in wake["proposal"]["components"]] == [
        "posterior",
        "posterior",
        "posterior",
        "prior",
    ]
    assert cfg["training"]["best_checkpoint_metric"] == "validation_sleep_nll"
    assert cfg["objective"]["selection_correction"]["enabled"] is True
    assert cfg["objective"]["selection_correction"]["n_prior_samples"] == 1024
    assert cfg["objective"]["selection_correction"]["prior_sample_batch_size"] == 256
    assert cfg["objective"]["selection_correction"]["gradient_preflight_samples"] == 64
