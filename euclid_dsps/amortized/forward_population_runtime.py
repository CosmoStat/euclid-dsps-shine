"""Thin DSPS adapter for forward-only population learning; no E-step is run."""

from __future__ import annotations

import copy
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.config import load_config

from .forward_population import PhysicalBasis, simplex


class PopulationPrior(eqx.Module):
    """Exact normalized latent-x mixture; physical density uses the usual Jacobian."""

    means: jax.Array
    scales: jax.Array
    weights: jax.Array

    def __init__(self, basis: PhysicalBasis, weights):
        means = np.zeros((basis.components, 15))
        scales = np.ones_like(means)
        means[:, basis.indices], scales[:, basis.indices] = basis.centers, basis.scales
        self.means, self.scales = jnp.asarray(means), jnp.asarray(scales)
        self.weights = jnp.asarray(simplex(weights))
        if self.weights.shape != (basis.components,):
            raise ValueError("weights must match basis")

    def log_prob(self, x):
        terms = -0.5 * jnp.sum(
            ((x[..., None, :] - self.means) / self.scales) ** 2
            + 2 * jnp.log(self.scales)
            + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        return jax.scipy.special.logsumexp(terms + jnp.log(self.weights), axis=-1)

    def sample(self, key, n_samples):
        label_key, normal_key = jax.random.split(key)
        labels = jax.random.categorical(
            label_key, jnp.log(self.weights), shape=(n_samples,)
        )
        return self.means[labels] + self.scales[labels] * jax.random.normal(
            normal_key, (n_samples, 15), dtype=self.means.dtype
        )

    def log_prob_theta(self, theta, latent_spec):
        from .latent import (
            _latent_scale,
            network_x_to_raw_x,
            theta_to_x,
            x_to_theta_log_abs_det_jacobian,
        )

        x = theta_to_x(theta, latent_spec)
        inside = jnp.all(
            (theta > latent_spec.lower) & (theta < latent_spec.upper), axis=-1
        )
        if latent_spec.normalization == "bounded_mixed_warp":
            jacobian = x_to_theta_log_abs_det_jacobian(x, latent_spec)
        elif latent_spec.normalization in {"identity", "standardized_logit"}:
            raw = network_x_to_raw_x(x, latent_spec)
            jacobian = jnp.sum(
                jnp.log(_latent_scale(latent_spec))
                + jnp.log(latent_spec.upper - latent_spec.lower)
                + jax.nn.log_sigmoid(raw)
                + jax.nn.log_sigmoid(-raw),
                axis=-1,
            )
        else:
            raise ValueError("unsupported physical-density transform")
        density = self.log_prob(x) - jacobian
        return jnp.where(inside, density, -jnp.inf)


def load_forward_runtime(source: Path, destination: Path, settings):
    from scripts.feniks_avi_experiments import read

    from .adaptive_smc_trainer import prepare_adaptive_training_runtime
    from .train import _sleep_runtime_config, load_checkpoint

    manifest = read(source / "MANIFEST.json")
    config = load_config(source / "source_config.yaml")
    model = load_checkpoint(manifest["source"]["checkpoint"], config)
    runtime = prepare_adaptive_training_runtime(
        config,
        destination,
        train_indices_file=source / "train.npy",
        validation_indices_file=source / "validation.npy",
        validation_catalog_path=manifest["validation_catalog"],
        fixed_feature_stats_path=manifest["source"]["feature_stats"],
        train_population_prior=False,
    )
    # Historical runtime requires observed_catalog sleep; resolve the forward
    # observation law explicitly afterwards, without changing that baseline.
    forward_config = copy.deepcopy(config)
    sleep = forward_config["amortized"]["objective"]["sleep"]
    sleep["error_model"] = settings["observation"]["error_model"]
    # Survey simulation is Gaussian m5. The old robust Student-t fitting
    # likelihood is not the catalogue's generative noise law.
    sleep["noise_family"] = settings["observation"]["noise_family"]
    sleep["selection"] = dict(enabled=True, band="lsst_r", max_mag_ab=settings["cut"])
    observation = _sleep_runtime_config(forward_config, runtime.feature_stats)
    observation["noise_family"] = settings["observation"]["noise_family"]
    if settings["observation"]["mask_model"] == "all_observed":
        if not np.all(runtime.train_arrays.mask) or not np.all(
            runtime.validation_arrays.mask
        ):
            raise ValueError(
                "catalogue has missing bands: supply a justified parent mask law"
            )
    else:
        raise ValueError("only verified all-observed masks supported in this first run")
    if settings["observation"]["error_model"] != "m5_depth":
        raise ValueError(
            "main run requires generative m5 errors, not selected-catalogue error recycling"
        )
    return model, runtime, observation, config


def simulator(model, runtime, observation):
    from .decoder import model_flux_from_x
    from .features import make_encoder_features
    from .latent import x_to_theta
    from .posterior_target import _apply_model_calibration, safe_decoder_inputs
    from .train import (
        _sample_sleep_noise,
        _sleep_m5_flux_error,
        _sleep_observed_selection_mask,
    )

    @eqx.filter_jit
    def simulate(x, key):
        safe, valid = safe_decoder_inputs(x, runtime.latent_spec)
        flux = model_flux_from_x(
            safe,
            runtime.latent_spec,
            runtime.context,
            runtime.model_args,
            runtime.parameter_names,
        )
        flux = _apply_model_calibration(model, flux, runtime.calibration_config)
        errors = _sleep_m5_flux_error(flux, observation)
        noise, _ = _sample_sleep_noise(
            key, errors, sleep=observation, likelihood_config=runtime.likelihood_config
        )
        noisy = flux + noise
        mask = jnp.ones_like(noisy, dtype=bool)
        valid &= jnp.all(
            jnp.isfinite(noisy) & jnp.isfinite(errors) & (errors > 0), axis=-1
        )
        selected = _sleep_observed_selection_mask(
            noisy,
            valid,
            band_index=observation["selection_band_index"],
            flux_min=observation["selection_flux_min_fnu_cgs"],
            observed_mask=mask,
        )
        return dict(
            x=x,
            theta=x_to_theta(x, runtime.latent_spec),
            flux=noisy,
            errors=errors,
            mask=mask,
            valid=valid,
            selected=selected,
            features=make_encoder_features(noisy, errors, runtime.feature_stats, mask),
        )

    return simulate


def posterior_template(model, runtime, config, settings):
    from scripts.feniks_avi_experiments import initialize_candidate

    from .avi_experiments import Arm

    config = copy.deepcopy(config)
    encoder = config["amortized"]["encoder"]
    encoder.update(settings["posterior"]["architecture"])
    encoder["flow_output_space"] = "latent_x"
    encoder["transport_float64"] = True
    candidate = initialize_candidate(
        model,
        config,
        runtime.latent_spec,
        Arm("forward_sleep", scratch=True, experts=settings["posterior"]["experts"]),
        settings["seed"],
    )
    return candidate
