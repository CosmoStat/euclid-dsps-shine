Amortized Inference
===================

This feature is the photometric posterior-inference path. It trains an
amortized encoder against the fixed DSPS decoder and can use either a standard
normal prior, a supervised truth-trained RealNVP prior, or a RealNVP prior
trained jointly with the encoder.

The public configs are:

.. code-block:: text

   configs/amortized_fs2_realnvp.yaml
   configs/amortized_diffsky_hltds_standard_normal_gpu.yaml
   configs/amortized_diffsky_hltds_supervised_prior_gpu.yaml
   configs/amortized_diffsky_hltds_joint_realnvp_gpu.yaml

FS2 remains the Euclid comparison path. Diffsky HLTDS 04/14 is the main
science-validation path because it has photometry plus physical truth columns.
Do not treat a good photometric fit as physical recovery: physical claims also
require same-parameter forward closure, supervised prior-vs-truth diagnostics,
posterior calibration, and comparison of derived quantities.

Model Contract
--------------

The amortized feature builder consumes the configured band set. FS2 uses ten
bands:

.. code-block:: text

   LSST u,g,r,i,z,y + Euclid VIS,Y,J,H

The encoder input is:

.. code-block:: text

   [flux_1, ..., flux_B, err_1, ..., err_B]

after per-band normalization. FS2 uses ``B=10`` and feature dimension 20.
Diffsky HLTDS uses fourteen bands:

.. code-block:: text

   LSST u,g,r,i,z,y + Roman F062,F087,F106,F129,F146,F158,F184,F213

so its encoder feature dimension is 28. The feature code is generic in the
number of bands; the config controls ``encoder.input_dim`` and the expected
band count.

Fluxes are normalized with a robust signed transform,
``asinh(flux / flux_scale)``, so bright FS2 objects do not produce MLP inputs of
hundreds of scale units. Errors are normalized with
``log(err / err_scale + eps)``. The error terms are part of the input because
two objects with identical fluxes but different per-band uncertainties do not
carry the same information and should not receive the same posterior width.
For Diffsky HLTDS, ``flux_scale`` and ``err_scale`` are learned once from the
training catalog by ``compute_feature_stats`` as robust per-band scales and are
stored in ``feature_stats.json``. DSPS never receives these normalized feature
values; they are encoder inputs only.

The posterior model is:

.. code-block:: text

   flux_B + err_B
       -> q_psi(x | flux, err)
       -> theta = h(x)
       -> DSPS(theta)
       -> flux_model_B

``x`` is the network latent vector. ``theta`` is the bounded physical
parameter vector, including redshift. FS2 uses the 16-parameter PopCosmos-like
schema. The active Diffsky HLTDS path uses a 12-parameter PopCosmos-bin schema:

.. code-block:: text

   z_obs
   log10_stellar_mass
   dlog10_sfr_1
   dlog10_sfr_2
   dlog10_sfr_3
   dlog10_sfr_4
   dlog10_sfr_5
   dlog10_sfr_6
   log10_stellar_metallicity
   tau2
   dust_index_n
   tau1_over_tau2

The active HLTDS configs set ``amortized.latent.normalization`` to
``standardized_logit``. The encoder and RealNVP prior live in standardized
bounded-logit coordinates. Before calling DSPS, the decoder maps
``x_network -> x_raw_logit -> theta_physical`` using the configured parameter
bounds, centers, and scales. DSPS receives physical values such as redshift,
stellar mass, SFH ratios, metallicity, ``tau2``, ``dust_index_n``, and
``tau1_over_tau2``; it does not receive normalized latent coordinates. The
true-param closure and supervised-prior paths remain the same-parameter tests
for Diffstar/Diffmah generated truths. ``psi`` denotes the encoder parameters.
``beta`` denotes RealNVP prior parameters when a RealNVP prior is used.

Implementation Architecture
---------------------------

The implementation is split so that the neural inference code does not depend
on private MAP helpers:

.. code-block:: text

   euclid_dsps/parameter_vectors.py
       theta vectors -> DSPS parameter pytrees -> model_mags_jax_dynamic

   euclid_dsps/observation_arrays.py
       catalog rows -> object_id, flux[B], err[B], mask[B]

   euclid_dsps/amortized/features.py
       flux[B], err[B] -> normalized features[2B]

   euclid_dsps/amortized/encoder.py
       GaussianEncoder q_psi(x | flux, err)

   euclid_dsps/amortized/flows.py
       StandardNormalPrior or RealNVPPrior p_beta(x)

   euclid_dsps/amortized/decoder.py
       x -> theta -> fixed DSPS -> model_flux[B]

   euclid_dsps/amortized/elbo.py
       Student-t log likelihood + Monte Carlo KL

   euclid_dsps/amortized/train.py
       Optax training with frozen or jointly trained prior modes

The DSPS decoder is fixed. It is evaluated through
``model_mags_jax_dynamic`` and ``parameter_vectors.py`` so JAX gradients flow
from the flux likelihood back to ``x`` and the encoder, while DSPS asset arrays
remain decoder inputs rather than trainable weights.

Neural Components
-----------------

``GaussianEncoder`` is a simple Equinox MLP. The default architecture is:

.. code-block:: text

   input_dim = 20
   hidden_sizes = [256, 256, 256]
   activation = GELU
   mean_head -> R^16
   log_std_head -> R^16

The encoder posterior is diagonal Gaussian:

.. code-block:: text

   q_psi(x | flux, err) = N(mu_psi, diag(sigma_psi^2))

The prior source is configured by:

.. code-block:: yaml

   amortized:
     prior:
       source: standard_normal | supervised_checkpoint | joint_realnvp
       train_jointly: false | true

``standard_normal`` uses a fixed isotropic Gaussian in ``x``. ``supervised_checkpoint``
loads a RealNVP prior trained on truth parameters by the supervised prior
workflow and normally sets ``train_jointly: false``. ``joint_realnvp`` builds a
RealNVP prior and trains it with the encoder, matching the original behavior.

``RealNVPPrior`` is an exact-density normalizing flow in the same
unconstrained latent space ``x``:

.. code-block:: text

   base u ~ N(0, I)
   x = T_beta(u)
   n_layers = 8
   hidden_size = 128
   alternating affine coupling masks

When ``train_jointly`` is true, the encoder and RealNVP prior are optimized
together in the same ``eqx.filter_value_and_grad`` call and the same Optax
update. When it is false, prior gradients are zeroed before the optimizer
update. The training log records ``encoder_grad_norm`` and ``prior_grad_norm``
at every step; ``prior_grad_norm`` should be nonzero only for the joint-prior
mode when ``kl_weight > 0``.

ELBO
----

For one object, the training objective is the negative ELBO:

.. code-block:: text

   L_n = E_{x_n ~ q_psi} [
       -log p(f_n | x_n)
       + lambda_KL * (log q_psi(x_n | f_n, err_n) - log p_beta(x_n))
   ]

The implementation estimates the expectation by Monte Carlo:

.. code-block:: text

   eps_{k,n} ~ N(0, I)
   x_{k,n} = mu_psi(f_n, err_n) + sigma_psi(f_n, err_n) * eps_{k,n}

   loss = mean_{k,n} [
       -log p(f_n | x_{k,n})
       + lambda_KL * (logq - logp)
   ]

The default likelihood is a robust flux-space Student-t likelihood with
``nu = 2``. Gaussian likelihood is kept only for ablation.

Global SED Scale Nuisance Parameter
-----------------------------------

Diffsky science configs include one global decoder-calibration nuisance
parameter:

.. code-block:: yaml

   calibration:
     global_sed_scale:
       enabled: true
       mode: learn_global
       parameterization: log_alpha
       initial_log_alpha: 0.0
       prior_sigma_log_alpha: 0.10
       trainable: true
     per_band_zero_points:
       enabled: false

The trainable value is ``log_alpha_sed`` and ``alpha_sed = exp(log_alpha_sed)``.
It multiplies the model SED before filter integration. In code paths that only
expose integrated model photometry, the same factor is applied once to the
model flux; this is equivalent for a single wavelength-independent scale.

``alpha_sed`` is global to the run or dataset. It is not per galaxy, not per
band, not per filter, and not per posterior sample. It is intentionally not a
Pop-COSMOS zero-point correction and ``per_band_zero_points.enabled`` remains
false by default.

The amortized trainable components are:

.. code-block:: text

   standard_normal prior:      psi + log_alpha_sed
   supervised frozen prior:    psi + log_alpha_sed
   joint RealNVP prior:        psi + beta + log_alpha_sed

The ELBO adds the Gaussian prior penalty
``0.5 * (log_alpha_sed / prior_sigma_log_alpha)^2``. Training and inference
logs report ``log_alpha_sed``, ``alpha_sed``, ``delta_mag_global``,
``alpha_prior_penalty``, ``mean_model_flux_raw``, and
``mean_model_flux_scaled``.

``alpha_sed improves flexibility against global SED normalization mismatch,
but it does not correct color-dependent residuals. Since it is degenerate with
stellar mass, mass recovery must be reported both before and after alpha
correction.``

For mass diagnostics, posterior samples and summaries include:

.. code-block:: text

   log10_stellar_mass_raw
   log10_stellar_mass_alpha_corrected

with
``log10_stellar_mass_alpha_corrected = log10_stellar_mass_raw + log10(alpha_sed)``.
The raw mass remains the direct latent value; the corrected mass is the
photometrically constrained ``alpha_sed * Mstar`` combination.

Why The KL Is Monte Carlo
-------------------------

In the classic Gaussian VAE case:

.. code-block:: text

   q_psi(x|f) = N(mu, diag(sigma^2))
   p(x) = N(0, I)

the KL has the closed form:

.. code-block:: text

   KL(q || p) = 0.5 * sum_i(mu_i^2 + sigma_i^2 - 1 - log sigma_i^2)

This implementation does not use that prior. The prior is a RealNVP:

.. code-block:: text

   u ~ N(0, I)
   x = T_beta(u)
   g_beta = T_beta^{-1}

   log p_beta(x) = log p0(g_beta(x)) + log |det J_g_beta(x)|

For one RealNVP inverse coupling:

.. code-block:: text

   u_a = x_a
   u_b = (x_b - t_beta(x_a)) * exp(-s_beta(x_a))
   log |det du/dx| = -sum_j s_beta,j(x_a)

The KL contains expectations under ``q_psi`` of nonlinear MLP outputs such as
``s_beta(x_a)`` and ``t_beta(x_a)``. Those expectations have no general closed
form under a diagonal Gaussian encoder. The implementation therefore computes
``log q_psi(x|flux,err)`` exactly for each sampled point, computes
``log p_beta(x)`` exactly through the RealNVP inverse and triangular Jacobian,
and estimates ``E_q[logq - logp]`` by Monte Carlo.

Do not replace this with the closed-form Gaussian/Gaussian VAE KL.

Prior Sources
-------------

The Diffsky HLTDS configs separate the three prior experiments:

.. code-block:: text

   configs/amortized_diffsky_hltds_standard_normal_gpu.yaml
   configs/amortized_diffsky_hltds_supervised_prior_gpu.yaml
   configs/amortized_diffsky_hltds_joint_realnvp_gpu.yaml

The supervised checkpoint mode validates the checkpoint latent schema and
bounds against the active amortized schema before training starts. If they do
not match, the run fails explicitly instead of silently reinterpreting a truth
prior in an incompatible photometric latent space.

Training Outputs
----------------

Training writes progressive diagnostics as epochs complete:

.. code-block:: text

   training_log.csv
   training_progress.json
   training_summary.json
   feature_stats.json
   checkpoints/best.eqx
   checkpoints/last.eqx
   checkpoints/epoch_0001.eqx
   checkpoints/epoch_0002.eqx
   ...

``checkpoint_every`` controls the epoch checkpoint cadence. ``last.eqx`` is
updated during training, and ``best.eqx`` is updated whenever the observed loss
improves. Checkpoint sidecars include an architecture summary with encoder,
RealNVP prior, fixed DSPS decoder, and objective metadata.

When ``save_training_curves`` is enabled, diagnostics are regenerated every
``diagnostics_every`` epochs. The default diagnostic plots include:

.. code-block:: text

   loss.png
   negative_loglike.png
   kl_mc_mean.png
   logprior_mean.png
   logq_mean.png
   residual_rms.png
   finite_fraction.png
   encoder_grad_norm.png
   prior_grad_norm.png
   joint_grad_norm.png

The gradient plots are part of the prior-source contract: they make it visible
whether a joint RealNVP prior is receiving gradients, and whether a frozen
supervised prior remains frozen.

The ``amortized-train-fs2`` and ``amortized-train-diffsky`` CLIs are verbose by
default. They print the output directory, feature-stat computation,
filter/DSPS loading, JAX backend/devices, model architecture, epoch starts, and
epoch summaries. Each epoch also shows a progress bar with live ``loss``,
negative log likelihood, Monte Carlo KL, encoder gradient norm, and RealNVP
prior gradient norm. Use ``--quiet`` to reduce console logs and
``--no-progress`` to disable progress bars.

Diffsky HLTDS training examples:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_hltds_standard_normal_gpu.yaml \
     amortized-train-diffsky \
     --limit 10000 \
     --batch-size 64 \
     --epochs 10 \
     --n-samples 2 \
     --out outputs/runs/amortized_diffsky_hltds_standard_normal_n10000

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_hltds_supervised_prior_gpu.yaml \
     amortized-train-diffsky \
     --limit 10000 \
     --batch-size 64 \
     --epochs 10 \
     --n-samples 2 \
     --out outputs/runs/amortized_diffsky_hltds_supervised_prior_n10000

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_hltds_joint_realnvp_gpu.yaml \
     amortized-train-diffsky \
     --limit 10000 \
     --batch-size 64 \
     --epochs 10 \
     --n-samples 2 \
     --out outputs/runs/amortized_diffsky_hltds_joint_realnvp_n10000

For Diffsky, ``--batch-size`` is the requested catalog/training batch size. The
public config also sets ``amortized.training.jax_batch_size: 4`` to cap the
actual DSPS/JAX compiled object batch. This avoids very large CUDA compilations
and half-precision SSP coefficient paths that were observed to segfault before
the first batch on some JAX/CUDA installs.

Inference Outputs
-----------------

Inference loads ``checkpoints/best.eqx`` or ``checkpoints/last.eqx``, reuses the
saved ``feature_stats.json``, samples ``x`` from ``q_psi``, converts samples to
``theta``, evaluates ``logq``, ``logprior``, ``loglike``, and writes:

.. code-block:: text

   posterior_samples.parquet
   posterior_summary.parquet
   posterior_predictive_flux.parquet
   posterior_predictive_residuals.parquet
   posterior_predictive_residual_summary.parquet
   top_posterior_predictive_chi2.parquet
   top_posterior_predictive_chi2.csv
   feature_diagnostics.parquet
   redshift_comparison.parquet
   catalog_proxy_comparison.parquet
   redshift_pit.parquet
   learned_prior_samples.parquet
   learned_or_loaded_prior_samples.parquet
   prior_predictive_flux.parquet
   photoz_metrics.csv
   posterior_vs_truth_metrics.csv
   learned_prior_summary.json
   inference_summary.json

``prior_predictive_flux.parquet`` decodes ``theta ~ p_beta(theta)`` through the
fixed DSPS decoder and applies the configured global ``alpha_sed`` once. It
contains ``model_flux_raw_fnu_cgs`` and ``model_flux_scaled_fnu_cgs`` so the
prior physical distribution remains separate from the scaled prior-predictive
photometry.

The posterior predictive DSPS decode is chunked over the posterior-sample axis.
This is important on GPU: ``posterior_samples * batch_size`` physical forward
passes can otherwise be compiled as one large vmap. The default
``decoder_sample_chunk_size = 1`` mirrors the training memory pattern, while
still allowing many posterior samples to be written by decoding them
sequentially. The CLI override is:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_fs2_realnvp.yaml \
     amortized-infer-fs2 \
     --checkpoint outputs/runs/dev_amortized_fs2/checkpoints/best.eqx \
     --limit 32 \
     --batch-size 8 \
     --posterior-samples 32 \
     --prior-samples 8192 \
     --decoder-sample-chunk-size 1 \
     --out outputs/runs/dev_amortized_fs2_infer

Diffsky HLTDS inference example:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_hltds_joint_realnvp_gpu.yaml \
     amortized-infer-diffsky \
     --checkpoint outputs/runs/amortized_diffsky_hltds_joint_realnvp_n10000/checkpoints/best.eqx \
     --limit 10000 \
     --batch-size 64 \
     --posterior-samples 64 \
     --prior-samples 8192 \
     --decoder-sample-chunk-size 1 \
     --out outputs/runs/amortized_diffsky_hltds_joint_realnvp_n10000_infer

The same conservative cap is applied during inference through
``amortized.inference.jax_batch_size: 4``.

Inference diagnostics include likelihood-normalized posterior-predictive
residuals ``(obs_flux - model_flux) / sigma_eff`` for each object and band.
``sigma_eff`` is the same effective uncertainty used by the likelihood,
including catalog ``fluxerr_*``, fractional floor, and jitter. The summary
tables and figures report median residuals by band, Gaussian-reference
histograms with ``-3`` and ``+3`` markers, tail fractions, the objects with
largest posterior predictive chi-square, histograms of maximum encoder-feature
amplitude per object, ``z_obs`` posterior median versus catalog redshift proxy
when columns such as ``z_true_gal`` are available, a redshift PIT histogram
``P(z < z_ref)``, contour-style posterior corner plots, and learned RealNVP
prior diagnostics.

The prior diagnostics sample ``x ~ p_beta(x)`` directly from the configured
prior, transform the samples to physical ``theta``, and write:

* ``learned_prior_samples.parquet`` and
  ``learned_or_loaded_prior_samples.parquet`` with ``x_00`` ... ``x_{D-1}``,
  physical parameters, and exact pointwise ``logprior``;
* ``learned_prior_summary.json`` with prior marginal quantiles;
* ``learned_prior_logprob_hist.png`` for the learned density values;
* ``learned_prior_corner.png`` for the learned prior in physical parameter
  space;
* ``posterior_vs_learned_prior_corner.png`` to compare the aggregate amortized
  posterior samples with the learned population prior.

For Diffsky HLTDS, run the explicit truth/prior overlap report after inference:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_hltds_joint_realnvp_gpu.yaml \
     amortized-prior-overlap-diffsky \
     --run outputs/runs/amortized_diffsky_hltds_joint_realnvp_n10000_infer \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet \
     --out outputs/runs/amortized_diffsky_hltds_joint_realnvp_n10000_infer/prior_overlap \
     --max-objects 10000

This writes ``prior_overlap_metrics.csv``, ``population_realism_report.md``,
and plots comparing truth, aggregate posterior, and learned or loaded prior for
directly comparable parameters. It includes redshift, stellar mass, derived
``log10_sfr_at_obs``/``log10_ssfr_at_obs`` when exported, dust terms when
fitted, raw and alpha-corrected stellar mass when available, and
Diffstar/Diffmah generated-truth marginals for supervised-prior diagnostics. It
does not compare raw ``dlog10_sfr_i`` ratios directly to ``logsfr_true``.
Physical prior distributions do not include ``alpha_sed``; prior predictive
photometry can depend on the selected global scale.

There is no exact published POP-COSMOS prior distribution implemented in this
repository. FS2 comparison plots therefore compare the learned RealNVP prior
against amortized posterior samples and available FS2 catalog redshift proxies,
rather than claiming an external POP-COSMOS prior baseline.

If FS2 proxy columns are available through ``truth.parameter_columns``, the
diagnostics also write ``catalog_proxy_comparison.parquet`` and plots for:

* posterior/prior/catalog-proxy stellar mass distributions;
* posterior median minus catalog-proxy stellar mass residuals;
* catalog-proxy SFR distribution;
* catalog-proxy mass-SFR plane colored by posterior predictive chi-square.

The SFR plot is intentionally catalog-proxy only in this first pass. The
Pop-COSMOS-like latent stores six adjacent ``dlog10_sfr_i`` ratios, not a direct
``log10_sfr_at_obs`` parameter. A posterior SFR overlay should therefore be
added only after exporting a model-derived SFR diagnostic from the DSPS/SFH
path.

Scientific Limitations
----------------------

The amortized posterior is approximate and must be checked against MAP/MCMC
diagnostics on selected rows. The Student-t likelihood is robust to outliers
and can hide model defects, so band-by-band posterior predictive residual
diagnostics are required. Physical recovery claims on Diffsky HLTDS additionally
require a successful same-parameter forward closure and a supervised
truth-prior comparison before interpreting posterior aggregates.
