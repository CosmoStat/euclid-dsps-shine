Amortized Inference
===================

This feature started as an amortized posterior prototype for the Euclid FS2
catalog. The project is now pivoting to OpenUniverse/Diffsky as the main
validation dataset while FS2 remains available for comparison and domain-shift
diagnostics. The existing MAP and posterior workflows remain the baseline tools
for per-object checks.

Model Contract
--------------

The amortized feature builder consumes the configured band set. The FS2 command
uses ten bands:

.. code-block:: text

   LSST u,g,r,i,z,y + Euclid VIS,Y,J,H

The encoder input is:

.. code-block:: text

   [flux_1, ..., flux_B, err_1, ..., err_B]

after per-band normalization. FS2 uses ``B=10`` and feature dimension 20.
OpenUniverse LSST+Roman uses ``B=14`` and feature dimension 28.
Fluxes are normalized with a robust signed transform,
``asinh(flux / flux_scale)``, so bright FS2 objects do not produce MLP inputs of
hundreds of scale units. Errors are normalized with
``log(err / err_scale + eps)``. The error terms are part of the input because
two objects with identical fluxes but different per-band uncertainties do not
carry the same information and should not receive the same posterior width.

The posterior model is:

.. code-block:: text

   flux_10 + err_10
       -> q_psi(x | flux, err)
       -> theta = h(x)
       -> DSPS(theta)
       -> flux_model_10

``x`` is the unconstrained latent vector in ``R^16``. ``theta`` is the bounded
physical PopCosmos-like parameter vector, also length 16, including redshift.
``psi`` denotes the encoder parameters. ``beta`` denotes the RealNVP prior
parameters.

Implementation Architecture
---------------------------

The implementation is split so that the neural inference code does not depend
on private MAP helpers:

.. code-block:: text

   euclid_dsps/parameter_vectors.py
       theta vectors -> DSPS parameter pytrees -> model_mags_jax_dynamic

   euclid_dsps/observation_arrays.py
       FS2 rows -> object_id, flux[10], err[10], mask[10]

   euclid_dsps/amortized/features.py
       flux[10], err[10] -> normalized features[20]

   euclid_dsps/amortized/encoder.py
       GaussianEncoder q_psi(x | flux, err)

   euclid_dsps/amortized/flows.py
       RealNVPPrior p_beta(x)

   euclid_dsps/amortized/decoder.py
       x -> theta -> fixed DSPS -> model_flux[10]

   euclid_dsps/amortized/elbo.py
       Student-t log likelihood + Monte Carlo KL

   euclid_dsps/amortized/train.py
       one Optax optimizer over encoder and RealNVP parameters

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

``RealNVPPrior`` is an exact-density normalizing flow in the same
unconstrained latent space ``x``:

.. code-block:: text

   base u ~ N(0, I)
   x = T_beta(u)
   n_layers = 8
   hidden_size = 128
   alternating affine coupling masks

The encoder and RealNVP prior are optimized together in the same
``eqx.filter_value_and_grad`` call and the same Optax update. The training log
records ``encoder_grad_norm`` and ``prior_grad_norm`` at every step; both should
be nonzero when ``kl_weight > 0``.

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

Joint RealNVP Prior
-------------------

The RealNVP prior is trained jointly with the encoder. It is not pretrained in
the first implementation. This lets ``p_beta(x)`` learn a flexible latent
population structure while the encoder learns object-level posterior
approximations. RealNVP is used because it provides exact pointwise density
evaluation and a cheap triangular Jacobian.

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

The gradient plots are part of the joint-training contract: they make it
visible whether the RealNVP prior is receiving gradients alongside the encoder.

The ``amortized-train-fs2`` CLI is verbose by default. It prints the output
directory, feature-stat computation, filter/DSPS loading, JAX backend/devices,
model architecture, epoch starts, and epoch summaries. Each epoch also shows a
progress bar with live ``loss``, negative log likelihood, Monte Carlo KL,
encoder gradient norm, and RealNVP prior gradient norm. Use ``--quiet`` to
reduce console logs and ``--no-progress`` to disable progress bars.

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
   learned_prior_summary.json
   inference_summary.json

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

Inference diagnostics include normalized residuals
``(model_flux - obs_flux) / obs_err`` for each object, sample, and band. The
summary tables and figures report median residuals by band, the objects with
largest posterior predictive chi-square, histograms of maximum encoder-feature
amplitude per object, ``z_obs`` posterior median versus catalog redshift proxy
when columns such as ``z_true_gal`` are available, a redshift PIT histogram
``P(z < z_ref)``, contour-style posterior corner plots, and learned RealNVP
prior diagnostics.

The learned prior diagnostics sample ``x ~ p_beta(x)`` directly from the trained
RealNVP, transform the samples to physical ``theta``, and write:

* ``learned_prior_samples.parquet`` with ``x_00`` ... ``x_15``, physical
  parameters, and exact pointwise ``logprior``;
* ``learned_prior_summary.json`` with prior marginal quantiles;
* ``learned_prior_logprob_hist.png`` for the learned density values;
* ``learned_prior_corner.png`` for the learned prior in physical parameter
  space;
* ``posterior_vs_learned_prior_corner.png`` to compare the aggregate amortized
  posterior samples with the learned population prior.

There is no exact published POP-COSMOS prior distribution implemented in this
repository. The comparison plots therefore compare the learned RealNVP prior
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

This is an FS2-only prototype. The amortized posterior is approximate and must
be checked against MAP/MCMC diagnostics on selected rows. The learned prior is
selection-dependent because it is trained on FS2 photometry. Redshift can be
compared to ``z_true_gal`` when available, but mass, SFR, dust, and metallicity
columns should be described as catalog proxies, not truth. The Student-t
likelihood is robust to outliers and can hide model defects, so band-by-band
posterior predictive residual diagnostics are required.
