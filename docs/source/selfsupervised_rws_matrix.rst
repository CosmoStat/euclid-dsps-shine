Self-supervised RWS prior and posterior matrix
==============================================

Purpose
-------

This four-task matrix learns an amortized posterior ``q(x | y)`` and, in three
tasks, a population RealNVP prior ``p(x)`` without using catalog truth columns
in the training loss. DSPS remains the fixed physical decoder. Synthetic truth
is read only by the held-out inference diagnostics that produce coverage,
accuracy, photometry-fit, and corner plots.

All tasks share the exact checkpoint-backed 15D
``mixed_log_shifted_asinh`` transform. The closure likelihood is Gaussian with
the catalog ``fluxerr`` and no extra floor or jitter because the synthetic
generator already includes its 0.005 mag systematic term.

Wake update
-----------

The wake proposal is a stratified mixture of the conditional flow and a
temperature-two version of that flow. For every real object, stopped normalized
weights are computed from

.. math::

   \log w_k = \log p(y\mid x_k) + \log p(x_k)
              - \log r(x_k\mid y).

The encoder minimizes ``-sum(w * log q)``. When the prior is learned, the same
particles, DSPS fluxes, and stopped weights also minimize ``-sum(w * log p)``.
This replaces the former prior M-step that fitted unweighted samples from the
encoder.

Sleep update
------------

Sleep epochs draw latents from the current prior, decode them with DSPS, apply
the configured LSST, Euclid, and Roman ``m5_depth`` noise model, and train the
encoder density on these model-generated pairs. No observed latent label is
used. Invalid or out-of-bound particles are safely decoded at an interior
point, assigned zero statistical weight, and reported through explicit valid
fractions.

Experiments
-----------

``fixed_ref_rws_k4_gaussian``
   Frozen reference prior, three sleep epochs followed by one real wake epoch
   with four particles. This isolates posterior learning.

``selfsup_rws_k4_weighted_prior``
   Identity-initialized RealNVP prior updated on every four-particle wake epoch.
   This isolates the importance-weighted prior update without sleep.

``selfsup_rws_sleep3_wake1_k4``
   Main self-supervised candidate: three physical sleep epochs followed by one
   shared four-particle wake epoch.

``selfsup_rws_sleep3_wake1_k8``
   Particle-count control using eight particles on wake epochs. Its smaller JAX
   batch keeps the per-device DSPS workload bounded.

Launch
------

From the repository root on Jean-Zay with the ``shine`` environment active:

.. code-block:: bash

   bash scripts/submit_feniks_selfsup_rws.sh

The script runs a four-task smoke array first and submits the full array with an
``afterok`` dependency. The preflight is JAX-free and checks input files,
normalization hashes, absence of supervised losses, likelihood settings, and
the 18-band oracle-noise residual distribution before allocating GPUs.
The smoke requests ten minutes and executes every sleep/wake path used by each
task with the production JAX batch shapes.
Production validation is spaced every eight epochs, JAX preallocation is
disabled, and plotting uses a private non-interactive Matplotlib cache to avoid
the failure mode previously observed around epoch 91.

Outputs
-------

The default full root is
``outputs/runs/feniks_selfsup_rws_jaxcosmo_v1``. Every task writes training
logs, checkpoints, held-out inference diagnostics, photometry-fit panels, a
full truth/prior/posterior corner, and a ``DONE`` marker. Once all tasks finish,
``comparison/README.md`` and ``comparison/experiment_metrics.csv`` summarize
speed, ESS, physical validity, posterior coverage, prior updates, photo-z, and
posterior-predictive chi-square.
