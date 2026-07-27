Common-15D mode-covering posterior matrix
=========================================

This four-task control tests unsupervised alternatives to the under-dispersed
reverse-KL posterior observed in the completed independent-prior matrix. Truth
columns never enter the training loss or proposal. They are read only after
training for held-out coverage, PIT, recovery, and corner diagnostics.

Common latent contract
----------------------

Every task resolves the immutable ``spline15d_mixed`` coordinate transform
from ``outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx``.
The transform hash covers parameter order, physical bounds, marginal families,
locations, scales, and standardization parameters. It is independent of the
new RealNVP prior weights. Training, checkpoint reload, inference, prior
sampling, and DSPS decoding fail if the effective hash changes.

The reference checkpoint is a 12-layer RealNVP according to its sidecar. The
old ``ind_frozen_rqspline`` label was therefore inaccurate. This control keeps
that exact reference for comparability and does not substitute the separate
dequantized RQ-spline checkpoint.

Experiments
-----------

* ``common15d_vem4_elbo_k1`` isolates the common normalization with a learned
  RealNVP prior, four encoder epochs per prior M-step, and the original
  one-sample reverse-KL ELBO.
* ``frozen_ref_elbo_k2_antithetic`` freezes the reference prior and uses paired
  antithetic Gaussian-base samples to measure Monte Carlo variance without
  changing the KL direction.
* ``frozen_ref_periodic_wake_k4`` freezes the reference prior and replaces
  every fourth encoder epoch, starting at encoder epoch 40, by a four-particle
  self-normalized wake update. Three particles come from ``q`` and one from a
  base-temperature-two proposal.
* ``common15d_vem4_periodic_wake_k4`` combines the common-normalized learned
  prior, VEM 4:1, and the same periodic wake update.

Wake weights are proportional to

.. math::

   w_k \propto \frac{p(y\mid\theta_k)p(\theta_k)}
                         {0.75q(\theta_k\mid y)+0.25q_{T=2}(\theta_k\mid y)}.

The normalized weights are stopped before minimizing their weighted
``-log q``. The encoder therefore receives an inclusive, mass-covering update;
the prior and flux-calibration parameters remain frozen during that step. The
training log records ESS, maximum weight, entropy, and non-finite fractions.
An ESS fraction below 0.25 invalidates a scientific conclusion about the wake
objective because the proposal itself has collapsed.

Outputs
-------

Each task writes the full truth/prior/posterior corner, held-out coverage and
PIT diagnostics, compact posterior-predictive residual quantiles, residual
plots by band, and an observed-versus-posterior photometry panel for the six
worst fits. The sample-level predictive-flux and residual tables are disabled,
so these plots do not require the largest inference artifacts. The aggregate
report is written to ``comparison/README.md`` after all four tasks finish.

Jean-Zay launch
---------------

.. code-block:: bash

   git pull --ff-only
   conda activate shine
   bash scripts/submit_feniks_mode_covering.sh

The launcher validates the catalog, checkpoint family, four resolved configs,
common normalization hash, absence of supervised objectives, four visible
H100 devices, and headless Matplotlib. It submits a ten-minute smoke array and
then the full array with ``afterok``. Four tasks at four H100s allow at most 16
H100 GPUs concurrently. Wake is forced into the smoke so both compiled paths
are exercised before the full allocation starts.
