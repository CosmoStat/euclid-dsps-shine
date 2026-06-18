Supervised Diffsky Prior Learning
=================================

Purpose
-------

The supervised prior workflow learns a population density directly from HLTDS
truth parameters:

.. math::

   L_\mathrm{prior} = -\frac{1}{N}\sum_i \log p_\beta(x_{\mathrm{true}, i})

where ``theta_true`` is mapped to unconstrained ``x_true`` with the same
bounded logistic transforms used by amortized latent variables. This workflow
does not use photometry, an encoder, or the DSPS decoder.

``alpha_sed`` is not part of this workflow. It is a global decoder calibration
nuisance parameter used by photometric likelihood paths, not a galaxy physical
parameter. The supervised prior therefore does not add ``alpha_sed`` to
``theta_true``, does not sample it per galaxy, and does not compare it to
object-level ground truth.

It is separate from:

* same-parameter forward closure, which tests whether ``theta_true`` can
  reproduce the catalog photometry;
* photometric amortized inference, which learns ``q(theta | flux)``.

A good photometric fit is not evidence of physical recovery. Physical claims
require same-parameter forward closure, supervised prior-vs-truth diagnostics,
posterior calibration, and comparison of derived physical quantities rather
than only raw latent parameters.

Schemas
-------

``diffsky_truth_basic`` uses the minimum available truth columns:

.. code-block:: text

   redshift_true        -> z_obs
   logsm_true           -> log10_stellar_mass
   logssfr_true         -> log10_ssfr_at_obs

If ``logssfr_true`` is unavailable but ``logsfr_true`` exists, it uses:

.. code-block:: text

   logsfr_true          -> log10_sfr_at_obs

Dust columns are included only when present:

.. code-block:: text

   dust_av or dust_av_true
   dust_delta

``diffsky_truth_extended`` starts from the basic schema and adds available
``diffstar_*``, ``diffmah_*``, ``dust_*``, and ``burst_*`` generated-truth
columns.

Missing optional columns are reported. With ``missing_policy: reduce``, the
schema is reduced to available columns. With ``missing_policy: fail``, missing
optional generated-truth columns stop the run explicitly.

Configs
-------

Basic supervised RealNVP prior:

.. code-block:: text

   configs/prior_diffsky_hltds_supervised_basic_realnvp.yaml

Extended supervised RealNVP prior:

.. code-block:: text

   configs/prior_diffsky_hltds_supervised_extended_realnvp.yaml

Train
-----

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/prior_diffsky_hltds_supervised_basic_realnvp.yaml \
     diffsky-train-supervised-prior \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet \
     --schema diffsky_truth_basic \
     --out outputs/runs/diffsky_supervised_prior_basic

The run writes:

.. code-block:: text

   prior_training_log.csv
   prior_validation_loglike.csv
   learned_prior_samples.parquet
   truth_theta_samples.parquet
   truth_x_samples.parquet
   supervised_prior_summary.json
   supervised_prior_vs_truth_report.md
   prior_vs_truth_metrics.csv
   checkpoints/best.eqx
   checkpoints/last.eqx

Sample
------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/prior_diffsky_hltds_supervised_basic_realnvp.yaml \
     diffsky-sample-supervised-prior \
     --checkpoint outputs/runs/diffsky_supervised_prior_basic/checkpoints/best.eqx \
     --n-samples 8192 \
     --out outputs/runs/diffsky_supervised_prior_basic_samples

Report
------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/prior_diffsky_hltds_supervised_basic_realnvp.yaml \
     diffsky-supervised-prior-report \
     --run outputs/runs/diffsky_supervised_prior_basic \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet

Diagnostics include per-parameter histogram comparisons, KS distance,
Wasserstein distance, mean/std/median residuals, a z/logM/logSFR pair plot when
those parameters are present, and a corner plot when ``corner`` is installed.
