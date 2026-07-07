Supervised Diffsky Prior Learning
=================================

Purpose
-------

The supervised prior workflow learns a population density directly from truth
parameters:

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

Production closure schema
~~~~~~~~~~~~~~~~~~~~~~~~~

``diffsky_dsps_closure_full`` is the production schema for the synthetic
FENIKS/DSPS closure dataset. It requires one truth column for every fitted
``diffsky_basic`` parameter and uses ``missing_policy: fail`` in
``configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml``.

.. code-block:: text

   redshift_true                         -> z_obs
   logsm_true                            -> log10_stellar_mass
   diffstar_lgmcrit_true                 -> diffstar_lgmcrit
   diffstar_lgy_at_mcrit_true            -> diffstar_lgy_at_mcrit
   diffstar_indx_lo_true                 -> diffstar_indx_lo
   diffstar_indx_hi_true                 -> diffstar_indx_hi
   diffstar_lg_qt_true                   -> diffstar_lg_qt
   diffstar_qlglgdt_true                 -> diffstar_qlglgdt
   diffstar_lg_drop_true                 -> diffstar_lg_drop
   diffstar_lg_rejuv_true                -> diffstar_lg_rejuv
   diffmah_logm0_true                    -> diffmah_logm0
   diffmah_logtc_true                    -> diffmah_logtc
   diffmah_early_index_true              -> diffmah_early_index
   diffmah_late_index_true               -> diffmah_late_index
   diffmah_t_peak_true                   -> diffmah_t_peak
   log10_stellar_metallicity_true        -> log10_stellar_metallicity
   dust_av_true                          -> dust_av
   dust_delta_true                       -> dust_delta

Train the production prior after closure validation passes:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-plan-prior-workflow \
     --out outputs/reports/feniks_prior_workflow

   python -m euclid_dsps.cli \
     --config configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml \
     diffsky-train-supervised-prior \
     --out outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp

The workflow plan is a cheap preflight. It reads the configured parquet schemas
and row counts, verifies that ``diffsky_dsps_closure_full`` can resolve all 18
truth parameters, and writes the expected supervised-prior, NN+DSPS+NF,
MAP-under-prior, MCLMC-baseline, and post-hoc inferred-prior commands.

Configs
-------

.. code-block:: text

   configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml

Train
-----

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml \
     diffsky-train-supervised-prior \
     --out outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp

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
     --config configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml \
     diffsky-sample-supervised-prior \
     --checkpoint outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp/checkpoints/best.eqx \
     --n-samples 50000 \
     --out outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp_samples

Report
------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml \
     diffsky-supervised-prior-report \
     --run outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp \
     --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet

Diagnostics include per-parameter histogram comparisons, KS distance,
Wasserstein distance, mean/std/median residuals, a z/logM/logSFR pair plot when
those parameters are present, and a corner plot when ``corner`` is installed.
