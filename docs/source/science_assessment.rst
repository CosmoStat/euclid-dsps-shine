Science Assessment
==================

Purpose
-------

Science assessment answers one question: is a generated catalogue and trained
inference model ready to support physical interpretation? A visually good
photometric reconstruction is not sufficient. The production decision must use
truth-contract checks, forward closure, posterior calibration, and population
diagnostics together.

Assessment Diagram
------------------

.. code-block:: text

   generated closure catalogue
       |
       +--> data contract checks
       |       schema, manifest, split sizes, truth completeness
       |
       +--> same-parameter forward closure
       |       theta_true -> DSPS -> flux_true_<band>
       |
       +--> population diagnostics
       |       n(z), mass, SFR, dust, metallicity, colors, S/N, duplicates
       |
       +--> supervised prior diagnostics
       |       truth distribution vs RealNVP samples
       |
       +--> amortized inference diagnostics
               coverage, PIT, posterior width, residuals, weak directions

Production Acceptance Gates
---------------------------

Promote a FENIKS/DSPS closure run only when all required gates are available
and pass:

.. list-table::
   :header-rows: 1
   :widths: 24 42 34

   * - Gate
     - Required evidence
     - Failure meaning
   * - Reproducibility
     - ``manifest.yaml`` records repo SHA, config hash, package versions, SSP
       hash, filter hashes, calibration hash, split seeds, and parameter order.
     - The catalogue cannot be reproduced or compared safely.
   * - Data contract
     - ``schema.json`` and ``validation_report.json`` confirm all 18 closure
       truths, all flux/error/mask columns, split sizes, and disjoint ids.
     - The run is not a strict closure dataset.
   * - Forward closure
     - ``validation_report.json`` reports successful DSPS flux recomputation
       from sampled truth vectors.
     - The stored truths and stored true fluxes are not same-model consistent.
   * - Noise model
     - Normalized noisy-flux residuals are centered and have approximately unit
       standard deviation.
     - The likelihood errors do not match the generated observations.
   * - Selection
     - Manifest counters and validation report confirm mass, metallicity, and
       true-S/N gates.
     - The final sample is not the intended observable learning catalogue.
   * - Prior quality
     - Supervised prior report compares RealNVP samples to truth marginals and
       correlations.
     - The prior may distort the training population before photometry is used.
   * - Inference calibration
     - Held-out evaluation reports coverage, PIT, posterior width, residuals,
       and parameter-wise behavior.
     - The posterior cannot be interpreted as calibrated uncertainty.

Minimum Artifacts To Keep
-------------------------

For the dataset:

.. code-block:: text

   Data/diffsky/synthetic/feniks_260617_dsps_closure/
       proposals/
       train.parquet
       validation.parquet
       test.parquet
       all_50k.parquet
       manifest.yaml
       schema.json
       validation_report.json
       diagnostics/population/report.md
       diagnostics/population/*.csv
       diagnostics/population/*.json
       diagnostics/population/plots/*.png

For the supervised prior:

.. code-block:: text

   outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp/
       prior_training_log.csv
       supervised_prior_summary.json
       supervised_prior_vs_truth_report.md
       prior_vs_truth_metrics.csv
       learned_prior_samples.parquet
       truth_theta_samples.parquet
       checkpoints/best.eqx

For amortized inference and evaluation:

.. code-block:: text

   outputs/runs/amortized_diffsky_synthetic_feniks_full/
       training_log.csv
       training_summary.json
       feature_stats.json
       checkpoints/best.eqx

   outputs/runs/amortized_diffsky_synthetic_feniks_full_infer/
       inference_summary.json
       posterior summaries and posterior-predictive residual diagnostics

   outputs/runs/amortized_diffsky_synthetic_feniks_full_eval/
       closure inference evaluation metrics and plots

Interpretation Rules
--------------------

* Keep direct truth, generated closure truth, projected truth, optimizer
  targets, and noisy observations separate in plots and text.
* Do not present projected HLTDS PopCosmos bins as direct object-level ground
  truth.
* Keep negative noisy fluxes; they are valid Gaussian/Student-t likelihood
  inputs.
* Report proposal ESS and duplicate fraction with every science use of the
  synthetic catalogue.
* Compare the z<=0.35 HLTDS reference only in the overlapping low-redshift
  interval.
* Expect weakly constrained directions in an 18D latent space observed through
  14 broad bands. Calibration and honest uncertainty are the target, not exact
  per-galaxy recovery of every latent.

Reference and Debug Paths
-------------------------

HLTDS MAP fits, MCLMC probes, reconstruction dashboards, and Euclid FS2 remain
useful for diagnosis. They are not production acceptance gates for the
FENIKS/DSPS closure model unless the runbook explicitly names them as a
required comparison for a specific release.
