Science Assessment
==================

Current Priority
----------------

The current project priority is a clean end-to-end validation loop:

.. code-block:: text

   photometry
       -> DSPS MAP recovery
       -> truth comparison
       -> amortized posterior / RealNVP prior learning
       -> population-level diagnostics

The public dataset choices are deliberately limited:

* Diffsky HLTDS 04/14/2026 for physical recovery checks with direct/basic truth
  columns;
* Euclid FS2 for Euclid-domain comparison and amortized prior learning.

Diffsky HLTDS Readiness
-----------------------

Use the current continuous low-z parquet with materialized synthetic flux
errors:

.. code-block:: text

   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet

Run:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-validate-dataset \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet \
     --manifest Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.manifest.yaml \
     --out outputs/reports/diffsky_hltds_04_14/prior_learning_validation_report.md

The dataset is useful for the current simple recovery path if the report says
``READY_BASIC`` or better and confirms:

* multi-band photometry;
* ``redshift_true``;
* ``logsm_true``;
* finite magnitudes in enough bands;
* unique ``object_id`` values.

Truth Interpretation
--------------------

Only direct/basic columns are used for first-pass recovery:

* redshift;
* stellar mass;
* recent-SFR proxy derived from ``logsm_true + logssfr_true`` when available.

Halo mass, central/satellite flags, and size columns are useful diagnostics but
are not recovered by the simplified DSPS model. Diffstar/Diffmah latents should
not be presented as recovered truths unless a matched generative model is used.

Assessment Order
----------------

1. Run ``configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml``.
2. Inspect band residuals and mass/SFR truth correlations.
3. Only then run ``configs/diffsky_hltds_04_14_simple_gpu.yaml`` with free
   redshift.
4. Compare against FS2 behavior with ``configs/fs2_gpu.yaml``.
5. Train or evaluate the FS2 amortized prior with
   ``configs/amortized_fs2_realnvp.yaml``.

A free-redshift collapse does not automatically invalidate the dataset. It can
mean that broad-band photometry, the simplified DSPS model, or the magnitude
tolerance is insufficient to identify redshift without additional constraints.
The fixed-redshift closure run separates those effects.

Minimum Reports To Keep
-----------------------

For every Diffsky fit batch, keep:

.. code-block:: text

   normalized_config.json
   batch_fit_results.parquet or batch_fit_results.csv
   batch_fit_photometry_comparison.parquet or .csv
   batch_fit_truth_metrics.csv
   batch_fit_summary_by_band.csv
   batch_fit_objective_components.csv
   batch_fit_diffsky_report.md

The Markdown report is regenerated with:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-fit-report \
     --run outputs/runs/diffsky_hltds_simple_n1000 \
     --config configs/diffsky_hltds_04_14_simple_gpu.yaml \
     --label batch_fit \
     --reporting-level light
