Diffsky Forward Closure
=======================

Purpose
-------

The true-parameter forward closure tests:

.. code-block:: text

   theta_true_diffsky -> DSPS/Diffstar decoder -> model photometry

against the prepared HLTDS magnitudes. It is not an optimizer benchmark. It is
the gatekeeper for physical interpretation of later amortized results.

Command
-------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_hltds_04_14_trueparam_closure_gpu.yaml \
     diffsky-forward-closure \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet \
     --limit 1024 \
     --out outputs/runs/diffsky_trueparam_forward_closure

The config uses:

.. code-block:: yaml

   model:
     sfh_model: diffsky_basic

and not ``popcosmos_bins``.

Outputs
-------

.. code-block:: text

   forward_closure_photometry.parquet
   forward_closure_residuals_by_band.csv
   forward_closure_parameter_sources.csv
   forward_closure_summary.json
   forward_closure_report.md

Interpretation
--------------

If ``theta_true -> photometry`` does not reproduce HLTDS magnitudes at an
acceptable level, photometric posterior results must not be described as
physical recoveries. They can still be useful as photometric fits or
population-regularized summaries, but the simulator/data contract failed.
