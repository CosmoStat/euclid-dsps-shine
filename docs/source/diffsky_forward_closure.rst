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
   calibration:
     global_sed_scale:
       enabled: true
       mode: fit_global
     per_band_zero_points:
       enabled: false

and not ``popcosmos_bins``.

Global SED Scale Modes
----------------------

The forward closure supports three global SED-scale modes:

``disabled``
   Use ``alpha_sed = 1`` and report raw residuals only.

``fixed``
   Use ``alpha_sed = exp(initial_log_alpha)`` from config.

``fit_global``
   Fit one scalar ``log_alpha_sed`` for the closure run. The implementation
   chooses the robust global magnitude offset that removes the median raw
   residual, then applies it once to all model fluxes. This tests whether a
   single normalization mismatch explains the closure residuals.

For a global scale, multiplying the SED before filter integration and
multiplying the integrated model flux are mathematically equivalent. The
closure reports both raw and scaled model fluxes so the application point is
auditable.

The equivalent magnitude offset is:

.. code-block:: text

   delta_mag_global = -2.5 * log10(alpha_sed)

If ``abs(delta_mag_global) > 0.3``, the report warns that the run may have a
normalization, units, stellar-mass scale, or SSP-scale problem.

Outputs
-------

.. code-block:: text

   forward_closure_photometry.parquet
   forward_closure_residuals_by_band.csv
   residuals_by_band_before_alpha.csv
   residuals_by_band_after_alpha.csv
   forward_closure_parameter_sources.csv
   forward_closure_summary.json
   closure_gate.json
   alpha_sed_fit.json
   forward_closure_report.md

Interpretation
--------------

If ``theta_true -> photometry`` does not reproduce HLTDS magnitudes at an
acceptable level, photometric posterior results must not be described as
physical recoveries. They can still be useful as photometric fits or
population-regularized summaries, but the simulator/data contract failed.

``alpha_sed`` does not validate physical recovery by itself. A successful
closure should have acceptable residuals after inspecting both raw and scaled
photometry, and a large fitted scale should be treated as a calibration warning
rather than as evidence that the galaxy parameters are recovered.
