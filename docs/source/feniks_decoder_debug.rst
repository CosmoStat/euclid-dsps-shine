FENIKS Decoder Debug Tracking
==============================

Last updated: 2026-09-09. Historical cluster evidence is transcribed from
operator-provided logs and receipts, not independently downloaded in this update.
The new residual audit is implemented but has not yet run on Jean-Zay.

.. warning::

   Job completion, numerical qualification, posterior validation and population
   readiness are distinct. No diagnostic here automatically starts training.

Current State
-------------

The versioned merged quadrature plus MDF64 candidate passes all metallicity
checks at six tested inputs. Four full points pass. Five required checks remain
INCONCLUSIVE: point 1 dust_delta and SFH08/09/10 centered likelihood derivatives,
and point 4 redshift in lsst_z. No required FAIL remains in the displayed MDF64
summary. This does not certify individual posterior distributions.

Evidence Timeline
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Step
     - Job / commit
     - Finding and decision
   * - Pure-sleep NPE
     - September 5--6
     - Redshift improves, importance support and joint posterior do not qualify.
   * - Topology / balanced NPE
     - 1890513; 1893047--1893049
     - All coordinates covered, but support remains poor. Stop blind retraining.
   * - Gradient isolation
     - 1914143 / 9660681
     - Likelihood-only control passes; investigate derivatives through DSPS.
   * - Redshift decomposition
     - 1915987 / 7f48a0e
     - Projection discrepancy persists in float64. Arithmetic precision alone
       does not repair historical quadrature.
   * - Fixed-spectrum reference
     - 1918919
     - Merged integration passes all 54 band checks at three points.
   * - Full decoder
     - 1920226 / ca92735
     - Nine metallicity failures remain. Qualify the whole target, not only
       a fixed spectrum.
   * - MDF precision
     - 1921589 / 293d5a9
     - Metallicity resolves; four of six points pass. Five checks inconclusive.
   * - Residual audit
     - Implemented; not submitted here
     - Correct the likelihood-resolution proxy; inspect denser redshift stencils.

Residual Audit Contract
-----------------------

The old screen treated a float64 likelihood sum as float32 for its output ULP
estimate. Its large absolute value blocked four otherwise stable coarse-step
comparisons. This screen is not a bound on the entire mixed-precision decoder.

The follow-up keeps the decoder and tolerances fixed. It computes Gaussian
differences per band relative to a fixed anchor, records actual representable
latent steps, propagates output-rounding estimates, and compares nested
second-order and Richardson stencils. Plateau selection does not consult AD.
One favorable redshift step is not enough. Unknown upstream roundoff remains
explicit; this is empirical convergence evidence rather than a numerical proof.

It requires the completed MDF receipt and identical parameter/context points.
At most eight unresolved checks are visited; this source has five. Perturbed
fluxes and hashes are exported for CPU-only replay. Old reports are preserved.

One node, one H100, 16 CPU threads; 45-minute allocation ceiling and 1000
decoder-call budget. No arrays, NPE, local VI or population training follow.

Commands and Detailed Log
-------------------------

The maintained Markdown report includes exact setup, submission, monitoring,
replay commands, artifact names, failure history and decision boundaries:

:download:`Download the step-by-step debug report <../feniks_decoder_debug_log.md>`

After completion::

   source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
   python scripts/summarize_feniks_sc_drws_target_resolution.py "$DIAGNOSTIC_ROOT"

The summary verifies final artifact hashes and recomputes the decisions on CPU.

Before Retraining
-----------------

Resolve numerical qualification, review simulator/likelihood compatibility,
rebuild versioned simulation banks, then run bounded frozen-parent local VI and
matched NPE validation. Never reuse old flux banks across numerical conventions.
Population learning requires separate posterior-support and integration checks.
No result here establishes identifiability of all SFH directions.
