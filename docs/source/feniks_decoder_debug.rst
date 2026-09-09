FENIKS Decoder Debug Tracking
==============================

Last updated: 2026-09-09. Historical cluster evidence is transcribed from
operator-provided logs and receipts, not independently downloaded in this update.
The residual audit has run on Jean-Zay (1922142). The targeted redshift-precision
follow-up is implemented locally, not yet executed on the cluster.

.. warning::

   Job completion, numerical qualification, posterior validation and population
   readiness are distinct. No diagnostic here automatically starts training.

Current State
-------------

The versioned merged quadrature plus MDF64 candidate passes all metallicity
checks at six tested inputs. The residual audit resolves the four point-1 density
checks within their existing tolerances. Point-4 redshift in lsst_z remains
INCONCLUSIVE: two favorable intermediate stencils precede unstable finer ones.
The four density PASS results select noisy fine stencils, not high-precision
agreement; earlier coarse stencils were more accurate. Historical full-audit
receipts are unchanged. This does not certify individual posterior distributions.

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
     - 1922142 / 7824dc9
     - Four density checks PASS; point-4 redshift remains inconclusive.
   * - Point-4 precision isolation
     - Implemented; not submitted here
     - Split stellar, IGM and projection paths; compare mixed and z-dependent
       float64 arithmetic with all-band stencils and trace checks.

Point-4 Follow-up
-----------------

The new mode requires the completed target-resolution receipt and unchanged
MDF source, points and contexts. It evaluates ten branches sequentially at the
same point: canonical latent target, mixed full/stellar/IGM/projection, float64
redshift-dependent full/stellar/IGM/projection, and float64 age weights with
native SED casts. Non-redshift parameters and dust transmission remain fixed.
The float64 branches must have exclusively float64 floating JAX traces.

The physical-redshift branches use a linear tangent coordinate with the central
dz/dx, not the nonlinear latent transform at finite displacement. Both are
recorded separately; do not interpret finite-step differences as identical
targets. Stored SSP/filter precision is not recovered by casting.

All bands and the centered likelihood retain the old tolerances and FD-only
plateau selector. Center flux shifts and branch-sum derivative identities are
reported, not hidden. No production configuration enables this diagnostic path.
One H100, one node, 16 CPU threads, 45 minutes and 1000 component evaluations
maximum; no automatic follow-up training.

After this new run::

   source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
   python scripts/summarize_feniks_sc_drws_redshift_precision.py "$DIAGNOSTIC_ROOT"

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

Integrated Precision and Overnight Pilot
----------------------------------------

Job 1922455 completed the targeted redshift precision diagnostic. On point 4,
``zpath64_full`` passed all bands and centered likelihood. The lsst_z AD/FD
difference was about 1.62e-6, the maximum center-flux shift 2.15e-5 photometric
sigma, and the branch-sum derivative discrepancy about 1e-13. These are results
reported from the cluster, not a new local execution. The diagnostic varied
physical z with other parameters fixed; it did not qualify the production 15D
nonlinear latent transform.

The new opt-in ``spline_precision: float64_v1`` integrates the arithmetic into
the real spline decoder, with versioned latent and likelihood precision.
Historical defaults and stored SSP assets remain unchanged. A single sequential
one-H100, ten-hour-ceiling pilot requalifies six complete 15D inputs before any
training. It then runs a gradient smoke, a fresh simulation bank, four sleep
epochs and four sleep-plus-ELBO epochs, followed by matched A/B/C diagnostics.
Any numerical failure or inconclusive check blocks training. Population
learning remains disabled even when every technical stage completes.

The new reference A uses unchanged source weights with the new decoder; B and C
are compared under that same decoder. This is a bounded fixed-parent experiment,
not a production launch or proof of catalogue-simulator compatibility.

:download:`Download the overnight runbook <../feniks_precision_night_runbook.md>`
