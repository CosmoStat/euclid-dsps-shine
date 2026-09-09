FENIKS Decoder Debug Tracking
==============================

Last updated: 2026-09-09. Historical cluster evidence is transcribed from
operator-provided logs and receipts, not independently downloaded in this update.
The precision-night job (1923347) has completed on Jean-Zay. Its numerical
qualification passed, but posterior support remains poor. The qualified local
VI follow-up completed in job 1938818 without support recovery. A controlled
optimizer comparison is now implemented; its cluster execution is pending.

.. warning::

   Job completion, numerical qualification, posterior validation and population
   readiness are distinct. The explicitly gated night pilot trains q only;
   no diagnostic here authorizes population training or scientific promotion.

Current State
-------------

The integrated spline64 candidate passed all six full-target qualification
points in job 1923347. The numerical blockers documented below are historical;
their receipts remain unchanged. The same job completed smoke, sleep and
sleep-plus-ELBO training. All A/B/C posterior support gates still fail. C reduces
photometric tails but does not provide usable importance integration. Job
1938818 subsequently completed local adaptation without support recovery.
The next bounded test compares learning rate and gradient draw count while
keeping the qualified decoder, prior and development contexts frozen.

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
     - 1922455
     - zpath64 branches pass at point 4; integrate and qualify the full target.
   * - Integrated precision night
     - 1923347 / 43c2886
     - Six full-target inputs PASS; A/B/C posterior support still FAIL.
   * - Qualified local VI
     - Prepared; not submitted here
     - Eight observed and eight new simulated cases, two starts, same family
       and fixed prior; no automatic global training.

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

Qualified Local Follow-up
-------------------------

The CPU-only night readback exposes absolute held-out residual references and
training gradients. Tiny observed/reference RMS ratios can be caused by a bad
simulated-q reference; they do not certify predictive performance. The small-N
rank audit reports a descriptive simultaneous bound without changing gates.

The GPU follow-up preserves C's target and optimizes local base/coupling
parameters only. It uses eight observations and eight newly selected simulations,
64 steps per start and two nearby starts, with two independent 128-draw final
evaluations. It is limited to one H100/node for three hours with a measured
cost preflight. Neither an ELBO gain nor high ESS proves mode completeness.

:download:`Download the qualified local VI runbook <../feniks_qualified_local_vi_runbook.md>`
Controlled local optimization, 2026-09-09
---------------------------------------

Job 1938818 passed its numerical contract but local importance support
decreased in 15/16 observed fits; simulated cases also remained problematic.
Residual improvements do not qualify the posterior. The next bounded test
compares three optimizer regimes on the same development contexts, with
saved distributions at steps 8, 16, 32 and 64. No best-checkpoint selection,
teacher or population promotion is performed. See
``docs/feniks_controlled_local_vi_runbook.md`` for protocol and commands.
