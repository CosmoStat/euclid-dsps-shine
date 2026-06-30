Scientific Validation Plan
==========================

Goal
----

The Diffsky HLTDS workflow is meant to validate a differentiable generative
model of a galaxy population. The goal is not only to fit photometry. The goal
is to test whether the model learns a physically realistic population.

Separate Objectives
-------------------

The workflow separates three objectives:

* supervised prior learning on ``theta_true``;
* same-parameter forward closure ``theta_true -> photometry``;
* photometric posterior inference ``q(theta | flux)``.

These objectives should not be mixed in one interpretation.

Required Warning
----------------

A good photometric fit is not evidence of physical recovery.

Physical claims require:

* same-param forward closure;
* supervised prior vs truth diagnostics;
* posterior calibration;
* comparison of derived quantities, not only raw latent parameters.

Likelihood and Global Calibration
---------------------------------

The Diffsky science path uses a robust photometric likelihood by default:

.. code-block:: yaml

   likelihood:
     type: student_t
     student_t_dof: 2.0

The DSPS decoder paths can also use one global SED normalization nuisance:

.. code-block:: yaml

   calibration:
     global_sed_scale:
       enabled: true
       parameterization: log_alpha
       prior_sigma_log_alpha: 0.10
     per_band_zero_points:
       enabled: false

``alpha_sed`` is global to a run, not per galaxy, not per band, and not part of
the supervised prior ``p_beta(theta_true)``. It absorbs only a global SED
normalization mismatch. It is deliberately not a Pop-COSMOS-style per-band
zero-point correction.

alpha_sed improves flexibility against global SED normalization mismatch, but
it does not correct color-dependent residuals. Since it is degenerate with
stellar mass, mass recovery must be reported both before and after alpha
correction.

Population realism reports must compare physical distributions without adding
``alpha_sed`` to ``theta``. Photometric predictive diagnostics may use the
scaled flux, and mass diagnostics should include both raw and
``log10_stellar_mass_alpha_corrected`` values.

Validation Order
----------------

1. Build a clean Diffsky dataset with explicit truth semantics and error model.
2. Learn a supervised prior from available truth parameters.
3. Run true-parameter forward closure.
4. Use standard-normal, supervised-frozen, and joint RealNVP priors in
   photometric amortized inference.
5. Run redshift ablation and posterior calibration checks.
6. Compare truth, prior, and aggregate posterior population diagnostics.
7. Clean public configs and documentation once the science path is stable.

Full Validation Entrypoint
--------------------------

The H100-scale orchestration command is:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/experiments/diffsky_hltds_full_h100.yaml \
     diffsky-run-full-validation \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr_projected_truth.parquet \
     --out outputs/runs/diffsky_hltds_full_validation

It runs or aggregates the supervised prior, true-parameter closure, three
amortized prior-source modes, redshift ablation, and population-realism
diagnostics. The final ``full_validation_report.md`` includes a
``Global SED scale calibration`` section with ``alpha_sed``,
``log_alpha_sed``, ``delta_mag_global``, prior penalty, warnings, and raw versus
alpha-corrected mass recovery.

For already-produced stage outputs, use ``--report-only`` with repeated
``--run label=path`` arguments and optional ``--closure-run path``.

Truth Coverage
--------------

The dataset integrity layer inventories and classifies all prepared columns,
but not every truth column is used by every scientific stage. This is
intentional: each stage should use only quantities that share the right
parameterization.

.. list-table::
   :header-rows: 1

   * - Available column family
     - Current use
     - Not claimed
   * - ``redshift_true``
     - Dataset readiness, supervised basic/extended prior as ``z_obs``,
       true-parameter closure as ``z_obs``, photometric posterior redshift
       metrics and redshift ablation.
     - None; this is the primary redshift truth.
   * - ``logsm_true``
     - Dataset readiness, supervised basic/extended prior as
       ``log10_stellar_mass``, true-parameter closure, posterior-vs-truth and
       population diagnostics.
     - None for stellar-mass comparison, subject to the dataset's own mass
       definition.
   * - ``logssfr_true`` / ``logsfr_true``
     - Supervised basic/extended prior uses ``logssfr_true`` when present, or
       ``logsfr_true`` as a fallback. Population diagnostics compare derived
       ``log10_sfr_at_obs`` / ``log10_ssfr_at_obs`` only when those quantities
       are exported.
     - Raw amortized ``dlog10_sfr_i`` ratios are not compared directly to
       ``logsfr_true``.
   * - ``diffstar_*``
     - Supervised extended prior, true-parameter closure for the
       ``diffsky_basic`` parameter subset, and generated-truth population
       diagnostics.
     - The compact PopCosmos-bin amortized posterior is not a direct Diffstar
       latent recovery.
   * - ``diffmah_*``
     - Supervised extended prior, true-parameter closure for the required
       ``diffsky_basic`` halo-history parameters, and generated-truth
       population diagnostics.
     - The compact amortized posterior is not a direct Diffmah latent recovery.
   * - ``dust_av`` / ``dust_delta``
     - Supervised prior when present, true-parameter closure, and diagnostics
       when the corresponding parameter exists in prior/posterior samples.
     - Other dust latent families are not inferred unless explicitly present in
       a matching schema.
   * - ``burst_*``
     - Supervised extended prior and population prior diagnostics when present.
     - Not used by the current ``diffsky_basic`` closure or compact amortized
       posterior.
   * - ``logmp_true``, ``logmp_host_true``, ``central_true``,
       ``r50_disk_true``, ``r50_bulge_true``
     - Preserved, classified, and reported as available truths/proxies for
       future selection or population diagnostics.
     - Not currently part of the supervised-prior schemas, forward closure, or
       photometric posterior target.
   * - Metallicities
     - If no compatible metallicity truth is available, the true-parameter
       closure records ``log10_stellar_metallicity`` as a fixed nuisance.
     - Fixed nuisance metallicity is not treated as truth recovery.

Therefore the answer to "are all truths exploited?" is no: all available
columns are inventoried and preserved, the matched generative columns are used
where they are parameterization-compatible, and several auxiliary truth/proxy
columns are intentionally kept for diagnostics rather than forced into an
incompatible inference target.
