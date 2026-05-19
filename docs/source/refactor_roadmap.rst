Refactor Roadmap
================

Current State
-------------

The project now has a usable architecture:

* ``euclid_dsps.model`` owns the DSPS/JAX boundary.
* ``euclid_dsps.io`` owns catalog rows, photometric units, and likelihood
  uncertainties.
* ``euclid_dsps.cosmos`` owns COSMOS-template pseudo-SED reconstruction.
* ``euclid_dsps.fit`` owns MAP and population optimizers.
* ``euclid_dsps.reporting`` owns plots and summary tables.
* ``euclid_dsps.workflows`` owns CLI orchestration.
* Sphinx, ruff, black, pytest, config validation, synthetic fixtures, and CI
  are in place.

Remaining Architecture Risks
----------------------------

1. Config dictionaries still cross most module boundaries.

   YAML compatibility is useful, but repeated dictionary access makes contracts
   hard to audit. Introduce typed config dataclasses after the scientific model
   stabilizes. Keep the YAML schema unchanged and convert to typed objects at
   load time.

2. ``workflows/core.py`` remains too large.

   It still contains shared helpers, MAP batch orchestration, population
   orchestration, MCMC orchestration, and report regeneration glue. Split into:

   * ``workflows/photometry.py`` for observation arrays and comparison rows;
   * ``workflows/fitting.py`` for MAP/population batch execution;
   * ``workflows/posterior.py`` for HMC/NUTS batch execution;
   * ``workflows/context.py`` for row context and truth/proxy extraction.

3. Reporting has a broad ``reporting/core.py`` module.

   The public facade is split, but implementation remains centralized. Move
   plot families into private modules:

   * ``reporting/eda_impl.py``;
   * ``reporting/sed_impl.py``;
   * ``reporting/batch_impl.py``;
   * ``reporting/posterior_impl.py``;
   * ``reporting/workflow_impl.py``.

4. Model parameter semantics are mixed.

   The main configs now use ``log10_formed_mass_msun`` as the luminosity
   amplitude and derive SFR diagnostics from the SFH. Remaining cleanup:
   remove old wording and examples where ``log10_sfr`` appears as the primary
   fit amplitude, while keeping backward compatibility for smoke configs.

5. Dust handling has two modes.

   The Salim-style scalar dust fallback is useful for generic DSPS runs. The
   COSMOS two-component dust mode is better for Flagship template diagnostics.
   Expose the active dust model in every run manifest and report title.

6. Some model hooks are implemented ahead of available data.

   Burst/quench SFH terms, binned SFH parameters, scalar fallback dust, and
   emission-line target plumbing should remain documented as inactive or
   experimental unless a config actually fits them and validation shows they
   are identifiable. The near-term goal is not to add more free knobs; it is to
   keep fast redshift/mass/SFR/metallicity inference calibrated.

7. Plot tests check file creation, not content.

   Add small image-integrity tests for key plots:

   * non-empty canvas;
   * expected number of panels;
   * finite plotted data in CSV source tables;
   * no all-white or all-transparent PNGs.

8. Full-run reproducibility needs stronger manifests.

   Each workflow should write:

   * git commit or dirty-state hash when available;
   * config path and normalized config hash;
   * JAX backend and device;
   * package versions for DSPS, JAX, NumPy, pandas, pyarrow;
   * input parquet path, size, and schema hash.

9. Error handling can be more uniform.

   COSMOS resources already fail with explicit searched paths. Apply the same
   pattern to filter files, SSP assets, parquet schema mismatches, and output
   write failures.

10. Performance profiling is missing.

   Baseline timing rows now exist per workflow chunk:

   * parquet read time;
   * photometry and parameter preparation time;
   * JAX optimization/forward time;
   * derived-quantity and materialization time;
   * plotting/reporting time;
   * process RSS and GPU memory from ``nvidia-smi`` when available.

   Remaining work: split JAX compile time from execution time explicitly and
   add a dedicated ``bench-forward`` command for forward-only throughput tests.

11. Population inference is still not a learned prior.

   Population MAP writes hyperparameter diagnostics and grouped validation, but
   it regularizes only within each processed chunk. A future pop-cosmos-like
   prior should be trained or calibrated separately and then used inside the
   objective.

   Fast production mode now avoids accidental slow population Adam runs:
   ``fit-population`` uses the fast per-galaxy fit when fast flags are active
   and writes empirical population summaries. True joint population MAP remains
   available through ``--full-adam`` for smaller validation subsets.

Result-Driven Refactor Priorities
---------------------------------

The latest 10-band runs show that several reported parameter distributions are
constant at their initial values while the fit quality is still poor. Diagnostics
should make these failure modes impossible to miss.

Baseline implementation now exists for fit-batch, fit-population, and COSMOS
DSPS likelihood runs:

* ``*_parameter_audit.csv`` labels free, fixed, row-injected, derived, and
  prior-context columns;
* ``*_objective_components.csv`` separates photometric chi-square, PHZ prior,
  physical prior, and population-prior terms when available;
* ``*_performance_benchmark.csv`` and ``*_performance_summary.json`` record
  stage wall time, peak process RSS, and GPU memory from ``nvidia-smi`` when
  available;
* ``normalized_config.json`` is written beside workflow metadata for audited
  runs.
* ``fit-batch`` writes per-chunk checkpoints under ``_chunks/`` so long GPU
  runs can be recovered if a later chunk fails.

1. Add a fit-audit report.

   New output: ``*_fit_parameter_audit.csv``. For every model parameter, write:

   * whether it was free, fixed, row-injected, derived, or prior-only;
   * number of unique finite values;
   * median, p16, p84, min, max;
   * fraction near configured lower and upper bounds;
   * initial value when available;
   * warning flags such as ``constant_free_parameter``,
     ``near_bound_population``, and ``not_inferred_column``.

   This directly addresses the misleading ``fit_sfh_t_peak``,
   ``fit_sfh_tau``, ``fit_log10_metallicity``, and ``fit_dust_av`` plots in the
   latest runs.

2. Report objective components separately.

   Current outputs write photometric ``chi2`` and total population loss only in
   trace files. Add per-row or chunk-level columns for:

   * photometric chi-square;
   * physical prior penalty;
   * PHZ interval penalty;
   * Gaussian population prior penalty;
   * relation-prior penalty.

   Without this split, it is hard to tell whether a parameter stayed fixed
   because photometry is insensitive, the prior dominates, or the optimizer did
   not move.

3. Preserve full normalized config in every run directory.

   The inspected run-config JSON files contain workflow metadata only. They do
   not contain ``fit.free_parameters``, priors, band definitions, SSP path,
   filter paths, or normalized model defaults. Every workflow should write both:

   * ``run_manifest.json`` for workflow metadata;
   * ``normalized_config.json`` for the exact validated config used by code.

4. Separate inferred parameters from injected catalog parameters.

   COSMOS dust parameters and PHZ interval columns are copied from the catalog.
   Reports should place these in ``catalog_injected_parameters.csv`` or mark
   them with ``source=row_column`` rather than plotting them beside true free
   MAP parameters.

5. Make population relation diagnostics stricter.

   For each relation, write predictor variance, target variance, learned slope,
   learned scatter, and a flag when the target parameter is constant or the
   slope is not identifiable. Current mass-metallicity slopes near zero should
   be reported as ``not_identifiable`` for the inspected run.

6. Add normalization and mass-calibration diagnostics.

   The latest runs infer formed masses about ``1.3--1.4 dex`` above the catalog
   stellar-mass proxy. Add a dedicated report comparing:

   * DSPS formed mass;
   * catalog stellar mass after ``h^-2`` conversion;
   * DSPS current SFR;
   * catalog ``log_sfr_true``;
   * median photometric residual by mass bin.

   This should be checked before adding more SFH freedom.

7. Add outlier and band-tail diagnostics.

   LSST ``u`` and red/NIR Euclid bands dominate several residual tails. Add
   per-band percentile tables, worst-row tables, and optional clipping masks for
   diagnostic plots only. Do not silently clip likelihood inputs.

Near-Term Refactor Tasks
------------------------

1. Add fit-audit and objective-component reports.

   Target modules: ``euclid_dsps.reporting.fit`` and ``euclid_dsps.fit``. This
   is now higher priority than broad module splitting because it protects
   scientific interpretation of current outputs.

2. Extract photometry likelihood arrays from ``workflows/core.py``.

   Target module: ``euclid_dsps.photometry``. Include flux-to-mag conversion,
   error-column handling, masks, and comparison-row construction.

3. Add typed config objects.

   Start with ``BandConfig``, ``FitConfig``, ``CosmosSedConfig``, and
   ``RuntimeConfig``. Keep dictionary export for compatibility.

4. Split ``reporting/core.py`` by plot family.

   Keep existing import paths working through ``euclid_dsps.reporting``.

5. Add full run manifests to all workflows.

   Current COSMOS workflow writes a partial manifest. Extend this to
   ``run-one``, ``fit-one``, ``run-batch``, ``fit-batch``, ``fit-population``,
   and ``fit-workflow`` and include a separate normalized-config JSON.

6. Add GPU smoke test profile.

   Keep CI CPU-only. Add a local command that writes timing/device diagnostics
   for ``cosmos-sed --population-dsps`` with a small sample.

7. Separate chunk regularization from learned population priors.

   Keep current population MAP as ``chunk_regularized_map`` in reports. Add a
   separate interface for external population priors once the prior family is
   scientifically defined.

Non-Goals
---------

* No vendored DSPS copy in this repository.
* No large parquet, SSP, LePhare cache, or generated output in source control.
* No broad rewrite before the SFH-basis and population-prior decision.
