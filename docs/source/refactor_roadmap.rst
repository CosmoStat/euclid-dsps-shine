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

   ``log10_sfr`` currently acts as SFH amplitude, not a physical instantaneous
   SFR label. Add explicit stellar mass or formed-mass amplitude and compute
   SFR diagnostics from the SFH. Rename reports so fitted amplitudes and catalog
   truth proxies cannot be confused.

5. Dust handling has two modes.

   The Salim-style scalar dust fallback is useful for generic DSPS runs. The
   COSMOS two-component dust mode is better for Flagship template diagnostics.
   Expose the active dust model in every run manifest and report title.

6. Plot tests check file creation, not content.

   Add small image-integrity tests for key plots:

   * non-empty canvas;
   * expected number of panels;
   * finite plotted data in CSV source tables;
   * no all-white or all-transparent PNGs.

7. Full-run reproducibility needs stronger manifests.

   Each workflow should write:

   * git commit or dirty-state hash when available;
   * config path and normalized config hash;
   * JAX backend and device;
   * package versions for DSPS, JAX, NumPy, pandas, pyarrow;
   * input parquet path, size, and schema hash.

8. Error handling can be more uniform.

   COSMOS resources already fail with explicit searched paths. Apply the same
   pattern to filter files, SSP assets, parquet schema mismatches, and output
   write failures.

9. Performance profiling is missing.

   Add timing rows per workflow chunk:

   * parquet read time;
   * COSMOS reconstruction time;
   * JAX compile time;
   * JAX execution time;
   * plotting/reporting time;
   * peak memory if available.

Near-Term Refactor Tasks
------------------------

1. Extract photometry likelihood arrays from ``workflows/core.py``.

   Target module: ``euclid_dsps.photometry``. Include flux-to-mag conversion,
   error-column handling, masks, and comparison-row construction.

2. Add typed config objects.

   Start with ``BandConfig``, ``FitConfig``, ``CosmosSedConfig``, and
   ``RuntimeConfig``. Keep dictionary export for compatibility.

3. Split ``reporting/core.py`` by plot family.

   Keep existing import paths working through ``euclid_dsps.reporting``.

4. Add run manifests to all workflows.

   Current COSMOS workflow has a manifest. Extend this to ``run-one``,
   ``fit-one``, ``run-batch``, ``fit-batch``, ``fit-population``, and
   ``fit-workflow``.

5. Add GPU smoke test profile.

   Keep CI CPU-only. Add a local command that writes timing/device diagnostics
   for ``cosmos-sed --population-dsps`` with a small sample.

Non-Goals
---------

* No vendored DSPS copy in this repository.
* No large parquet, SSP, LePhare cache, or generated output in source control.
* No broad rewrite before the stellar-mass/SFH model decision.
