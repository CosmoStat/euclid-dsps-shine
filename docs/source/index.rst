Euclid DSPS SHINE
=================

.. container:: shine-hero

   **Synthetic Diffsky/FENIKS closure and photometric inference workflows.**

   Euclid DSPS SHINE wraps native ``dsps`` in a small, testable interface for
   synthetic closure catalogue generation, catalog preparation, forward
   photometry, MAP/posterior inference, and reproducible diagnostics.

Where to Start
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38
   :class: shine-start-table

   * - Goal
     - Read
     - Main command or config
   * - Install the package
     - :doc:`installation`
     - ``python -m pip install -e .``
   * - Run the production workflow
     - :doc:`production`, :doc:`diffsky_synthetic_closure`
     - ``configs/diffsky_synthetic_feniks_260617_50k.yaml``
   * - Validate closure data
     - :doc:`production`, :doc:`diffsky_synthetic_closure`
     - ``diffsky-validate-dsps-closure``
   * - Train supervised priors
     - :doc:`prior_learning`
     - ``configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml``
   * - Run amortized inference
     - :doc:`amortized_inference`
     - ``configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml``
   * - Use HLTDS/FS2 debug references
     - :doc:`data_download`, :doc:`diffsky_dataset`
     - ``configs/diffsky_dataset_hltds_04_14.yaml``

Workflow Map
------------

.. list-table::
   :header-rows: 1
   :widths: 22 48 30
   :class: shine-workflow-table

   * - Stage
     - What it owns
     - Primary outputs
   * - Data contract
     - FENIKS closure truth, prepared parquet rows, units, truth semantics,
       and error-model provenance. HLTDS/FS2 are debug references.
     - Manifests, schema files, inventory reports, integrity reports.
   * - Forward model
     - The DSPS boundary, filter handling, SSP assets, calibration terms, and
       photometric likelihood setup.
     - Predicted fluxes, residuals, closure diagnostics.
   * - Inference
     - MAP fits, posterior sampling, supervised truth priors, and amortized
       encoder/prior experiments.
     - CSV/JSON summaries, checkpoints, posterior samples.
   * - Assessment
     - Population realism, redshift calibration, forward closure, and
       truth-vs-posterior diagnostics.
     - Markdown reports, metrics tables, plots.

Project Boundaries
------------------

* ``euclid_dsps.model`` is the only module that calls native DSPS.
* ``euclid_dsps.io`` owns parquet row contracts and photometry unit
  conversions.
* ``euclid_dsps.fit`` and ``euclid_dsps.mcmc`` own point estimates and
  posterior sampling.
* ``euclid_dsps.workflows`` composes command-line workflows.
* ``euclid_dsps.reporting`` writes CSV, JSON, Markdown, and plot artifacts.

Scientific Guardrails
---------------------

A good photometric fit is not enough to claim physical recovery. Use
same-parameter forward closure, supervised truth-prior diagnostics, posterior
calibration, and derived-quantity comparisons before interpreting recovered
parameters physically.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   production
   data_download
   run_setup
   testing

.. toctree::
   :maxdepth: 2
   :caption: Science Workflows

   diffsky_dataset
   diffsky_synthetic_closure
   forward_model
   prior_learning
   amortized_inference
   science_assessment
   ssp_compression
   catalog_columns

.. toctree::
   :maxdepth: 2
   :caption: Reference

   architecture
   api
