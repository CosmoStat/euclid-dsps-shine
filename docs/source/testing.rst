Testing
=======

Active Test Scope
-----------------

The active pytest suite covers the FENIKS/prior/MAP/MCLMC code path plus FS2
and HLTDS reference contracts:

.. list-table::
   :header-rows: 1

   * - Area
     - Scope
   * - Config and IO
     - Config normalization, active public config loading, parquet row
       selection, photometry units, truth transforms, and catalog contracts.
   * - Forward model
     - DSPS boundary helpers, parameter vectors, filters, compressed assets,
       dust/IGM/nebular paths used by active configs.
   * - Synthetic FENIKS
     - Closure config loading, truth extraction, metallicity convention,
       selection gates, manifests, validation, and CLI smoke coverage.
   * - Prior learning
     - 18D closure schema, bounded transforms, RealNVP training helpers,
       supervised-prior reporting, inferred-prior input loading.
   * - Amortized inference
     - Feature construction, latent transforms, likelihood, encoder/flow
       helpers, inference catalogs, diagnostics, MAP-under-prior plumbing.
   * - MAP and MCLMC
     - Optimizer options, posterior-target behavior, NUTS/HMC/MCLMC controls.

Historical OpenUniverse, COSMOS SED, reconstruction-dashboard, and old HLTDS
experiment tests live under ``legacy/tests/`` and are not collected by the
active pytest configuration.

Commands
--------

Fast local checks:

.. code-block:: bash

   python -m compileall euclid_dsps scripts tests
   python -m pytest tests

CLI smoke:

.. code-block:: bash

   python -m euclid_dsps.cli --help
   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-plan-prior-workflow \
     --out outputs/reports/dev_feniks_prior_workflow

Docs:

.. code-block:: bash

   uv run --with sphinx --with sphinx-rtd-theme python -m sphinx \
     -W --keep-going -b html docs/source docs/build/html

Before changing scientific contracts, also run a small active workflow smoke:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-validate-dsps-closure \
     --dataset-dir Data/diffsky/synthetic/feniks_260617_dsps_closure \
     --runtime cpu \
     --sample-size 16
