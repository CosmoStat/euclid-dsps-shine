Parameters And Run Setup
========================

Default Configuration
---------------------

The default Euclid FS2 PHZ setup is:

.. code-block:: text

   configs/fs2_phz1.yaml

Every CLI command accepts ``--config``:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml run-one --out outputs/runs/dev_one

Top-Level Paths
---------------

``catalog_path``
  Local parquet catalog path. For FS2 PHZ this is
  ``Data/Euclid FS2 LC galaxy catalog_phz1.parquet``.

``ssp_path``
  DSPS SSP template file. The default is
  ``Data/ssp_data_fsps_v3.2_lgmet_age.h5``.

Selection
---------

``selection.index``
  Catalog row index for one-galaxy workflows.

``selection.require_positive_flux``
  Reject rows with non-positive configured photometry when selecting a row.

``selection.sort_by_flux``
  Optional photometry column used to sort before selecting.

Redshift
--------

The default redshift setup fixes DSPS ``z_obs`` from the catalog photo-z:

.. code-block:: yaml

   redshift:
     column: z_phz
     truth_column: z_true
     fixed_value: 0.5
     min: 0.0001
     max: 6.0

``column`` is used for DSPS. ``truth_column`` is diagnostic only. ``fixed_value``
is a fallback when the row value is missing or invalid.

Bands
-----

Each band entry defines the catalog column, units, uncertainty, and passband:

.. code-block:: yaml

   bands:
     - name: euclid_vis
       column: euclid_vis
       units: fnu_cgs
       sigma_mag: 0.05
       filter:
         path: filters/Euclid_VIS.vis.dat

Supported photometry units are:

.. list-table::
   :header-rows: 1

   * - Unit
     - Meaning
   * - ``fnu_cgs``
     - ``Fnu`` in ``erg/s/cm^2/Hz``.
   * - ``abmag``
     - AB magnitude.
   * - ``microjy`` or ``ujy``
     - MicroJansky.

Model Parameters
----------------

``model.fixed_parameters`` contains the baseline DSPS parameter dictionary.
Default free parameters can override these values during fitting.

.. code-block:: yaml

   model:
     n_sfh_bins: 96
     fixed_parameters:
       log10_sfr: 0.0
       sfh_t_peak: 4.0
       sfh_tau: 0.6
       log10_metallicity: -2.0
       metallicity_scatter: 0.2
       dust_av: 0.2
       dust_slope: -0.7
     parameter_columns: {}

``parameter_columns`` can map model parameters to catalog columns when a value
should come from each row instead of the fixed config.

Fit Parameters
--------------

``fit.free_parameters`` declares fitted parameters, initial values, and hard
bounds:

.. code-block:: yaml

   fit:
     method: jax_adam
     maxiter: 80
     learning_rate: 0.1
     tolerance: 1.0e-5
     patience: 18
     free_parameters:
       log10_sfr:
         initial: 0.0
         bounds: [-4.0, 3.0]
       dust_av:
         initial: 0.2
         bounds: [0.0, 1.0]
       log10_metallicity:
         initial: -2.25
         bounds: [-4.2, -1.4]

Use ``initial: from_base`` when the initial value should come from the resolved
base parameter dictionary for each row.

Bayesian Sampling
-----------------

``sample`` controls NumPyro HMC/NUTS:

.. code-block:: yaml

   sample:
     sampler: nuts
     num_warmup: 100
     num_samples: 200
     num_chains: 1
     chain_method: parallel
     target_accept_prob: 0.85
     max_tree_depth: 10
     num_steps: 8
     seed: 42
     init_from_map: true

Use ``--sampler hmc`` and a small ``--num-steps`` for predictable debugging.
Use ``--sampler nuts`` for more adaptive posterior checks on selected rows.

Truth Comparisons
-----------------

Truth entries are report diagnostics. They do not constrain the default forward
model or MAP fit:

.. code-block:: yaml

   truth:
     redshift_column: z_true
     parameter_columns:
       log10_metallicity:
         column: metallicity_true
         offset: -10.61
       log10_sfr_at_obs: log_sfr_true
       dust_av:
         column: dust_ebv_true
         scale: 4.05

CLI Workflows
-------------

EDA:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml eda --out outputs/eda_phz1

One forward model:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml run-one --out outputs/runs/phz1_one

One MAP fit:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml fit-one --out outputs/runs/phz1_fit_one

Batch forward model:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml run-batch --limit 1000 --batch-size 500 --out outputs/runs/phz1_batch

Batch MAP fit:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml fit-batch --limit 1024 --batch-size 64 --out outputs/runs/phz1_fit_batch

Posterior sample for one row:

.. code-block:: bash

   euclid-dsps --config configs/fs2_phz1.yaml fit-one \
     --index 0 \
     --bayesian \
     --sampler hmc \
     --num-warmup 120 \
     --num-samples 400 \
     --num-chains 1 \
     --num-steps 8 \
     --no-progress \
     --out outputs/runs/phz1_mcmc_row_0_fast

Use ``--all`` only when you intend to process the full parquet catalog. Use
``fit-batch`` with a small ``--limit`` while iterating, because it runs one
optimizer per galaxy.
