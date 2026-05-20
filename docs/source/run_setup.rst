Run Setup
=========

Use the science preset unless you are debugging internals:

.. code-block:: text

   configs/fs2_phz1_science.yaml

This preset expands to the full internal schema at load time. Every audited run
writes ``normalized_config.json`` beside the outputs.

Main Commands
-------------

MAP fit:

.. code-block:: bash

   euclid-dsps fit \
     --limit 1000 \
     --batch-size 512 \
     --sed-samples 16 \
     --out outputs/runs/science_fit

One row:

.. code-block:: bash

   euclid-dsps fit --index 0 --out outputs/runs/row0_fit

Posterior subset:

.. code-block:: bash

   euclid-dsps posterior \
     --row-indices-file outputs/rows_for_hmc.txt \
     --num-warmup 300 \
     --num-samples 800 \
     --out outputs/runs/posterior_subset

Checks without fitting:

.. code-block:: bash

   euclid-dsps check --kind eda --out outputs/check/eda
   euclid-dsps check --index 0 --out outputs/check/row0_forward
   euclid-dsps check --kind cosmos --limit 20 --out outputs/check/cosmos

Config Shorthands
-----------------

``runtime: gpu``
  Expands to CUDA JAX settings with ``require_gpu: true`` and NVIDIA device
  check.

``bands: lsst_euclid_10``
  Expands to LSST ``ugrizy`` plus Euclid VIS/Y/J/H with catalog Euclid error
  columns.

``bands: euclid_4``
  Expands to Euclid VIS/Y/J/H only.

``column_groups``
  Replaces long ``extra_columns`` lists. Useful groups are ``truth_basic``,
  ``phz_diagnostics``, ``cosmos_proxy``, ``photometry_errors``,
  ``emission_line_diagnostics``, and ``morphology_halo``.

``dust_model: cosmos_proxy_fixed``
  Injects COSMOS dust columns into DSPS. These values are copied from the row
  and must not be interpreted as inferred DSPS dust.

Science Meaning
---------------

The current fit infers:

* ``z_obs``;
* ``log10_formed_mass_msun``;
* lognormal SFH shape ``sfh_t_peak`` and ``sfh_tau``;
* stellar metallicity proxy ``log10_metallicity``;
* derived current SFR from fitted mass plus SFH shape.

Truth columns are diagnostics only. ``phz_median`` initializes redshift, but
PHZ intervals are not used as hard priors in the science preset.

Current priors are broad ``weak_physical`` priors. They are not yet POP-COSMOS
priors. A POP-COSMOS-like mode needs exact variable mapping and learned
population-prior calibration before it should be used.

Legacy Commands
---------------

Old commands still work for reproducibility:

``run-one``, ``run-batch``, ``fit-one``, ``fit-batch``, ``cosmos-sed``,
``fit-population``, and ``fit-workflow``.

They are kept as compatibility/advanced entry points. New runs should start
from ``fit``, ``posterior``, and ``check``.
