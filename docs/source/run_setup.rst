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

Optional reproducible wrapper:

.. code-block:: bash

   snakemake -j1

Override Snakemake run settings with ``--config run_dir=... limit=...``.

Config Shorthands
-----------------

``runtime: auto``
  Default science runtime. Lets JAX choose an available backend and clears stale
  ``JAX_PLATFORMS=cuda`` values inside the process, so CPU-only conda installs
  do not fail before fitting starts.

``runtime: gpu``
  Expands to CUDA JAX settings with ``require_gpu: true`` and NVIDIA device
  check. Use only when the active environment exposes CUDA JAX.

``bands: lsst_euclid_10``
  Expands to LSST ``ugrizy`` plus Euclid VIS/Y/J/H with local passbands from
  ``filters/`` and catalog flux-error columns for all ten bands.

``bands: euclid_4``
  Expands to Euclid VIS/Y/J/H only.

``fit.likelihood_space: flux``
  Uses Gaussian residuals in ``Fnu`` cgs flux units for MAP and MCMC inference.
  ``mag`` remains available for legacy comparisons.

``selection.nondetection_policy: gaussian_flux``
  Keeps finite negative/non-positive flux measurements when a valid flux error
  exists. ``drop`` is the legacy positive-flux-only behavior. ``upper_limit`` is
  reserved and fails validation until an upper-limit likelihood exists.

``band_calibration.mode: fixed_offsets``
  Applies fixed per-band magnitude offsets or flux multipliers to the model
  during likelihood evaluation. Defaults are null and do not change physics.

``column_groups``
  Replaces long ``extra_columns`` lists. Useful groups are ``truth_basic``,
  ``cosmos_proxy``, ``photometry_errors``, ``emission_line_diagnostics``, and
  ``morphology_halo``. ``phz_diagnostics`` exists only for explicit audits.

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

Truth columns are diagnostics only. The science preset fits ``z_obs`` from
Euclid + LSST photometry with no photo-z prior and no ``phz_median``
initialization. ``redshift.initial: fixed`` only supplies the MAP starting
value; ``z_obs`` remains a free bounded parameter. Multi-start MAP redshift was
removed because posterior sampling is the intended inference path.
``phz_median`` remains available only through the optional ``phz_diagnostics``
column group. PHZ interval priors were removed.

``reduced_chi2`` now means ``chi2 / dof`` with
``dof = max(n_valid_bands - n_free_effective, 1)``. The older per-band metric is
reported separately as ``chi2_per_band``.

Current priors are broad ``weak_physical`` priors. They are not yet POP-COSMOS
priors. A POP-COSMOS-like mode needs exact variable mapping and learned
population-prior calibration before it should be used.

Legacy configs live under ``configs/legacy`` as examples only. Active runs
should start from ``fit``, ``posterior``, and ``check``.
