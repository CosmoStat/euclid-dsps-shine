Euclid DSPS SHINE
==================

Euclid DSPS SHINE is a standalone workflow around native ``dsps`` for Euclid
FS2 and Diffsky HLTDS photometric experiments. It loads catalog photometry,
prepares DSPS inputs, runs forward photometry, fits small parameter sets, and
writes diagnostic tables and plots.

The project keeps the scientific boundary narrow:

* ``euclid_dsps.model`` is the only module that calls native DSPS.
* ``euclid_dsps.io`` owns the parquet row contract and photometry unit
  conversions.
* ``euclid_dsps.fit`` and ``euclid_dsps.mcmc`` own inference.
* ``euclid_dsps.workflows`` composes CLI workflows.
* ``euclid_dsps.reporting`` writes CSV, JSON, and plot artifacts.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   architecture
   data_download
   diffsky_dataset
   forward_model
   ssp_compression
   amortized_inference
   catalog_columns
   cosmos_sed
   run_setup
   science_assessment
   testing
   api
