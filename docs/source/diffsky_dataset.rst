HLTDS Reference Data
====================

Role
----

HLTDS is no longer the center of the science workflow. It remains useful for:

* downloading and preparing external Diffsky HLTDS artifacts;
* checking photometry and error-model contracts;
* debugging low-redshift behavior against a non-FENIKS reference;
* comparing FENIKS synthetic populations to an external reference.

Do not use projected HLTDS PopCosmos-like quantities as direct object-level
truth for the learned prior. The controlled prior-learning dataset is FENIKS.

Active HLTDS Configs
--------------------

.. list-table::
   :header-rows: 1

   * - Config
     - Purpose
   * - ``configs/diffsky_dataset_hltds_04_14.yaml``
     - Download/preparation/validation config for the low-z 04/14 HLTDS debug
       parquet with materialized ``fluxerr_*`` columns and projected DSPS
       truth columns.
   * - ``configs/diffsky_dataset_hltds_03_31_zmax335_m5depth.yaml``
     - Download/preparation/validation config for the higher-redshift 03/31
       truth-rich debug parquet with the current ``m5_depth`` error model.

Commands
--------

Inventory remote files:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_dataset_hltds_04_14.yaml \
     diffsky-list-remote

Download a bounded subset:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_dataset_hltds_04_14.yaml \
     diffsky-download-subset \
     --limit-files 2

Prepare the normalized parquet:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_dataset_hltds_04_14.yaml \
     diffsky-prepare-dataset

Validate and diagnose:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_dataset_hltds_04_14.yaml \
     diffsky-validate-dataset

   python -m euclid_dsps.cli \
     --config configs/diffsky_dataset_hltds_04_14.yaml \
     diffsky-dataset-diagnostics

Data Contract
-------------

Prepared HLTDS tables must keep:

* object identity and source-file provenance;
* native ``flux_<band>`` and materialized ``fluxerr_<band>`` columns;
* explicit error-model metadata;
* direct truth columns separate from generated or projected truth columns;
* validation reports that flag missing truth dimensions instead of silently
  filling them.

Relationship To FENIKS
----------------------

FENIKS is the controlled training dataset for prior learning and NN+DSPS+NF.
HLTDS can be used to sanity-check redshift ranges, colors, noise scales, and
failure modes, but it should not determine the learned prior target unless a
separate experiment explicitly says so.
