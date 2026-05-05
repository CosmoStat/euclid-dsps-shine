Catalog Columns
===============

Source Context
--------------

The default dataset is a CosmoHub export from catalog 353, table
``euclid_fs2_mock_dr_v1_1_phz``. CosmoHub is PIC's Hadoop-backed web platform
for exploring and exporting large cosmological datasets. The Euclid Consortium
identifies catalog 353 as the Euclid Flagship galaxy mock, and describes the
release as 3.4 billion galaxies with more than 400 modelled properties.

The public catalog page is:

.. code-block:: text

   https://cosmohub.pic.es/catalogs/353

The column table below documents the subset selected by the project SQL query.
Names in ``local_name`` are the parquet names after SQL aliases are applied.
Names in ``source_or_expression`` are the original CosmoHub columns or SQL
expressions.

Unit Notes
----------

The public CosmoHub page is not machine-readable without an interactive session
in this environment, so this table combines:

* the SQL query used for the project export,
* the downloaded parquet schema,
* Euclid Flagship catalog paper descriptions of positions, halo properties,
  morphology, SFR, and metallicity,
* PHZ documentation for ``phz_mode_1`` as the first mode of the redshift PDF,
* the project config contract for photometry units.

Flux columns are interpreted by ``configs/fs2_phz1.yaml`` as ``Fnu`` in cgs:
``erg s^-1 cm^-2 Hz^-1``. If a future CosmoHub export uses magnitudes,
microJanskys, or error columns, update the config ``bands[*].units`` and this
table together.

Selected Column Metadata
------------------------

.. csv-table::
   :file: _static/cosmohub_catalog353_columns.csv
   :header-rows: 1
   :widths: 14 18 12 8 14 24 24

References
----------

* CosmoHub catalog page: https://cosmohub.pic.es/catalogs/353
* Euclid Flagship simulation release:
  https://www.euclid-ec.org/public/press-releases/euclid-flagship-simulations/
* Euclid Flagship galaxy mock paper:
  https://www.aanda.org/articles/aa/full_html/2025/05/aa50853-24/aa50853-24.html
* IRSA PHZ tutorial showing ``phz_mode_1`` as the first PHZ PDF mode:
  https://caltech-ipac.github.io/irsa-tutorials/euclid-intro-phz-catalog/
