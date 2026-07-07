Catalog Columns
===============

Source Context
--------------

This page documents the Euclid FS2 comparison dataset. It is a CosmoHub export
from catalog 353, table ``euclid_fs2_mock_dr_v1_1_phz``. CosmoHub is PIC's
Hadoop-backed web platform for exploring and exporting large cosmological
datasets. The Euclid Consortium identifies catalog 353 as the Euclid Flagship
galaxy mock, and describes the release as a synthetic galaxy catalog containing
billions of galaxies with hundreds of modeled physical, photometric, and halo
properties.

The public catalog page is:

.. code-block:: text

   https://cosmohub.pic.es/catalogs/353

The column table below documents the subset selected by the project SQL query.
Names in ``local_name`` are the parquet names after SQL aliases are applied.
Names in ``source_or_expression`` are the original CosmoHub columns or SQL
expressions.

This project intentionally exports three complementary classes of photometric
quantities:

* continuum-only observed fluxes,
* rest-frame absolute fluxes at 10 parsec,
* fully forward-modeled observed fluxes including emission lines, attenuation,
  Milky Way extinction, and survey-like noise realizations.

The export also includes latent COSMOS template identifiers
(``sed_cosmos_1`` and ``sed_cosmos_2``), which makes it possible to construct
approximate pseudo-ground-truth SEDs for benchmarking differentiable SED
pipelines such as DSPS.

SED Reconstruction Context
--------------------------

The Euclid Flagship mock does not directly provide high-resolution galaxy
spectra or tabulated wavelength-by-wavelength SEDs in the exported CosmoHub
table. However, the catalog contains several latent variables describing the
underlying synthetic SED generation process.

The most important columns are:

* ``sed_cosmos_1``
* ``sed_cosmos_2``
* ``frac_cosmos_1`` and ``frac_cosmos_2``
* ``ebv_cosmos_1``
* ``ebv_cosmos_2``
* ``ext_curve_cosmos_1``
* ``ext_curve_cosmos_2``

These columns identify the COSMOS/Ilbert template components and associated
dust attenuation parameters used internally by the Flagship photometric model.

This allows the project to reconstruct approximate template-level SEDs by:

#. loading the corresponding COSMOS templates,
#. applying the catalog attenuation parameters,
#. combining the two components with normalized catalog fractions,
#. normalizing the rest-frame proxy SED to Euclid ``*_abs`` fluxes,
#. comparing rest-frame shape against DSPS and observed-frame photometry
   against catalog forward-model outputs.

The reconstructed SEDs should be interpreted as
pseudo-ground-truth template reconstructions rather than exact intrinsic
physical spectra.

Photometry Semantics
--------------------

The catalog includes several related photometric representations.

Observed Continuum Fluxes
^^^^^^^^^^^^^^^^^^^^^^^^^

Columns such as:

* ``lsst_u``
* ``lsst_g``
* ``lsst_r``
* ``lsst_i``
* ``lsst_z``
* ``lsst_y``
* ``euclid_vis``
* ``euclid_nisp_y``
* ``euclid_nisp_j``
* ``euclid_nisp_h``

represent observed-frame continuum flux densities including internal
attenuation.

These are the simplest fluxes and exclude explicit emission-line forward
modeling and Milky Way extinction.

Rest-Frame Absolute Fluxes
^^^^^^^^^^^^^^^^^^^^^^^^^^

Columns ending in ``_abs`` represent rest-frame flux densities normalized at
10 parsec:

* ``lsst_u_abs`` through ``lsst_y_abs``
* ``euclid_vis_abs``
* ``euclid_nisp_y_abs``
* ``euclid_nisp_j_abs``
* ``euclid_nisp_h_abs``

These quantities are especially useful for:

* SED normalization,
* pseudo-ground-truth template anchoring,
* rest-frame comparisons,
* intrinsic luminosity diagnostics.

Forward-Modeled Survey Fluxes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Columns ending in:

.. code-block:: text

   _el_model3_ext_odonnell_ext

represent the most realistic survey-like observables available in the export.

These include:

* stellar continuum,
* modeled emission lines,
* internal attenuation,
* Milky Way extinction,
* survey noise modeling.

Associated columns ending in:

.. code-block:: text

   _error

store the modeled photometric uncertainty, while columns ending in:

.. code-block:: text

   _error_realization

store noisy realizations of the observed fluxes.

These columns are the preferred targets for:

* probabilistic inference,
* likelihood evaluation,
* chi-square fitting,
* simulation-to-observation benchmarking,
* differentiable forward-model validation.


Selected Column Metadata
------------------------

.. csv-table::
   :file: _static/cosmohub_catalog353_columns.csv
   :header-rows: 1
   :widths: 14 18 12 8 14 24 24

Practical DSPS Usage
--------------------

The exported dataset supports several complementary DSPS workflows.

Minimal Photometric Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use:

* ``euclid_vis``
* ``euclid_nisp_y``
* ``euclid_nisp_j``
* ``euclid_nisp_h``

for simple observed-frame photometric inference without explicit survey-noise
forward modeling.

Pseudo-Ground-Truth SED Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use:

* ``sed_cosmos_1``
* ``sed_cosmos_2``
* ``ebv_cosmos_1``
* ``ebv_cosmos_2``
* ``ext_curve_cosmos_1``
* ``ext_curve_cosmos_2``
* ``*_abs`` fluxes

to reconstruct approximate latent template SEDs and compare them against DSPS
spectral predictions.

Survey-Level Forward Modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use:

* ``*_el_model3_ext_odonnell_ext``
* ``*_error``
* ``*_error_realization``

for likelihood-based inference, noisy photometric simulations, or differentiable
survey forward-model validation.

The ``*_error`` columns are flux-density uncertainties in the same units as the
catalog fluxes. When a band config declares ``error_column``, the pipeline
converts the flux uncertainty into a local AB-magnitude uncertainty,

.. math::

   \sigma_m \simeq \frac{2.5}{\ln 10}\frac{\sigma_F}{F},

and uses it in the photometric likelihood. This gives high signal-to-noise
objects stronger weight and prevents faint noisy points from dominating the fit
as if every band had the same fixed ``sigma_mag``. ``sigma_mag_floor`` and
``sigma_mag_ceiling`` in the config keep the weights numerically stable.

The noisy realization columns are not uncertainties. They are one simulated
measurement drawn from the survey-like flux plus noise model. Use them when the
experiment should mimic measured survey photometry; use the corresponding
``*_error`` columns for the likelihood denominator and chi-square.

The science config wires catalog fluxes and catalog flux errors for all ten
LSST+Euclid bands. The current DSPS model includes stellar light, dust, raw
FSPS/CLOUDY gas, AGN, and IGM, so the active PopCosmos-like setup uses the
survey-like ``*_el_model3_ext_odonnell_ext`` photometry columns where configured
by the band preset. The gas emission-line treatment is still raw FSPS/CLOUDY,
not PopCosmos line-calibrated.

.. code-block:: bash

   python -m euclid_dsps.cli --config configs/fs2_gpu.yaml fit \
     --limit 32 \
     --batch-size 32 \
     --out outputs/runs/dev_fit_batch_10band

For COSMOS SED validation:

.. code-block:: bash

   python -m euclid_dsps.cli --config configs/fs2_gpu.yaml check \
     --kind cosmos \
     --limit 20 \
     --out outputs/check/cosmos

Gas and AGN spectral components come from the generated FSPS assets. Noisy
realization columns remain diagnostics unless explicitly selected in config.

References
----------

* CosmoHub catalog page:
  https://cosmohub.pic.es/catalogs/353

* Euclid Flagship simulation release:
  https://www.euclid-ec.org/public/press-releases/euclid-flagship-simulations/

* Euclid Flagship galaxy mock paper:
  https://www.aanda.org/articles/aa/full_html/2025/05/aa50853-24/aa50853-24.html

* IRSA PHZ tutorial showing ``phz_mode_1`` as the first PHZ PDF mode:
  https://caltech-ipac.github.io/irsa-tutorials/euclid-intro-phz-catalog/
