OpenUniverse / Diffsky Validation
=================================

OpenUniverse/Diffsky is now the main validation direction for the amortized
DSPS workflow. The goal is to validate the scientific process:

.. code-block:: text

   multi-band photometry
       -> differentiable DSPS inference
       -> posterior over physical parameters
       -> learned RealNVP prior
       -> comparison to true or generative distributions

FS2 remains useful, but only as a comparison/domain-shift diagnostic dataset:
magnitude histograms, color-color diagrams, redshift distributions, and related
photometric checks.

First Photometric Target
------------------------

The first OpenUniverse subset uses 14 bands:

.. code-block:: text

   LSST:  u, g, r, i, z, y
   Roman: W146, R062, Z087, Y106, J129, H158, F184, K213

The amortized encoder feature dimension for this subset is therefore 28:
14 fluxes plus 14 flux errors. Euclid VIS/Y/J/H is intentionally not forced
into this first OpenUniverse implementation.

Input Files
-----------

OpenUniverse SkyCatalogs are organized by nside=32 HEALPix. For each selected
HEALPix, the preparation command expects:

.. code-block:: text

   galaxy_<hpix>.parquet
   galaxy_flux_<hpix>.parquet
   galaxy_sed_<hpix>.hdf5  # optional, not read by the PR 1 preparer

The main and flux parquet files are joined on ``galaxy_id``. ``galaxy_id`` must
exist in both files and be unique in each table.

Prepared Parquet Format
-----------------------

The normalized output contains:

.. code-block:: text

   galaxy_id, ra, dec, redshift, redshiftHubble, stellar_mass
   flux_truth_<band>
   flux_<band>
   fluxerr_<band>
   mask_<band>

for all 14 LSST+Roman bands. ``stellar_mass`` is copied from
``um_source_galaxy_obs_sm``. ``redshift_truth`` and
``redshift_hubble_truth`` are aliases of direct OpenUniverse columns when
available.

Units
-----

OpenUniverse fluxes are photon rates in ``photon_per_sec_cm2``. The current
OpenUniverse subset keeps this native unit internally. The code raises a clear
``NotImplementedError`` rather than silently converting to ``fnu_cgs`` or AB
magnitudes.

Tracked unit TODOs:

* choose the definitive internal unit for OpenUniverse training;
* implement DSPS photon-rate photometry or a validated filter-aware conversion;
* verify LSST/Roman filter curves and response conventions.

Truth Policy
------------

Use these labels in reports:

.. code-block:: text

   truth            direct OpenUniverse public columns
   generated_truth  exported Diffsky/Diffstar generative parameters
   proxy            derived stand-ins, clearly labeled
   unavailable      not present or not reconstructed

Direct OpenUniverse truths include ``redshift``, ``redshiftHubble``, and
``um_source_galaxy_obs_sm`` when present. SFH, Diffstar parameters, dust,
metallicity, and halo latents are not called truth unless they are actually
present in the data or exported from Diffsky.

Prepare Command
---------------

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/openuniverse_lsst_roman_14.yaml \
     openuniverse-prepare \
     --input-root Data/openuniverse/raw \
     --hpix 9812 9813 \
     --limit 10000 \
     --out Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet

The command writes a sibling manifest:

.. code-block:: text

   Data/openuniverse/processed/ou_lsst_roman_14_subset.manifest.yaml

The manifest records HEALPix ids, input root, row counts, band names, flux unit,
noise model, creation time, and truth/unit caveats.

Next Phases
-----------

Planned follow-up commands and reports:

* ``openuniverse-extract-truth`` for basic and optional Diffsky truth exports;
* ``openuniverse-inventory-truth`` for field inventory reports;
* ``amortized-train-openuniverse`` and ``amortized-infer-openuniverse``;
* standard-normal versus RealNVP redshift ablation;
* ``openuniverse-prior-overlap`` for truth versus posterior aggregate versus
  learned-prior comparisons;
* ``compare-fs2-openuniverse`` for magnitude, color-color, and redshift
  distribution diagnostics.
