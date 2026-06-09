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
   galaxy_sed_<hpix>.hdf5  # optional low-resolution generated SED product

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
OpenUniverse preparation step keeps this native unit internally and preserves
the public truth fluxes as ``flux_truth_<band>``.

For DSPS fit and amortized smoke tests, build a second derived parquet in
``fnu_cgs`` with explicit lensing and filter-response conventions. The derived
table preserves the original public photon columns as
``flux_truth_lensed_photon_<band>``, ``flux_lensed_photon_<band>``, and
``fluxerr_lensed_photon_<band>``. It also writes
``flux_truth_unlensed_photon_<band>`` and related columns by dividing by
``mu_lensing``:

.. code-block:: text

   mu_lensing = 1 / ((1 - convergence)^2 - shear1^2 - shear2^2)

The standard ``flux_*`` and ``fluxerr_*`` columns in the fit-ready table are
then converted to AB-equivalent ``fnu_cgs`` using the per-band photon rate of a
flat 0 AB source. By default this conversion clips filter responses to
``[0, 1]`` before computing the zero point, matching
``euclid_dsps.filters.load_ascii_filter``. This matters for Roman WFI files
whose second column is an effective-area-like response with values above one.

Tracked unit TODOs:

* choose the definitive internal unit for OpenUniverse training;
* continue validating Roman WFI zeropoints and response conventions;
* decide whether production should use DSPS ``fnu_cgs`` photometry, a
  photon-rate decoder, or an explicitly calibrated band conversion.

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

Truth and SED Inventory
-----------------------

After downloading the optional SED HDF5 for the same HEALPix, inventory the
public truth fields and SED layout without scanning the full payload:

.. code-block:: bash

   python -m euclid_dsps.openuniverse.cli inventory-truth \
     --input Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
     --input-root Data/openuniverse/raw \
     --hpix 10307 \
     --sed \
     --sed-sample-limit 3 \
     --out outputs/reports/openuniverse_truth_inventory_10307

This writes:

.. code-block:: text

   openuniverse_truth_inventory.md
   openuniverse_truth_inventory.json
   truth_schema.json

For the preview ``10307`` files inspected locally, the SED HDF5 contains
``meta/wave_list`` with 312 wavelengths and per-galaxy datasets shaped
``(3, 312)``. The component rows are exposed as ``disk``, ``bulge``, and
``knot`` when that 3-component shape is present.

Basic Truth Export
------------------

Direct public OpenUniverse truths can be exported from the prepared parquet:

.. code-block:: bash

   python -m euclid_dsps.openuniverse.cli extract-truth \
     --input Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
     --out Data/openuniverse/processed/ou_truth_basic.parquet \
     --schema-out Data/openuniverse/processed/truth_schema.json

The basic export includes only quantities directly present in the table, such
as ``galaxy_id``, ``redshift_truth``, ``redshift_hubble_truth``, and
``stellar_mass_truth``. Missing SFH, Diffstar, internal dust, metallicity, and
halo latent parameters remain ``unavailable``.

OpenUniverse Feature Stats
--------------------------

The prepared table can be loaded into generic ``PhotometryArrays`` and used to
compute amortized encoder feature statistics:

.. code-block:: bash

   python -m euclid_dsps.openuniverse.cli feature-stats \
     --input Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
     --limit 10000 \
     --out outputs/runs/openuniverse_feature_stats_10307/feature_stats.json

For LSST+Roman this validates ``n_bands=14`` and ``feature_dim=28`` before any
physical DSPS decoder is connected.

Fit-Ready DSPS Parquet
----------------------

After preparing the photon-rate table and downloading the main parquet for the
same HEALPix, create the DSPS-compatible table:

.. code-block:: bash

   python -m euclid_dsps.openuniverse.cli make-fit-ready \
     --input Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
     --main Data/openuniverse/raw/galaxy_10307.parquet \
     --out Data/openuniverse/processed/ou_lsst_roman_14_subset_fit_ready.parquet

The command writes a sibling manifest containing the band list, AB0 photon-rate
zero points, lensing summary, filter sources, and
``filter_response_mode: dsps_clipped``. Use
``configs/openuniverse_lsst_roman_14_fit_ready.yaml`` for per-object MAP smoke
tests and
``configs/amortized_openuniverse_lsst_roman_fit_ready_realnvp.yaml`` for the
14-band amortized path.

Minimal checks:

.. code-block:: bash

   python -m euclid_dsps.openuniverse.cli feature-stats \
     --input Data/openuniverse/processed/ou_lsst_roman_14_subset_fit_ready.parquet \
     --limit 10000 \
     --out outputs/runs/openuniverse_fit_ready_feature_stats_10307/feature_stats.json

   python -m euclid_dsps.cli \
     --config configs/openuniverse_lsst_roman_14_fit_ready.yaml \
     fit \
     --index 0 \
     --fit-maxiter 30 \
     --out outputs/runs/dev_openuniverse_fit_ready_one \
     --sed-samples 1

The current fit-ready path is good enough for smoke tests and data-contract
validation. Roman bands still need zeropoint/response validation before the
resulting MAP parameters should be interpreted as science-grade physical
inference.

SED-to-Flux Closure
-------------------

The data-side closure path projects the generated SED HDF5 through filter
curves and compares the resulting photon rates to the public OpenUniverse flux
table. This does not touch the DSPS decoder.

For LSST, exact repository filter files are available:

.. code-block:: bash

   python -m euclid_dsps.openuniverse.cli sed-flux-closure \
     --catalog Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
     --sed Data/openuniverse/raw/galaxy_sed_10307.hdf5 \
     --bands lsst_u lsst_g lsst_r lsst_i lsst_z lsst_y \
     --limit 200 \
     --out outputs/reports/openuniverse_sed_flux_closure_10307_lsst200

Outputs:

.. code-block:: text

   sed_flux_closure_rows.parquet
   sed_flux_closure_metrics.csv
   sed_flux_closure_calibration.csv
   sed_flux_closure_summary.json

Roman bands require exact Roman filter curves supplied with repeated
``--filter band=/path/to/filter.dat`` options. The command has
``--allow-approx-filters`` for smoke tests only; those Roman metrics are not
science-grade.

External Diffsky Truth Merge
----------------------------

Full Diffsky/Diffstar latents are not present in the public main/flux/SED files
inspected so far. If a separate generation/export table becomes available, merge
it explicitly:

.. code-block:: bash

   python -m euclid_dsps.openuniverse.cli merge-external-truth \
     --input Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
     --truth Data/openuniverse/processed/diffsky_latents_export.parquet \
     --out Data/openuniverse/processed/ou_with_diffsky_latents.parquet \
     --schema-out Data/openuniverse/processed/diffsky_latents_schema.json \
     --truth-level generated_truth \
     --truth-column diffstar_u_param \
     --truth-column halo_mass

This command only labels columns according to the caller-provided
``--truth-level``. It does not reconstruct missing latents and does not promote
proxy columns to truth.

Next Phases
-----------

Planned follow-up commands and reports:

* ``amortized-train-openuniverse`` and ``amortized-infer-openuniverse``;
* standard-normal versus RealNVP redshift ablation;
* ``openuniverse-prior-overlap`` for truth versus posterior aggregate versus
  learned-prior comparisons;
* ``compare-fs2-openuniverse`` for magnitude, color-color, and redshift
  distribution diagnostics.
