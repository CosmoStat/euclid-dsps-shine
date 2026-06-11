Diffsky HLTDS Dataset
=====================

Dataset Choice
--------------

The current validation dataset is:

.. code-block:: text

   hltds_cosmos_260215_04_14_2026

from:

.. code-block:: text

   https://portal.nersc.gov/cfs/hacc/aphearin/diffsky_data/hltds_cosmos_260215_04_14_2026/

This dataset is preferred over the public OpenUniverse SkyCatalog parquet pair
for the current project because it exposes a cleaner local HDF5 sample with
photometry and useful physical columns in the same investigation workflow.
OpenUniverse tooling can remain in the package, but it is not the public
end-to-end path documented here.

Processed File
--------------

The normalized file used by the public Diffsky configs is:

.. code-block:: text

   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet

It is built with:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-prepare-dataset \
     --raw-root Data/diffsky/raw/hltds_cosmos_260215_04_14_2026 \
     --inventory outputs/diffsky_hltds_04_14_local_inventory.json \
     --out Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet \
     --no-synthetic-errors

The companion files are:

.. code-block:: text

   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_manifest.yaml
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_schema.json
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_truth_report.md

Photometry Contract
-------------------

The public simple configs use 14 native AB-magnitude bands:

.. code-block:: text

   lsst_u
   lsst_g
   lsst_r
   lsst_i
   lsst_z
   lsst_y
   roman_F062
   roman_F087
   roman_F106
   roman_F129
   roman_F146
   roman_F158
   roman_F184
   roman_F213

The prepared parquet keeps magnitude columns named ``mag_<band>``. Native
photometric error columns were not confirmed in the downloaded sample, so the
recommended prepared file does not write synthetic ``fluxerr_*`` values. The
fit configs use a magnitude-space tolerance:

.. code-block:: yaml

   bands: diffsky_hltds_lsst_roman_14_abmag_modelerr
   fit:
     likelihood_space: mag
     photometric_likelihood: gaussian

Each band has ``sigma_mag: 0.10``. This is a model-tolerance assumption for
MAP debugging, not a native survey error model.

Truth Policy
------------

The simple recovery path compares only direct/basic columns that are present in
the prepared dataset:

.. list-table::
   :header-rows: 1

   * - Column
     - Meaning in this pipeline
     - Fit use
   * - ``redshift_true``
     - Direct truth redshift.
     - Fit as ``z_obs`` in the simple config; fixed in the closure config.
   * - ``logsm_true``
     - Direct stellar-mass truth/proxy from the dataset.
     - Compared to recovered ``log10_stellar_mass``.
   * - ``logssfr_true``
     - Direct specific-SFR truth/proxy.
     - Used to derive ``logsfr_true`` when possible.
   * - ``logsfr_true``
     - ``logsm_true + logssfr_true`` when both are finite.
     - Compared to recovered recent-SFR proxy.
   * - ``logmp_true`` and ``logmp_host_true``
     - Halo-mass truth/proxy columns when present.
     - Stored for diagnostics, not fitted by the simple DSPS config.
   * - ``central_true``
     - Central/satellite flag when present.
     - Stored for diagnostics.
   * - ``r50_disk_true`` and ``r50_bulge_true``
     - Size proxies when present.
     - Stored for diagnostics.

Diffstar/Diffmah latent columns, dust-generation parameters, metallicity
latents, and halo MAH parameters are not fitted by the public simple configs.
If they are present, they can be inventoried and kept for later population
diagnostics, but broad-band DSPS MAP recovery should not label them as
recovered physical truths unless the forward model and parameterization are
explicitly matched.

Public Fit Configs
------------------

Free-redshift simple fit:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_hltds_04_14_simple_gpu.yaml \
     fit \
     --limit 1000 \
     --batch-size 128 \
     --fit-maxiter 220 \
     --sed-samples 0 \
     --reporting-level light \
     --out outputs/runs/diffsky_hltds_simple_n1000

Fixed-redshift closure:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml \
     fit \
     --limit 128 \
     --batch-size 128 \
     --fit-maxiter 180 \
     --sed-samples 0 \
     --reporting-level light \
     --out outputs/runs/diffsky_hltds_fixedz_closure_n128

Fit Report
----------

After a MAP run, regenerate the Diffsky-specific report:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-fit-report \
     --run outputs/runs/diffsky_hltds_simple_n1000 \
     --config configs/diffsky_hltds_04_14_simple_gpu.yaml \
     --label batch_fit \
     --reporting-level light

The report summarizes photometric residuals, objective components, optimizer
diagnostics, and truth recovery for columns that exist in the dataset. It is
the first place to check for redshift collapse, mass bias, band calibration
problems, or a mismatch between HLTDS photometry and the simplified DSPS model.

Amortized Prior Learning
------------------------

The main Diffsky amortized config is:

.. code-block:: text

   configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml

It uses the same 14 HLTDS LSST+Roman AB-magnitude bands, converts them to the
internal flux-density representation for the fixed DSPS decoder, and trains a
Gaussian encoder plus RealNVP prior in a compact 9-parameter latent:

.. code-block:: text

   z_obs
   log10_stellar_mass
   dlog10_sfr_1
   dlog10_sfr_2
   dlog10_sfr_3
   log10_stellar_metallicity
   tau2
   dust_index_n
   tau1_over_tau2

The remaining PopCosmos-bin SFH and nuisance parameters are fixed from
``model.fixed_parameters``. This is deliberate: broad-band photometry is
degenerate in age, metallicity, dust, SFR history, and redshift, so this path
learns a population prior over the degenerate physical manifold instead of
forcing an over-parameterized per-object recovery.

The config is conservative for CUDA stability:

.. code-block:: yaml

   model:
     compressed_ssp_runtime_dtype: float32
   amortized:
     training:
       jax_batch_size: 4
     inference:
       jax_batch_size: 4

The compressed SSP can remain ``coeff16`` on disk, but it is upcast to
``float32`` when resident on GPU. ``jax_batch_size`` caps the actual compiled
DSPS batch size. You can still pass ``--batch-size 32`` or ``--batch-size 64``;
the command will log the cap and process multiple safe micro-batches.

Before training, build the compressed HLTDS SSP asset described in
:doc:`data_download`. Then run:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml \
     amortized-train-diffsky \
     --limit 10000 \
     --batch-size 64 \
     --epochs 10 \
     --n-samples 2 \
     --out outputs/runs/amortized_diffsky_hltds_realnvp_n10000

Infer posterior samples under the learned prior:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml \
     amortized-infer-diffsky \
     --checkpoint outputs/runs/amortized_diffsky_hltds_realnvp_n10000/checkpoints/best.eqx \
     --limit 10000 \
     --batch-size 64 \
     --posterior-samples 64 \
     --prior-samples 8192 \
     --out outputs/runs/amortized_diffsky_hltds_realnvp_n10000_infer

Compare truth, aggregate posterior, and learned prior:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml \
     amortized-prior-overlap-diffsky \
     --run outputs/runs/amortized_diffsky_hltds_realnvp_n10000_infer \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet \
     --out outputs/runs/amortized_diffsky_hltds_realnvp_n10000_infer/prior_overlap \
     --max-objects 10000

The overlap report currently scores only directly comparable quantities:
``z_obs`` against ``redshift_true`` and ``log10_stellar_mass`` against
``logsm_true``. ``logsfr_true`` is kept in the dataset, but it is not equivalent
to the fitted ``dlog10_sfr_*`` ratios; do not call SFR recovered until a
derived-DSPS SFR diagnostic is used.
