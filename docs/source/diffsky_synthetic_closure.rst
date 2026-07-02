Diffsky FENIKS DSPS Closure Dataset
===================================

This workflow builds a synthetic closure dataset from a Diffsky/FENIKS latent
population and then regenerates the final photometry with this repository's
``euclid_dsps`` forward model.

No external N-body halo catalog is required. The proposal population is drawn
with Diffsky's analytic weighted lightcone generator,
``weighted_lc_photdata``. Diffsky/FENIKS is used only to sample correlated
latent galaxy parameters and proposal weights. The final closure fluxes are not
``phot_info.obs_mags``; they are recomputed from the 18 recorded truths with
the same DSPS parameterization, SSP, filters, cosmology, dust model,
metallicity model, and IGM model used later for inference.

Outputs
-------

The production config writes:

.. code-block:: text

   Data/diffsky/synthetic/feniks_260617_dsps_closure/
       proposals/train/
       proposals/validation/
       proposals/test/
       train.parquet
       validation.parquet
       test.parquet
       all_50k.parquet
       manifest.yaml
       schema.json
       diagnostics/
          population/
       validation_report.json

The proposal shards keep ``cen_weight``, ``sat_weight`` and
``galaxy_weight = cen_weight * sat_weight``. The train, validation and test
catalogs are independent weighted-resampling outputs from independent Diffsky
proposal pools and seeds. They are not random splits of one parent catalog.

Redshift Range and Realism Checks
---------------------------------

The production FENIKS closure config is set to ``0.001 <= z <= 3.0``. This is a
broad LSST+Roman/OpenUniverse-like photometric range. OpenUniverse2024 validates
its extragalactic Diffsky-based catalog with galaxy number counts, redshift
distributions in magnitude bins, and optical/NIR color evolution; the same
classes of checks are written automatically by this generator. The current local
z<=0.35 HLTDS parquet is still useful as a low-redshift reference, but
comparisons to it are restricted to the overlapping low-redshift interval.

Every generation run with ``synthetic_diffsky.diagnostics.enabled: true`` writes:

.. code-block:: text

   diagnostics/population/
       population_diagnostics_summary.json
       report.md
       parameter_stats.csv
       photometry_stats.csv
       color_stats.csv
       proposal_vs_final_metrics.csv
       correlation_matrices.json
       plots/
           truth_parameter_histograms.png
           physical_diagnostic_histograms.png
           magnitude_histograms.png
           color_histograms.png
           photometry_band_summary.png
           mass_redshift_sfr_dust.png
           corner_core_truths.png
           corner_18_truths.png
       reference_comparison/

These outputs are part of the scientific acceptance checks. FENIKS supplies a
calibrated Diffsky population prior, but realism for a specific training set is
established by the generated diagnostics after applying proposal weights,
resampling, redshift cuts, and any survey-like selection.

Metallicity Convention
----------------------

The closure uses:

.. code-block:: yaml

   model:
     sfh_model: diffsky_basic
     stellar_metallicity_model: lognormal_mdf_fixed_scatter
     stellar_metallicity_scatter_dex: 0.2

``log10_stellar_metallicity_true`` is the median ``log10(Z/Z_sun)`` of the
stellar MDF. The MDF scatter is a fixed internal-galaxy hyperparameter, not a
second fitted latent and not an inter-galaxy random draw around the median.

The FENIKS mass-metallicity-time relation returns absolute ``log10(Z)``:

.. code-block:: python

   lgmet_abs_median = dsps.metallicity.umzr.mzr_model(
       phot_info.logsm_obs,
       lc_data.t_obs,
       *feniks_params.mzr_params,
   )

The catalog stores ``lgmet_abs_median_true`` and converts the fitted truth to
``log10(Z/Z_sun)`` using exactly ``model.z_sun``. If a median falls outside the
SSP metallicity grid, clipping only happens when
``synthetic_diffsky.metallicity_grid_policy`` explicitly requests it. Clipped
counts and fractions are written to ``manifest.yaml`` and checked during
validation.

Ground Truth Contract
---------------------

The schema is ``diffsky_dsps_closure_full``. It requires exactly one truth
column for each free parameter in ``DIFFSKY_BASIC_PARAMETER_NAMES``.

.. list-table::
   :header-rows: 1

   * - Free parameter
     - Truth column
     - Source
     - Transformation
     - Units/convention
     - Bounds
   * - ``z_obs``
     - ``redshift_true``
     - ``lc_data.z_obs``
     - none
     - dimensionless
     - ``[0.001, 3.0]``
   * - ``log10_stellar_mass``
     - ``logsm_true``
     - ``phot_info.logsm_obs``
     - none
     - ``log10(Mstar/Msun)``
     - ``[6.0, 13.5]``
   * - ``diffstar_lgmcrit``
     - ``diffstar_lgmcrit_true``
     - ``phot_info.lgmcrit``
     - none
     - Diffstar native
     - ``[9.0, 14.5]``
   * - ``diffstar_lgy_at_mcrit``
     - ``diffstar_lgy_at_mcrit_true``
     - ``phot_info.lgy_at_mcrit``
     - none
     - Diffstar native
     - ``[-13.0, -8.0]``
   * - ``diffstar_indx_lo``
     - ``diffstar_indx_lo_true``
     - ``phot_info.indx_lo``
     - none
     - Diffstar native
     - ``[-2.0, 6.0]``
   * - ``diffstar_indx_hi``
     - ``diffstar_indx_hi_true``
     - ``phot_info.indx_hi``
     - none
     - Diffstar native
     - ``[-6.0, 3.0]``
   * - ``diffstar_lg_qt``
     - ``diffstar_lg_qt_true``
     - ``phot_info.lg_qt``
     - none
     - Diffstar native
     - ``[-2.0, 2.5]``
   * - ``diffstar_qlglgdt``
     - ``diffstar_qlglgdt_true``
     - ``phot_info.qlglgdt``
     - none
     - Diffstar native
     - ``[-4.0, 4.0]``
   * - ``diffstar_lg_drop``
     - ``diffstar_lg_drop_true``
     - ``phot_info.lg_drop``
     - none
     - Diffstar native
     - ``[-4.0, 1.0]``
   * - ``diffstar_lg_rejuv``
     - ``diffstar_lg_rejuv_true``
     - ``phot_info.lg_rejuv``
     - none
     - Diffstar native
     - ``[-4.0, 2.0]``
   * - ``diffmah_logm0``
     - ``diffmah_logm0_true``
     - ``lc_data.mah_params.logm0``
     - not ``lc_data.logmp0``
     - Diffmah native
     - ``[7.0, 16.0]``
   * - ``diffmah_logtc``
     - ``diffmah_logtc_true``
     - ``lc_data.mah_params.logtc``
     - none
     - Diffmah native
     - ``[-2.5, 1.5]``
   * - ``diffmah_early_index``
     - ``diffmah_early_index_true``
     - ``lc_data.mah_params.early_index``
     - none
     - Diffmah native
     - ``[-1.0, 6.0]``
   * - ``diffmah_late_index``
     - ``diffmah_late_index_true``
     - ``lc_data.mah_params.late_index``
     - none
     - Diffmah native
     - ``[-1.0, 3.0]``
   * - ``diffmah_t_peak``
     - ``diffmah_t_peak_true``
     - ``lc_data.mah_params.t_peak``
     - none
     - Gyr
     - ``[0.0, 15.0]``
   * - ``log10_stellar_metallicity``
     - ``log10_stellar_metallicity_true``
     - FENIKS UMZR
     - ``log10(Z) - log10(model.z_sun)``
     - median ``log10(Z/Z_sun)``
     - ``[-2.5, 0.5]``
   * - ``dust_av``
     - ``dust_av_true``
     - ``phot_info.av``
     - none
     - magnitudes
     - ``[0.0, 5.0]``
   * - ``dust_delta``
     - ``dust_delta_true``
     - ``phot_info.delta``
     - none
     - attenuation slope offset
     - ``[-2.5, 1.0]``

Commands
--------

CPU smoke generation still requires Diffsky, Diffstar and Diffmah for the
science backend:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-generate-dsps-closure \
     --smoke \
     --overwrite

Validate the generated dataset:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_trueparam_closure.yaml \
     diffsky-validate-dsps-closure \
     --dataset-dir Data/diffsky/synthetic/feniks_260617_dsps_closure

Production generation:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-generate-dsps-closure \
     --split all \
     --resume

Generation is verbose by default. It reports split/shard progress, proposal
pool size, ESS, resampling duplication, DSPS photometry batches, and the
application of the configured flux-error model. The production configuration
keeps population plots enabled and writes, in addition to the core catalogues:

- ``diagnostics/population/parameter_stats.csv``;
- ``diagnostics/population/photometry_stats.csv``;
- ``diagnostics/population/error_model_stats.csv``;
- ``diagnostics/population/color_stats.csv``;
- ``diagnostics/population/proposal_vs_final_metrics.csv``;
- ``diagnostics/population/plots/error_model_band_summary.png``;
- ``diagnostics/population/plots/normalized_noise_residual_histograms.png``;
- ``diagnostics/population/plots/fluxerr_vs_mag_true.png``;
- ``diagnostics/population/plots/corner_18_truths.png`` when enough dynamic
  range is available.

Run exact forward closure on the test set:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_trueparam_closure.yaml \
     diffsky-forward-closure \
     --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet \
     --limit 1024 \
     --out outputs/runs/diffsky_synthetic_feniks_trueparam_closure_smoke

Train the supervised multivariate prior:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml \
     diffsky-train-supervised-prior \
     --out outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp

Train amortized inference with the learned prior frozen:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
     amortized-train-diffsky \
     --out outputs/runs/amortized_diffsky_synthetic_feniks_full

After inference on the held-out test set, evaluate calibration metrics:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
     diffsky-evaluate-dsps-closure-inference \
     --run outputs/runs/amortized_diffsky_synthetic_feniks_full_infer \
     --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet \
     --out outputs/runs/amortized_diffsky_synthetic_feniks_full_eval

Scientific Limits
-----------------

There are 18 latent parameters and 14 broad-band fluxes. The intended success
criteria are calibrated posteriors, posterior predictive checks, faithful
multivariate prior learning and explicit identification of weakly constrained
directions. Exact per-galaxy recovery of every latent is not expected.
