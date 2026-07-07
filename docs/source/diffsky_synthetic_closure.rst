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

The survey-like 18-band configuration writes a layered product:

.. code-block:: text

   Data/diffsky/synthetic/feniks_260617_dsps_closure_18band/
       proposals/                 # raw weighted Diffsky proposals
       survey_like/               # observable-selected LSST+Euclid+Roman sample
          train.parquet
          validation.parquet
          test.parquet
          all.parquet
       inference_ready/           # stricter closure/inference sample
          train.parquet
          validation.parquet
          test.parquet
          all_50k.parquet
       train.parquet              # mirrors inference_ready/train.parquet
       validation.parquet
       test.parquet
       all_50k.parquet
       diagnostics/

``raw_weighted`` is represented by the proposal shards and must be used with
``galaxy_weight`` for weighted population diagnostics. ``survey_like`` applies
observable magnitude/S/N cuts and is the right layer for realism checks.
``inference_ready`` applies a stricter minimum number of detected bands and is
the default layer used by closure validation and amortized inference.

Creation Process
----------------

The production command is intentionally a two-population workflow:

1. Diffsky/FENIKS generates a raw weighted proposal lightcone for each split.
   The generator calls ``weighted_lc_photdata`` and ``mc_lc_phot`` with the
   FENIKS calibration, independent split seeds, and ``mc_merge: 0``. These raw
   proposal shards are written under ``proposals/<split>/`` and are kept for
   weighted diagnostics such as ``n(z)``. They are not the final learning
   catalog.
2. The generator extracts compact truth scalars from Diffsky immediately:
   redshift, stellar mass, Diffstar parameters, Diffmah parameters, dust,
   central/satellite state, SFR diagnostics, halo-mass diagnostics, and proposal
   weights. Large intermediate tensors such as full SFHs, SEDs, SSP weights and
   transmission tables are not stored per object.
3. The FENIKS mass-metallicity-time relation supplies
   ``lgmet_abs_median_true`` in absolute ``log10(Z)``. The code clips this
   absolute value only against the absolute SSP metallicity grid when explicit
   clipping is configured, stores ``lgmet_abs_used_true``, and then computes
   ``log10_stellar_metallicity_true = lgmet_abs_used_true - log10(model.z_sun)``.
   Production selection requires no clipped metallicities in the final catalog.
4. Proposal-level selection is applied before weighted resampling. The legacy
   14-band production defaults keep ``logsm_true >= 8`` and reject clipped
   metallicities. The 18-band survey-like configuration removes the stellar
   mass cut from the final survey-like definition and keeps only the explicit
   metallicity-grid guardrail. Selection counters are written to
   ``manifest.yaml`` under ``proposal_selection``; the raw shards remain on
   disk unchanged.
5. The selected proposal pool is resampled with replacement using probabilities
   proportional to ``galaxy_weight``. ESS, weight sums, pool size and duplicate
   fraction are recorded. The final splits are independent because they are
   generated from independent proposal pools and seeds.
6. If photometric selection is enabled, the code first draws an oversampled
   candidate catalog, runs the DSPS closure photometer, and then keeps rows
   satisfying the configured observable gates. Gates can combine magnitude
   limits, S/N thresholds, and minimum detected-band counts. The 18-band config
   writes both a looser ``survey_like`` layer and a stricter
   ``inference_ready`` layer. This creates observable learning samples rather
   than volume-complete samples dominated by non-detections.
7. Final true fluxes are generated only with this repository's DSPS forward
   model using the 18 stored truths in ``DIFFSKY_BASIC_PARAMETER_NAMES`` order.
   The same SSP, filters, cosmology, dust, metallicity and IGM settings are used
   by closure validation and inference. Diffsky ``phot_info.obs_mags`` are not
   used as closure fluxes.
8. The configured LSST+Roman error model writes ``fluxerr_<band>`` and draws
   noisy ``flux_<band> = flux_true_<band> + Normal(0, fluxerr_<band>)``.
   Negative noisy fluxes are preserved. S/N-count diagnostic columns are kept in
   the final catalog.
9. The generator writes ``train.parquet``, ``validation.parquet``,
   ``test.parquet``, ``all_50k.parquet``, ``schema.json``, ``manifest.yaml`` and
   population diagnostics. Validation then recomputes DSPS fluxes from a random
   truth sample, checks exact split sizes and disjoint identifiers, checks
   metallicity-grid conventions, checks the S/N selection, checks noise
   residuals, and compares the selected proposal ``n(z)`` to the final catalog
   where appropriate.

Redshift Range and Realism Checks
---------------------------------

The production FENIKS closure proposal range is set to ``0.001 <= z <= 5.5``.
The final learning catalog is not a raw volume-complete proposal: it is an
observable-population sample after the explicit selection described above.
OpenUniverse2024 validates its extragalactic Diffsky-based catalog with galaxy
number counts, redshift distributions in magnitude bins, and optical/NIR color
evolution; the same classes of checks are written automatically by this
generator. The current local z<=0.35 HLTDS parquet is still useful as a
low-redshift reference, but comparisons to it are restricted to the overlapping
low-redshift interval.

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

FS2 / Euclid Comparison
-----------------------

``configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml`` adds
Euclid VIS/Y/J/H filters to the LSST+Roman closure bands and writes an automatic
FS2 comparison when ``Data/Euclid FS2 LC galaxy catalog_phz1.parquet`` is
available. The FS2 comparison uses like-for-like LSST and Euclid color-color
planes. Roman colors are still generated for the synthetic catalog, but they
are not interpreted as direct FS2 matches because FS2 does not contain Roman
bands.

Manual FS2 comparison:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml \
     diffsky-compare-dsps-closure-reference \
     --synthetic Data/diffsky/synthetic/feniks_260617_dsps_closure_18band/survey_like/all.parquet \
     --reference "Data/Euclid FS2 LC galaxy catalog_phz1.parquet" \
     --reference-kind fs2 \
     --out outputs/audits/feniks18_survey_like_vs_fs2 \
     --max-reference 100000

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

The catalog stores both ``lgmet_abs_median_true`` and ``lgmet_abs_used_true``
and converts the fitted truth to ``log10(Z/Z_sun)`` using exactly
``model.z_sun``. If a median falls outside the SSP metallicity grid, clipping
only happens when ``synthetic_diffsky.metallicity_grid_policy`` explicitly
requests it. Clipped counts and fractions are written to ``manifest.yaml`` and
checked during validation. The production selection rejects clipped
metallicities from the final learning catalog.

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
     - ``[0.001, 5.5]``
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

The current production-defining configuration is:

.. code-block:: yaml

   synthetic_diffsky:
     z_min: 0.001
     z_max: 5.5
     metallicity_grid_policy: clip_with_warning
     stellar_metallicity_scatter_dex: 0.2
     selection:
       min_logsm: 8.0
       require_metallicity_unclipped: true
       max_metallicity_clipped_fraction: 0.0
       snr_threshold: 5.0
       min_true_snr_bands: 5
       min_observed_snr_bands: 0
       photometric_oversample_factor: 5.0

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
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-validate-dsps-closure \
     --dataset-dir Data/diffsky/synthetic/feniks_260617_dsps_closure

Production generation:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
     diffsky-generate-dsps-closure \
     --split all \
     --overwrite

Survey-like LSST+Euclid+Roman generation:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml \
     diffsky-generate-dsps-closure \
     --split all \
     --overwrite

Validate the 18-band inference-ready layer mirrored at the dataset root:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml \
     diffsky-validate-dsps-closure \
     --dataset-dir Data/diffsky/synthetic/feniks_260617_dsps_closure_18band \
     --sample-size 512 \
     --batch-size 256 \
     --runtime gpu

Use ``--resume`` instead of ``--overwrite`` only when continuing an interrupted
generation with compatible configuration and proposal shards.

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
- ``diagnostics/population/reference_comparison/plots/color_color_reference_black_synthetic_green.png``
  for OpenUniverse-style color-color overlays, with black points for the
  z<=0.35 reference sample and green points for the synthetic DSPS-closure
  catalog in the overlapping redshift range.

The production resampling gate treats 10% duplicate source proposals as a
warning threshold. The generator keeps accumulating proposal shards while it can
reduce duplication, but with the high-dynamic-range FENIKS weights the z<=5.5
train pool can still have duplicate_fraction around 0.15 after all 256 shards.
In that case ``duplication_gate: warn_after_max_shards`` lets the run continue
only if the selected-pool-size and ESS gates pass. The realized pool and final
duplication fractions are recorded in ``manifest.yaml`` and should be reported
with any science use.

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
