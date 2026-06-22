Diffsky HLTDS Dataset
=====================

Dataset Choice
--------------

Four local HLTDS Diffsky artifacts are useful for the current validation
program. They should not be treated as interchangeable because they cover
different redshift ranges and have different error semantics.

.. list-table::
   :header-rows: 1

   * - Dataset
     - Local processed parquet
     - Objects / shards
     - Redshift range
     - Recommended role
   * - ``hltds_cosmos_260215_04_14_2026_continuous_lowz``
     - ``Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet``
     - ``78651`` objects from the 04/14 ``m5_depth`` prepared source
     - ``z = 0.0069 -- 0.3347``; median ``z = 0.250``
     - Main training, no-KL autoencoder, MAP, and inference dataset.
   * - ``hltds_cosmos_260215_04_14_2026_source_m5depth``
     - ``Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet``
     - ``369264`` objects across 16 source files
     - ``z = 0.007 -- 1.006``; median ``z = 0.693``
     - Full source artifact with the current synthetic ``fluxerr_*`` contract.
   * - ``hltds_cosmos_260215_04_14_2026_source_noerr``
     - ``Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet``
     - about 369k objects across 16 source files
     - ``z = 0.007 -- 1.006``; median ``z = 0.693``
     - Historical source artifact without ``fluxerr_*`` columns.
   * - ``hltds_cosmos_260215_03_31_2026``
     - ``Data/diffsky/processed/hltds_cosmos_260215_03_31_2026_photometry_truth.parquet``
     - about 494k objects across 6 source files
     - ``z = 1.055 -- 3.032``; median ``z = 2.429``
     - Better high-redshift candidate for photo-z calibration and population
       prior tests, after it is regenerated with the current no-error
       preparation/integrity contract.

The continuous low-z ``04_14`` subset is the current public config target. Its
redshift distribution is continuous and easier to learn than the multi-clump
full 04/14 source distribution, and the subset materializes an explicit
``fluxerr_*`` error model used by the likelihood and posterior-predictive
residual diagnostics. The full ``04_14`` ``m5_depth`` parquet is the source
artifact for rebuilding subsets. The ``03_31`` sample is still a useful
high-redshift candidate after it is regenerated with the same current
preparation and error-model contract.

.. image:: _static/diffsky_hltds_redshift_distributions.png
   :alt: Redshift distributions for the two Diffsky HLTDS dataset versions.
   :width: 95%

.. image:: _static/diffsky_hltds_parameter_distributions.png
   :alt: Redshift, stellar mass, sSFR, SFR, and dust distributions for the two Diffsky HLTDS dataset versions.
   :width: 95%

Truth/parameter ranges for the two samples:

.. list-table::
   :header-rows: 1

   * - Dataset
     - Quantity
     - 5%
     - Median
     - 95%
   * - ``03_31``
     - ``redshift_true``
     - ``1.062``
     - ``2.429``
     - ``3.017``
   * - ``04_14``
     - ``redshift_true``
     - ``0.201``
     - ``0.693``
     - ``0.997``
   * - ``03_31``
     - ``logsm_true``
     - ``8.992``
     - ``9.917``
     - ``10.881``
   * - ``04_14``
     - ``logsm_true``
     - ``9.115``
     - ``10.069``
     - ``10.998``
   * - ``03_31``
     - ``logssfr_true``
     - ``-12.025``
     - ``-9.386``
     - ``-8.693``
   * - ``04_14``
     - ``logssfr_true``
     - ``-12.969``
     - ``-10.232``
     - ``-9.312``
   * - ``03_31``
     - ``logsfr_true``
     - ``-1.925``
     - ``0.490``
     - ``1.540``
   * - ``04_14``
     - ``logsfr_true``
     - ``-3.198``
     - ``-0.293``
     - ``1.046``
   * - ``03_31``
     - ``dust_av``
     - ``0.148``
     - ``0.590``
     - ``1.978``
   * - ``04_14``
     - ``dust_av``
     - ``0.134``
     - ``0.570``
     - ``1.858``
   * - ``03_31``
     - ``dust_delta``
     - ``-0.305``
     - ``-0.172``
     - ``0.337``
   * - ``04_14``
     - ``dust_delta``
     - ``-0.306``
     - ``0.052``
     - ``0.344``

Both HLTDS versions expose the core truth columns used by the pipeline
(``redshift_true``, ``logsm_true``, ``logssfr_true``, ``logsfr_true``) and the
same generated-truth families used by the extended prior and true-parameter
closure diagnostics: ``diffstar_*``, ``diffmah_*``, ``dust_*``, and
``burst_*``.

Important caveat: the continuous low-z ``04_14`` subset and the locally
existing ``03_31`` parquet use synthetic fractional-SNR ``fluxerr_*`` columns.
Those errors are not native observational errors. They are explicit likelihood
assumptions and are recorded in the dataset manifest/schema.

The current validation dataset is:

.. code-block:: text

   hltds_cosmos_260215_04_14_2026_continuous_lowz

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

The normalized file used by the public Diffsky configs is now the continuous
low-z subset with materialized flux errors:

.. code-block:: text

   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet

It is rebuilt from the full prepared 04/14 source parquet with:

.. code-block:: bash

   python -m euclid_dsps.cli diffsky-redshift-subset \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet \
     --out Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet \
     --redshift-min 0.0 \
     --redshift-max 0.35 \
     --error-model m5_depth

The companion files are:

.. code-block:: text

   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.manifest.yaml
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.schema.json
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.summary.json
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.truth_summary.csv
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.truth_report.md
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.report.md
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr/flux_error_summary.csv
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr/redshift_true_distribution.png
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr/truth_distributions.png
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr/flux_fractional_error_by_band.png
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr/flux_snr_by_band.png
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr/flux_vs_fluxerr_by_band.png
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr/flux_error_model_curves_by_band.png
   Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr/flux_fractional_error_model_curves_by_band.png

The rebuilt subset should show no separated ``z~0.4`` island:

.. image:: _static/diffsky_04_14_continuous_z035_redshift_distribution.png
   :alt: Redshift distribution for the Diffsky 04/14 continuous z<0.35 subset.
   :width: 90%

.. image:: _static/diffsky_04_14_continuous_z035_truth_distributions.png
   :alt: Truth distributions for the Diffsky 04/14 continuous z<0.35 subset.
   :width: 90%

.. image:: _static/diffsky_04_14_continuous_z035_flux_fractional_error_by_band.png
   :alt: Synthetic fractional flux-error distributions by band.
   :width: 90%

.. image:: _static/diffsky_04_14_continuous_z035_flux_snr_by_band.png
   :alt: Synthetic catalog SNR distributions by band.
   :width: 90%

.. image:: _static/diffsky_04_14_continuous_z035_flux_vs_fluxerr_by_band.png
   :alt: Synthetic flux-error versus flux by band.
   :width: 90%

.. image:: _static/diffsky_04_14_continuous_z035_flux_error_model_curves_by_band.png
   :alt: Synthetic absolute flux-error model curves by band.
   :width: 90%

.. image:: _static/diffsky_04_14_continuous_z035_flux_fractional_error_model_curves_by_band.png
   :alt: Synthetic fractional flux-error model curves by band.
   :width: 90%

On Jean-Zay, the H100 Slurm scripts call the same
``diffsky-redshift-subset`` command automatically if the subset parquet is
missing and the full ``*_photometry_truth_m5depth.parquet`` source exists. This
keeps train/inference jobs from failing on a missing derived subset file.

Integrity Contract
------------------

Prepared datasets preserve the native ``core_tag`` column when it is present.
``object_id`` is guaranteed unique in the prepared parquet. If ``core_tag`` is
globally unique across all processed shards, ``object_id`` equals
``core_tag``. If duplicate ``core_tag`` values are detected across shards, the
preparer adds:

.. code-block:: text

   global_object_id
   source_file
   source_row

and sets ``object_id`` to ``global_object_id``. The manifest records
``object_id.strategy``, ``core_tag_unique_global``, and ``object_id_unique`` so
downstream reports do not silently treat duplicated source ids as unique
objects.

The schema JSON and manifest classify every prepared column into:

.. code-block:: text

   truth
   generated_truth
   derived_truth
   diagnostic
   proxy
   unavailable

This classification is also written to the truth report and the integrity
report. Diffstar, Diffmah, dust, and burst latent exports are
``generated_truth``: they are simulator parameters, not recovered quantities
from a photometric fit.

Photometry Contract
-------------------

The public Diffsky configs use 14 LSST+Roman bands:

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

The prepared source parquet keeps magnitude columns named ``mag_<band>`` and
flux-density columns named ``flux_<band>``. The main continuous low-z subset
uses the flux-density columns and writes ``fluxerr_<band>`` columns in
``fnu_cgs``:

.. code-block:: yaml

   bands: diffsky_hltds_lsst_roman_14_fnu_cgs
   fit:
     likelihood_space: flux
     photometric_likelihood: student_t
     student_t_dof: 2.0
     flux_error_floor_frac: 0.02

Native photometric error columns were not confirmed in the downloaded sample.
The adopted synthetic error model is therefore explicit, deterministic, and
flux dependent. For object ``i`` and band ``b``:

.. math::

   f_{5,b} = f_\nu(m_{5,b})

.. math::

   \sigma_{\mathrm{rand}, i b}^2 =
   (0.04-\gamma_b)\,|f_{i b}|\,f_{5,b}
   + \gamma_b\,f_{5,b}^2

Following the PhotErr convention, the materialized catalog error also includes
an irreducible systematic term in flux space:

.. math::

   \mathrm{sys\_frac} =
   10^{\sigma_{\mathrm{sys,mag}} / 2.5} - 1

.. math::

   \sigma_{\mathrm{cat}, i b}^2 =
   \sigma_{\mathrm{rand}, i b}^2 +
   \left(\mathrm{sys\_frac}\,|f_{i b}|\right)^2

The current default is ``sigma_sys_mag = 0.005``, i.e. a relative floor of
about ``0.46%`` in the materialized ``fluxerr_*`` columns.

Equivalently, with explicit terms:

.. math::

   \mathrm{flux\_limit\_5sigma}_b =
   f_\nu(\mathrm{magnitude\_limit\_5sigma}_b)

.. math::

   \mathrm{catalog\_error\_variance}_{i b} =
   \mathrm{source\_variance\_weight}_b\,
   |f_{i b}|\,\mathrm{flux\_limit\_5sigma}_b
   +
   \mathrm{background\_variance\_weight}_b\,
   \mathrm{flux\_limit\_5sigma}_b^2
   +
   \left(\mathrm{systematic\_flux\_fraction}\,
   |f_{i b}|\right)^2

with:

.. math::

   \mathrm{source\_variance\_weight}_b = 0.04-\gamma_b

.. math::

   \mathrm{background\_variance\_weight}_b = \gamma_b

and:

.. math::

   \mathrm{catalog\_error}_{i b} =
   \sqrt{\mathrm{catalog\_error\_variance}_{i b}}

The random term is the Rubin/LSST ``m5,gamma`` form rewritten in flux space,
matching the core point-source model used by PhotErr. At ``|f| = f5`` it gives
``sigma_rand = f5 / 5`` by construction, because ``m5`` is a 5-sigma limiting
magnitude. ``sigma_cat`` is then slightly larger when ``sigma_sys_mag > 0``.
The useful interpretation is:

.. math::

   \sigma_{\mathrm{cat}, b}^2 =
   \sigma_{\mathrm{depth}, b}^2 +
   \sigma_{\mathrm{source}, b}^2 +
   \sigma_{\mathrm{sys}, b}^2

with ``gamma_b f5_b^2`` acting as the depth/background term and
``(0.04-gamma_b)|f|f5_b`` acting as the source/Poisson-like term. The
``sigma_sys`` term is a small fractional photometric floor. What should
decrease with luminosity is the relative error ``sigma_flux / flux`` until this
floor is reached. The absolute error may increase for bright objects because
source noise and the systematic floor grow with flux, but it grows more slowly
than the flux until the fractional floor dominates, so the SNR still improves.

For Roman bands, there is no official Roman equivalent of the Rubin ``gamma``
parameter in the public WFI sensitivity tables. The current synthetic
approximation uses Roman WFI one-hour point-source 5-sigma depths and sets
``eta=0.95``, i.e. ``gamma=0.04*eta=0.038``. This keeps the depth term dominant
while adding a simple non-zero source/Poisson-like term. An ETC/Pandeia-derived
Roman SNR table should replace this approximation if a more realistic Roman
noise model is needed.

This is materialized once in the parquet; no new random noise draw is made
during MAP or amortized inference.

The active likelihood then inflates this catalog uncertainty with the configured
fractional floor. Current MAP/MCMC paths use the observed flux as the floor
reference:

.. math::

   \sigma_{\mathrm{eff}, i b}^2 =
   \sigma_{\mathrm{cat}, i b}^2 +
   \left(0.02\,|f_{i b}|\right)^2

The amortized JAX likelihood and posterior-predictive residual diagnostics use
the model flux as the floor reference:

.. math::

   \sigma_{\mathrm{eff}, i b}^2 =
   \sigma_{\mathrm{cat}, i b}^2 +
   \left(0.02\,|f_{\mathrm{model}, i b}|\right)^2

because the active configs set ``fit.flux_error_floor_frac: 0.02`` and
``fit.flux_error_jitter: 0.0`` / ``amortized.likelihood.error_jitter: 0.0``.
The ``fluxerr_*`` columns are catalog synthetic errors, including the
``0.005 mag`` photometric systematic floor. The configured ``2%`` fractional
floor is a separate DSPS/model/calibration tolerance. The floor-reference
difference is an implementation detail that should be unified before comparing
MAP and amortized likelihood values quantitatively.

The generated diagnostics show the two complementary views:
``flux_error_model_curves_by_band.png`` shows the absolute error versus flux,
and ``flux_fractional_error_model_curves_by_band.png`` shows the relative error
versus flux. With the current Roman ``eta=0.95`` setting and the PhotErr-style
systematic floor, the Roman absolute error curves are no longer pure depth
floors.

The dataset manifest uses explicit error model labels:

.. list-table::
   :header-rows: 1

   * - Label
     - Meaning
   * - ``native_error``
     - Error columns came from the source dataset.
   * - ``m5_depth``
     - Current synthetic depth model: Rubin/PhotErr random term plus
       ``sigma_sys_mag=0.005`` systematic floor in quadrature.
   * - ``fractional_snr``
     - Legacy/test synthetic ``fluxerr_* = abs(flux) / snr`` with a positive
       floor.
   * - ``synthetic_snr_error``
     - Historical alias for ``fractional_snr``.
   * - ``model_tolerance_mag``
     - Fit configuration uses a magnitude tolerance such as ``sigma_mag``.
   * - ``none``
     - No observation-error columns were written.

The historical HLTDS no-error prepared dataset uses ``error_model.type: none``
and must not be described as having native observational errors.

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

Readiness
---------

``diffsky-validate-dataset`` and the integrity report summarize readiness:

.. list-table::
   :header-rows: 1

   * - Status
     - Contract
   * - ``READY_BASIC``
     - Photometry plus direct redshift and stellar-mass truth are present.
   * - ``READY_EXTENDED``
     - Basic readiness plus Diffstar and Diffmah generated-truth exports.
   * - ``NOT_READY``
     - Missing photometry or required basic truth columns.

This readiness is for dataset integrity only. Physical recovery claims require
the later same-parameter forward closure and posterior calibration checks.

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

The main Diffsky joint-prior amortized config is:

.. code-block:: text

   configs/amortized_diffsky_hltds_joint_realnvp_gpu.yaml

It extends the shared base config
``configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml`` and explicitly sets
``amortized.prior.source: joint_realnvp``.

It uses the same 14 HLTDS LSST+Roman flux-density bands and trains a Gaussian
encoder plus RealNVP prior in the 12-parameter HLTDS PopCosmos-like latent:

.. code-block:: text

   z_obs
   log10_stellar_mass
   dlog10_sfr_1
   dlog10_sfr_2
   dlog10_sfr_3
   dlog10_sfr_4
   dlog10_sfr_5
   dlog10_sfr_6
   log10_stellar_metallicity
   tau2
   dust_index_n
   tau1_over_tau2

Gas and AGN parameters are not part of this HLTDS simplified decoder, but all
six PopCosmos-like SFH ratio bins are active. There should be no
``dlog10_sfr_i`` entries in ``model.fixed_parameters`` for the active 04/14
PopCosmos configs.

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
     --config configs/amortized_diffsky_hltds_joint_realnvp_gpu.yaml \
     amortized-train-diffsky \
     --limit 10000 \
     --batch-size 64 \
     --epochs 10 \
     --n-samples 2 \
     --out outputs/runs/amortized_diffsky_hltds_realnvp_n10000

Infer posterior samples under the learned prior:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/amortized_diffsky_hltds_joint_realnvp_gpu.yaml \
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
     --config configs/amortized_diffsky_hltds_joint_realnvp_gpu.yaml \
     amortized-prior-overlap-diffsky \
     --run outputs/runs/amortized_diffsky_hltds_realnvp_n10000_infer \
     --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet \
     --out outputs/runs/amortized_diffsky_hltds_realnvp_n10000_infer/prior_overlap \
     --max-objects 10000

The overlap report currently scores only directly comparable quantities:
``z_obs`` against ``redshift_true`` and ``log10_stellar_mass`` against
``logsm_true``. ``logsfr_true`` is kept in the dataset, but it is not equivalent
to the fitted ``dlog10_sfr_*`` ratios; do not call SFR recovered until a
derived-DSPS SFR diagnostic is used.
