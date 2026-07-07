Forward Model
=============

Public Science Setups
---------------------

The public forward-model setups are:

.. list-table::
   :header-rows: 1

   * - Config
     - Dataset
     - Model role
   * - ``configs/diffsky_synthetic_feniks_260617_50k.yaml``
     - Synthetic FENIKS/DSPS closure
     - Production 18D ``diffsky_basic`` closure model with local DSPS true
       photometry.
   * - ``configs/fs2_gpu.yaml``
     - Euclid FS2
     - Full PopCosmos-like DSPS baseline with compressed stellar, gas, and AGN assets.
   * - ``configs/diffsky_hltds_04_14_simple_gpu.yaml``
     - Diffsky HLTDS 04/14/2026
     - Simplified DSPS recovery model with no AGN and fixed nebular treatment.
   * - ``configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml``
     - Diffsky HLTDS 04/14/2026
     - Same simplified model with redshift fixed to ``redshift_true`` for closure checks.

The synthetic closure path uses Diffstar/Diffmah latent parameters through the
``diffsky_basic`` DSPS wrapper. The HLTDS PopCosmos proxy path does not recover
Diffstar/Diffmah latents directly; those columns are inventoried and used only
where a matched truth schema is configured.

Synthetic FENIKS Closure Model
------------------------------

The production synthetic closure config uses fourteen LSST+Roman bands:

.. code-block:: text

   LSST u,g,r,i,z,y + Roman F062,F087,F106,F129,F146,F158,F184,F213

The free parameter vector is the 18-parameter ``diffsky_basic`` vector:

.. code-block:: text

   z_obs
   log10_stellar_mass
   diffstar_lgmcrit
   diffstar_lgy_at_mcrit
   diffstar_indx_lo
   diffstar_indx_hi
   diffstar_lg_qt
   diffstar_qlglgdt
   diffstar_lg_drop
   diffstar_lg_rejuv
   diffmah_logm0
   diffmah_logtc
   diffmah_early_index
   diffmah_late_index
   diffmah_t_peak
   log10_stellar_metallicity
   dust_av
   dust_delta

During generation, these truths are decoded with the local ``euclid_dsps``
model to write ``flux_true_<band>``. The configured error model then writes
``fluxerr_<band>`` and noisy ``flux_<band>``. Validation recomputes the true
fluxes from the stored truth vector and checks that the saved true photometry
is same-model consistent.

FS2 Baseline
------------

``configs/fs2_gpu.yaml`` uses ten bands:

.. code-block:: text

   LSST u,g,r,i,z,y + Euclid VIS,Y,J,H

The free parameter vector is the 16-parameter PopCosmos-like DSPS vector:

.. code-block:: text

   z_obs
   log10_stellar_mass
   dlog10_sfr_1 ... dlog10_sfr_6
   log10_stellar_metallicity
   tau2
   dust_index_n
   tau1_over_tau2
   log10_gas_metallicity
   log10_gas_ionization
   ln_fagn
   ln_tauagn

The model uses:

* compressed stellar SSP basis;
* compressed gas grid basis;
* compressed AGN component basis;
* Prospector/FSPS-style dust;
* FSPS Madau95 IGM;
* Student-t flux-space likelihood.

This setup remains useful for Euclid/FS2 comparisons and for FS2 amortized
prior learning.

Diffsky HLTDS Simple Model
--------------------------

The Diffsky simple configs intentionally remove poorly constrained components:

* no AGN;
* no gas-metallicity or gas-ionization fit;
* no Diffstar/Diffmah latent recovery;
* all six PopCosmos-like SFH ratio bins free;
* flux-space Student-t likelihood using the explicit synthetic ``fluxerr_*``
  model plus a fractional likelihood floor.

The free-redshift config fits:

.. code-block:: text

   z_obs
   log10_stellar_mass
   dlog10_sfr_1 ... dlog10_sfr_6
   log10_stellar_metallicity
   tau2
   dust_index_n
   tau1_over_tau2

The fixed-redshift closure config fits:

.. code-block:: text

   log10_stellar_mass
   dlog10_sfr_1 ... dlog10_sfr_6
   log10_stellar_metallicity
   tau2
   dust_index_n
   tau1_over_tau2

These parameters are compared only to direct/basic HLTDS truth columns:
``redshift_true``, ``logsm_true``, and ``logsfr_true``. Halo mass, central flag,
and size columns are stored for diagnostics but are not fitted by DSPS.

SSP Assets
----------

FS2 uses local compressed FSPS assets:

.. code-block:: text

   Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5
   Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5
   Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5

Synthetic FENIKS closure generation and Diffsky HLTDS simple fits use the SSP
distributed with the HLTDS sample:

.. code-block:: text

   Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/diffsky_hltds_cosmos_260215_04_14_2026_ssp_data.hdf5

The synthetic and HLTDS amortized configs decode the same SSP through the
compressed basis:

.. code-block:: text

   Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/diffsky_hltds_cosmos_260215_04_14_2026_ssp_basis_k64_coeff16.hdf5

The compressed basis is a runtime representation of the dataset SSP, not a
different stellar-population model. It is upcast to ``float32`` on GPU in the
active configs. The HLTDS SSP file does not carry the same PopCosmos metadata
as the local FSPS assets, so Diffsky configs set
``model.asset_metadata_policy: permissive``. This is an explicit compatibility
choice for a recovery test; it is not evidence that the DSPS model exactly
reproduces the HLTDS generator.

Dust Parameterization
---------------------

The active Diffsky PopCosmos proxy uses ``model.dust_model:
prospector_fsps``. The fitted dust parameters are:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Meaning in the DSPS wrapper
   * - ``tau2``
     - Diffuse dust optical-depth amplitude applied to the stellar SED.
   * - ``dust_index_n``
     - Power-law slope/shape of the diffuse attenuation curve.
   * - ``tau1_over_tau2``
     - Birth-cloud optical-depth ratio; the code uses
       ``tau1 = tau1_over_tau2 * tau2`` for young populations.

The configs also fix ``dust_tesc_logyr: 7.0`` and ``dust1_index: -1.0``. These
dust parameters are coherent with the HLTDS SSP in the sense that they are
applied by DSPS to the SSP wavelength/age grid loaded from the configured SSP
asset. They are not metadata stored inside the SSP file, and they are not
validated by the SSP itself. Use ``diffsky-dust-ssp-audit`` to write the
current SSP wavelength/age summary and attenuation curves over the configured
``tau2``, ``dust_index_n``, and ``tau1_over_tau2`` bounds.

Likelihoods
-----------

FS2 uses flux-space Student-t likelihood because catalog flux errors are part
of the FS2 contract.

Diffsky HLTDS simple fits now also use flux-space Student-t likelihood. Native
photometric error columns were not confirmed in the downloaded HLTDS sample,
so the current source and continuous low-z subset materialize synthetic
``m5_depth`` errors. For each band, ``f5=fnu(m5)`` and
``sigma_rand^2=(0.04-gamma)*abs(flux)*f5 + gamma*f5^2``. The materialized
``fluxerr_*`` values then add the PhotErr-style ``sigma_sys_mag=0.005``
photometric floor in quadrature. LSST bands use the Rubin/LSST ``m5,gamma``
form; Roman bands currently use WFI one-hour point-source 5-sigma depths with
``eta=0.95`` so the depth/background term dominates but a source/Poisson-like
term remains. The active likelihood then adds a separate 2% fractional model
floor in quadrature. Current MAP/MCMC paths use the
observed flux as the floor reference, while amortized likelihood and
posterior-predictive residual diagnostics use the model flux as the floor
reference; that choice should be unified before comparing MAP and amortized
likelihood values quantitatively.

What To Check First
-------------------

For Diffsky HLTDS, start with the fixed-redshift closure config. If it cannot
recover reasonable mass and recent-SFR trends, the issue is the simplified
model/photometry contract, not the photo-z optimizer. Only after the closure
run is acceptable should the free-redshift config be used for blind recovery
tests.
