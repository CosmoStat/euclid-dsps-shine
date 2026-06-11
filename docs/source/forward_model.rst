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
   * - ``configs/fs2_gpu.yaml``
     - Euclid FS2
     - Full PopCosmos-like DSPS baseline with compressed stellar, gas, and AGN assets.
   * - ``configs/diffsky_hltds_04_14_simple_gpu.yaml``
     - Diffsky HLTDS 04/14/2026
     - Simplified DSPS recovery model with no AGN and fixed nebular treatment.
   * - ``configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml``
     - Diffsky HLTDS 04/14/2026
     - Same simplified model with redshift fixed to ``redshift_true`` for closure checks.

No public Diffstar config is kept. Diffstar/Diffmah columns can be inventoried
from the dataset, but they are not part of the first-pass DSPS recovery model.

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
* only one free recent-SFR offset, with older SFH offsets fixed to zero;
* magnitude-space likelihood using a documented model tolerance.

The free-redshift config fits:

.. code-block:: text

   z_obs
   log10_stellar_mass
   dlog10_sfr_1
   tau2
   dust_index_n

The fixed-redshift closure config fits:

.. code-block:: text

   log10_stellar_mass
   dlog10_sfr_1
   tau2
   dust_index_n

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

Diffsky HLTDS simple fits use the SSP distributed with the HLTDS sample:

.. code-block:: text

   Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/diffsky_hltds_cosmos_260215_04_14_2026_ssp_data.hdf5

That file does not carry the same PopCosmos metadata as the local FSPS assets,
so Diffsky configs set ``model.asset_metadata_policy: permissive``. This is an
explicit compatibility choice for a simple recovery test; it is not evidence
that the DSPS model exactly reproduces the HLTDS generator.

Likelihoods
-----------

FS2 uses flux-space Student-t likelihood because catalog flux errors are part
of the FS2 contract.

Diffsky HLTDS simple fits use AB magnitudes with ``sigma_mag: 0.10`` because
native photometric error columns were not confirmed in the downloaded HLTDS
sample. The prepared Diffsky parquet is built with ``--no-synthetic-errors`` so
the pipeline does not invent survey errors.

What To Check First
-------------------

For Diffsky HLTDS, start with the fixed-redshift closure config. If it cannot
recover reasonable mass and recent-SFR trends, the issue is the simplified
model/photometry contract, not the photo-z optimizer. Only after the closure
run is acceptable should the free-redshift config be used for blind recovery
tests.
