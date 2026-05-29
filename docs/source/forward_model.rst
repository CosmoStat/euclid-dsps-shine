Forward Model And SED Pipeline
==============================

Purpose
-------

The goal of this repository is to evaluate a DSPS/JAX approximation to the
same broad-band forward model that FSPS/Prospector evaluates for a
PopCosmos-like parameterization. FSPS is used offline to generate spectral
assets. The fit-time code then interpolates those assets in JAX, so it can be
optimized and eventually used inside learned-prior workflows.

The default science configuration is the full AGN binned-SFH model:

.. code-block:: text

   configs/popcosmos_binned.yaml

``configs/popcosmos_binned_noagn.yaml`` is kept as a fallback/debug config.

Glossary
--------

``SED``
   Spectral energy distribution: luminosity or flux as a function of wavelength.
   The model first predicts a rest-frame SED, then turns it into observed
   photometry.

``SSP``
   Simple stellar population: a single-age, single-metallicity stellar
   population formed with a fixed IMF. SSPs are the spectral building blocks.
   A galaxy is modeled as a weighted sum of SSP ages and metallicities.

``SFH``
   Star-formation history: the amount of stellar mass formed as a function of
   time. In the binned PopCosmos-like config, the SFH is a set of time bins; in
   Diffstar, it is a smooth parametric history.

``IMF``
   Initial mass function: the distribution of stellar masses at birth. It
   affects luminosity per unit mass, surviving mass, and mass-to-light ratios.
   The active assets use Chabrier, FSPS ``imf_type=1``.

``Isochrones``
   Stellar-evolution tracks sampled at a fixed age. They define where stars of
   different masses sit in temperature, luminosity, and evolutionary phase. The
   active FSPS assets use MIST isochrones.

``Spectral library``
   The stellar spectra attached to the isochrone points. The active assets use
   C3K because that is the local FSPS/Prospector setup used for the closure
   benchmark.

``IGM``
   Intergalactic medium. It absorbs rest-frame UV light, especially around the
   Lyman series at high redshift. The active model uses an FSPS Madau95-like
   implementation.

``CLOUDY``
   Photoionization code used by FSPS to model nebular gas emission. It produces
   nebular continuum and emission lines.

``Emission lines``
   Narrow features such as Balmer and oxygen lines. They can change broad-band
   photometry when a strong line falls inside a filter. The current model uses
   raw FSPS/CLOUDY lines; PopCosmos learned line-by-line corrections are not
   included.

``AGN``
   Active galactic nucleus. In this repo, full AGN means the model includes an
   FSPS AGN component grid and fits ``ln_fagn`` and ``ln_tauagn``. No-AGN means
   ``agn_model: none`` and those parameters are removed.

``FSPS``, ``Prospector``, and ``DSPS``
   FSPS is Flexible Stellar Population Synthesis. Prospector is an inference
   framework built around FSPS. DSPS is the differentiable/JAX path used here to
   evaluate a similar forward model efficiently.

Full AGN And No-AGN Modes
-------------------------

Full AGN is the default path:

.. code-block:: text

   configs/popcosmos_binned.yaml
   configs/popcosmos_diffstar.yaml

It includes stars, dust, gas, IGM, and AGN. The AGN parameters are:

.. code-block:: text

   ln_fagn
   ln_tauagn

No-AGN is the controlled ablation path:

.. code-block:: text

   configs/popcosmos_binned_noagn.yaml
   configs/popcosmos_diffstar_noagn.yaml

It keeps the same stellar, dust, gas, redshift, filters, and likelihood setup,
but sets ``agn_model: none`` and removes the AGN free parameters.

Active Spectral Assets
----------------------

The active assets are all Chabrier IMF assets generated from the same local FSPS
setup:

.. list-table::
   :header-rows: 1

   * - File
     - Role
   * - ``Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5``
     - Base Chabrier SSP. Includes fixed nebular emission at ``gas_logu=-2`` and ``gas_logz=0``. Used as the main wavelength, age, metallicity, and surviving-mass reference.
   * - ``Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5``
     - Pure-stellar Chabrier SSP. ``add_neb_emission=0`` and ``add_neb_continuum=0``. Used by benchmark levels that must be gas-free.
   * - ``Data/popcosmos_chabrier_gas_ssp_grid.h5``
     - FSPS/CLOUDY gas-varying SSP grid over gas metallicity and gas ionization.
   * - ``Data/popcosmos_chabrier_agn_component_ssp_grid.h5``
     - FSPS-native AGN component grid over ``fagn``, ``agn_tau``, stellar metallicity, SSP age, and wavelength.

All PopCosmos-like assets must declare:

.. code-block:: text

   imf_type = 1
   imf_name = chabrier
   z_sun = 0.0142

Contradictory or missing PopCosmos-like IMF metadata fails validation. Legacy
Kroupa files are not valid for the active configs.

Why These Models
----------------

The model choices are driven by consistency with the local FSPS/Prospector
reference benchmark:

* Chabrier IMF is used because PopCosmos-like configs should not mix Kroupa and
  Chabrier mass-to-light conventions.
* MIST + C3K are used because they are the active local FSPS libraries and give
  the wavelength, age, and metallicity axes used by the benchmark.
* ``z_sun=0.0142`` fixes the relative-to-absolute metallicity conversion.
* FSPS/CLOUDY gas is used because it is the standard FSPS nebular path. The
  current implementation validates the raw broad-band behavior, not official
  PopCosmos line-corrected gas.
* The AGN component grid is generated directly from FSPS differences so that the
  DSPS AGN contribution follows the same effective FSPS/Prospector convention.

Step-By-Step SED Construction
-----------------------------

1. Load HDF5 spectral axes
~~~~~~~~~~~~~~~~~~~~~~~~~~

``euclid_dsps.model.load_context`` reads the base SSP:

.. code-block:: text

   ssp_wave[wave]
   ssp_lg_age_gyr[age]
   ssp_lgmet[stellar_metallicity]
   ssp_flux[stellar_metallicity, age, wave]
   ssp_surviving_mstar[stellar_metallicity, age]

The wavelength unit is Angstrom. ``ssp_lg_age_gyr`` is ``log10(age/Gyr)``.
``ssp_lgmet`` is absolute ``log10(Z)``. The PopCosmos-like sampled metallicity
parameter is relative solar:

.. code-block:: text

   log10(Zstar absolute) = log10(0.0142) + log10_stellar_metallicity

2. Build the SFH on the SSP age grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the binned config, ``model.sfh_time_grid: prospector_step`` maps the six
``dlog10_sfr_i`` parameters onto a Prospector-like step SFH. The implementation
integrates the overlap between the PopCosmos lookback-time bins and the SSP age
grid. This replaced the earlier generic linear time-grid approximation because
it was the main pure-stellar mismatch against FSPS/Prospector.

For Diffstar, the same spectral assets are used but the SFH weights come from
the reduced Diffstar path.

3. Normalize by formed mass and surviving mass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sampled ``log10_stellar_mass`` is interpreted consistently with the FSPS SSP
mass convention. The generated SSP files store ``ssp_surviving_mstar`` from
FSPS. Using this table avoids the few-percent offset that came from a generic
DSPS analytic surviving-mass approximation.

4. Interpolate stellar metallicity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model interpolates the stellar SSP grid in absolute ``log10(Z)``. The gas
parameters remain in relative units, ``log10(Zgas/Zsun)`` and ``log10 U``.

The PopCosmos-like hard constraint is:

.. code-block:: text

   log10_gas_metallicity >= log10_stellar_metallicity

Both sides are in ``log10(Z/Zsun)``. Invalid points receive zero probability in
fit/MCMC paths rather than being clipped.

5. Apply stellar dust
~~~~~~~~~~~~~~~~~~~~~

The active dust model is:

.. code-block:: yaml

   model:
     dust_model: prospector_fsps
     dust_tesc_logyr: 7.0
     dust1_index: -1.0

``tau2`` is diffuse dust and affects all stellar ages. ``tau1_over_tau2``
creates a birth-cloud optical depth ``dust1 = tau1_over_tau2 * tau2`` that only
affects populations younger than ``10 Myr``. The diffuse curve follows the local
FSPS/Prospector ``dust_type=4`` shape.

6. Add nebular emission
~~~~~~~~~~~~~~~~~~~~~~~

The production gas path uses:

.. code-block:: yaml

   model:
     nebular_model: gas_grid
     gas_grid_path: Data/popcosmos_chabrier_gas_ssp_grid.h5
     emission_line_corrections: none

The gas grid axes must match the base SSP wavelength, age, and stellar
metallicity axes. The grid is interpolated in gas metallicity and ionization.

``emission_line_corrections: none`` means raw FSPS/CLOUDY nebular emission. The
code has a ``popcosmos_table`` hook for enriched line/continuum grids, but the
official PopCosmos learned correction table is not present in this repository.

7. Add AGN
~~~~~~~~~~

The active full AGN model uses an FSPS-native component grid:

.. code-block:: yaml

   model:
     agn_model: fsps_component_grid
     agn_component_grid_path: Data/popcosmos_chabrier_agn_component_ssp_grid.h5
     agn_host_attenuation: fsps_diffuse_unit_tau
     agn_igm_order: fsps_after_igm
     agn_baked_attenuation: fsps_powerlaw_unit_tau
     agn_baked_dust_index: -0.7

The component grid stores:

.. code-block:: text

   FSPS(fagn, agn_tau) - FSPS(fagn=0)

for each SSP age and stellar metallicity. DSPS interpolates this grid in
``fagn = exp(ln_fagn)``, ``agn_tau = exp(ln_tauagn)``, stellar metallicity, and
age, then sums it with the same SFH weights as the stellar component.

The AGN component generated by FSPS already carries a baked unit-tau
``dust_type=0`` attenuation. The runtime model replaces that baked curve with
the FSPS/Prospector-like unit-tau diffuse curve used by ``dust_type=4``. This
was the key fix that closed the dusty AGN benchmark.

8. Apply IGM and ordering
~~~~~~~~~~~~~~~~~~~~~~~~~

The active IGM model is:

.. code-block:: yaml

   model:
     igm_model: fsps_madau95
     agn_igm_order: fsps_after_igm

``fsps_after_igm`` matches the local FSPS ordering used in the benchmark:
stellar and gas light are IGM-attenuated before the AGN component is added.

9. Integrate photometry
~~~~~~~~~~~~~~~~~~~~~~~

The model redshifts the rest-frame SED, applies luminosity distance, and
integrates through the ten configured filters:

.. code-block:: text

   LSST u g r i z y
   Euclid VIS Y J H

Fits use flux-space residuals. This matters because extremely faint
magnitude-space diagnostic rows can become non-finite even when flux-space
behavior is well defined.

Implementation Map
------------------

.. list-table::
   :header-rows: 1

   * - Concern
     - File
   * - Config normalization and validation
     - ``euclid_dsps/config.py``
   * - HDF5 asset loading and DSPS/JAX forward model
     - ``euclid_dsps/model.py``
   * - Flux-space likelihoods and optimizer integration
     - ``euclid_dsps/fit.py``
   * - Posterior sampling constraints
     - ``euclid_dsps/mcmc.py``
   * - SSP generation
     - ``scripts/generate_fsps_ssp_grid.py``
   * - Gas grid generation
     - ``scripts/generate_fsps_gas_grid.py``
   * - AGN component grid generation
     - ``scripts/generate_fsps_agn_component_grid.py``
   * - FSPS/Prospector benchmark
     - ``scripts/benchmark_against_fsps_prospector.py``

Benchmark Status
----------------

The latest full closure benchmark is:

.. code-block:: text

   outputs/benchmarks/popcosmos_binned_full_forward_fsps_closure_n500
   outputs/report/popcosmos_binned_full_forward_fsps_closure_n500/report.md

It sampled 500 points, 8 levels, and 10 bands. Production broad-band levels
pass the configured bright finite FSPS/Prospector-like target:

.. list-table::
   :header-rows: 1

   * - Level
     - Median band p95 abs delta mag
     - Max band p95 abs delta mag
   * - ``full_noagn``
     - ``0.0129``
     - ``0.0166``
   * - ``full_agn``
     - ``0.0123``
     - ``0.0371``

This validates the current DSPS broad-band forward model against the local
FSPS/Prospector reference. It does not validate official PopCosmos emission-line
corrections.

Scientific Assumptions
----------------------

* IMF: Chabrier, FSPS ``imf_type=1``.
* Isochrones: MIST, as reported by the local FSPS build.
* Spectral library: C3K, as reported by the local FSPS build.
* Solar metallicity convention: ``z_sun=0.0142``.
* SFH: PopCosmos-style step bins by default; Diffstar optional comparison.
* Dust: Prospector/FSPS-like ``dust_type=4`` approximation in JAX.
* IGM: FSPS Madau95-like implementation.
* Gas: raw FSPS/CLOUDY grid unless a future PopCosmos correction table is
  supplied.
* AGN: FSPS-native component grid with FSPS-like host attenuation/order.
