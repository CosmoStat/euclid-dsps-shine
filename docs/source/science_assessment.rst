Science Assessment
==================

Implemented State
-----------------

The current branch has one active PopCosmos-like model configuration:
``configs/popcosmos_binned.yaml``.

Implemented pieces:

* seven-bin PopCosmos-like SFH parameterization;
* free ``z_obs`` from LSST+Euclid photometry, with no photo-z prior;
* single stellar metallicity interpolation;
* Charlot-Fall age-dependent dust;
* Madau95-style approximate IGM attenuation;
* generated FSPS gas SSP grid over ``gas_logz`` and ``gas_logu``;
* generated FSPS/CLUMPY AGN template grid over native Nenkova ``agn_tau``;
* flux-space likelihood using catalog per-band flux errors;
* MAP fitting for one-row and batched runs;
* HMC/NUTS posterior smoke support for selected rows.

Generated Assets
----------------

The gas asset is:

.. code-block:: text

   Data/popcosmos_gas_ssp_grid.h5
   ssp_flux shape: (7, 7, 12, 107, 11149)

The AGN asset is:

.. code-block:: text

   Data/popcosmos_agn_template_grid.h5
   template_lnu_per_lbol shape: (9, 11149)

Both assets are fixed during DSPS/JAX fitting. The JAX model interpolates and
weights them rather than calling FSPS at fit time.

Parameter Choices
-----------------

The active 16-parameter vector is:

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

``ln_tauagn`` is initialized at ``ln(10)`` and bounded to
``[ln(5), ln(150)]`` because those are the native FSPS/CLUMPY template limits.

Known Caveats
-------------

This is PopCosmos-like, not yet an audited reproduction of the full POP-COSMOS
population model.

Scientific caveats kept in the model and asset metadata:

* gas metallicity is varied independently of stellar metallicity; FSPS warns
  this is not fully self-consistent for all line ratios;
* the AGN template normalization follows the repository convention
  ``fagn * integrated stellar Lbol`` and is marked approximate until the exact
  FSPS/CLUMPY bolometric convention is independently audited;
* the IGM implementation is a stable Madau95-style approximation;
* broad-band photometry alone can trade AGN, dust, gas, and SFH parameters
  against each other, so posterior checks remain important.

Runtime Caveat
--------------

The full gas grid is about 2.6 GiB in ``float32``. The fit and post-fit batch
prediction code now pass the large SSP, gas, AGN, and filter arrays as dynamic
arguments to jitted model calls, instead of closing over them through
``DspsContext``. This keeps JIT available for single-row and batched Adam fits
without compiling the gas grid as an XLA constant. Gas-grid interpolation also
gathers only the four bracketing gas-metallicity/gas-ionization slabs before
interpolation, avoiding a temporary array over the full gas-ionization axis.

On GPU, the remaining memory requirement is real device memory for the grid and
working buffers. If CUDA-enabled JAX is installed, use a GPU runtime config/env;
otherwise keep the documented CPU-safe environment variables.

Diagnostics
-----------

Catalog truth/proxy columns are diagnostics only. The likelihood uses observed
photometry and flux errors. COSMOS-template SED reconstruction remains a
diagnostic comparison, not ground truth.
