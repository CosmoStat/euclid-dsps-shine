Science Assessment
==================

Active Model
------------

The active science workflow is intentionally small:

* fit ``z_obs`` from photometry with no photo-z prior;
* fit ``log10_formed_mass_msun`` as the SED amplitude;
* fit lognormal SFH shape parameters ``sfh_t_peak`` and ``sfh_tau``;
* fit scalar stellar metallicity ``log10_metallicity``;
* derive current SFR from fitted mass plus fitted SFH shape;
* inject COSMOS proxy dust columns when ``dust_model: cosmos_proxy_fixed`` is
  selected.

``log10_sfr`` and ``dust_av`` are not free parameters in the main science
config. ``log10_sfr`` is only a fixed internal SFH scale before mass
normalization. With ``dust_model: cosmos_proxy_fixed``, COSMOS dust columns
drive attenuation and ``dust_av`` is marked inactive, not inferred.

Redshift
--------

The science preset does not initialize ``z_obs`` from ``phz_median`` and does
not use a photo-z prior. ``redshift.initial: fixed`` is only the MAP starting
value; fitted ``z_obs`` is free within configured bounds. Redshift multi-start
MAP was removed, so posterior runs should be used when redshift degeneracy is
scientifically important.

PHZ interval priors were removed. Redshift validation must compare fitted
``z_obs`` against ``z_true_gal``.

Photometric Errors
------------------

The active 10-band config uses continuum flux columns as fit targets and
per-band catalog uncertainties from ``*_el_model3_ext_odonnell_ext_error`` for
LSST and Euclid. Those uncertainties are converted from flux units into local
AB-magnitude errors for reporting, but the default MAP and MCMC likelihood is
Gaussian in ``Fnu`` cgs flux space. Set ``fit.likelihood_space: mag`` only for
legacy diagnostics.

Truth And Proxies
-----------------

Catalog truth columns are diagnostics, not likelihood terms.

``log_stellar_mass`` is converted from catalog ``h`` units for mass comparison.
The fitted DSPS amplitude is ``log10_formed_mass_msun``. This is formed mass,
not necessarily surviving stellar mass, so an offset relative to catalog stellar
mass can be semantic rather than a pure conversion bug.
``log_sfr_true`` is compared to derived current SFR. ``metallicity_true`` is
treated as a proxy only: gas-phase oxygen abundance is not the same observable
as DSPS stellar metallicity.

Fit-vs-catalog plots only include comparable active/free or derived
parameters. Fixed inactive quantities such as ``dust_av`` under COSMOS proxy
dust are audited but not plotted as inferred parameters.

SED Diagnostics
---------------

COSMOS-template SED reconstruction is pseudo-ground truth. It helps inspect
whether fitted DSPS SEDs have plausible broad-band shapes. It is not physical
SPS truth.

Photometric markers in SED plots are model-anchored ratios. They are visual
residual diagnostics, not an independent rest-frame spectrum. SED diagnostics
also include a flux-space residual panel showing
``(F_obs - F_model) / sigma_F`` per band.

Batch MAP runs split SED diagnostics between worst-fit and best-fit galaxies.
COSMOS proxy SED CSVs and manifests include the scalar normalization factor and
normalization-band residuals. Large COSMOS/DSPS height mismatches should be
debugged as normalization, filter, dust, redshift, or template-unit issues
before being interpreted as physical disagreement.

Non-Detections And Calibration
------------------------------

In flux-space mode, finite non-positive fluxes with valid errors are kept by
``selection.nondetection_policy: gaussian_flux``. ``upper_limit`` is reserved
but not implemented.

``band_calibration.mode: fixed_offsets`` can test fixed per-band zero-point
offsets. Offsets are applied to the model during likelihood evaluation, never
to both data and model.

Goodness Of Fit
---------------

``reduced_chi2`` is the DOF-corrected value ``chi2 / dof``. The number of valid
bands, effective free parameters, and DOF are written per object. The old
``chi2 / n_bands`` diagnostic is still available as ``chi2_per_band``.

Priors
------

Current named prior sets:

* ``weak_physical``: broad stabilizing priors used by
  ``configs/fs2_phz1_science.yaml``;
* ``flat_debug``: flat priors inside configured fit bounds;
* ``popcosmos_like``: reserved and intentionally unavailable until exact
  POP-COSMOS parameter mapping, units, and selection treatment are implemented.

Current priors are not POP-COSMOS priors yet. Do not label results as
POP-COSMOS-like until that mapping exists.

Out Of Likelihood
-----------------

Emission-line catalog columns remain diagnostics only. The active DSPS model
does not include a calibrated line model or compatible local line assets, so
line columns must stay out of the likelihood.
