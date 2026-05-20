Science Assessment
==================

Active Model
------------

The active science workflow is intentionally small:

* fit ``z_obs`` from photometry with a broad flat prior;
* fit ``log10_formed_mass_msun`` as the SED amplitude;
* fit lognormal SFH shape parameters ``sfh_t_peak`` and ``sfh_tau``;
* fit scalar stellar metallicity ``log10_metallicity``;
* derive current SFR from fitted mass plus fitted SFH shape;
* inject COSMOS proxy dust columns when ``dust_model: cosmos_proxy_fixed`` is
  selected.

``log10_sfr`` and ``dust_av`` are not free parameters in the main science
config. ``log10_sfr`` is only a fixed internal SFH scale before mass
normalization. COSMOS dust is not mixed with a free DSPS dust fit.

Redshift
--------

The science preset does not feed ``phz_median`` into DSPS. It initializes
``z_obs`` from a deterministic random draw inside broad bounds, then fits
``z_obs`` directly from Euclid + LSST photometry.

PHZ interval priors were removed. Redshift validation must compare fitted
``z_obs`` against ``z_true_gal``.

Truth And Proxies
-----------------

Catalog truth columns are diagnostics, not likelihood terms.

``log_stellar_mass`` is converted from catalog ``h`` units for mass comparison.
``log_sfr_true`` is compared to derived current SFR. ``metallicity_true`` is
treated as a proxy only: gas-phase oxygen abundance is not the same observable
as DSPS stellar metallicity.

SED Diagnostics
---------------

COSMOS-template SED reconstruction is pseudo-ground truth. It helps inspect
whether fitted DSPS SEDs have plausible broad-band shapes. It is not physical
SPS truth.

Batch MAP runs save SED diagnostics for worst-fit galaxies by default, so the
plots show useful failures rather than arbitrary first rows.

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
