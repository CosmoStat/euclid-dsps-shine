Science Assessment
==================

Scope
-----

This page documents scientific assumptions, current weaknesses, and required
model changes for the Euclid Flagship / CosmoHub DSPS prototype.

The catalog does not contain wavelength-by-wavelength truth spectra. The
COSMOS-template reconstruction is therefore a pseudo-ground-truth diagnostic,
not physical SPS truth.

Current Forward Model
---------------------

``euclid_dsps.model`` implements a compact differentiable DSPS model:

* formed-mass amplitude ``log10_formed_mass_msun`` plus a lognormal SFH shape
  with ``sfh_t_peak`` and ``sfh_tau``;
* optional smooth burst term with ``sfh_burst_fraction``, ``sfh_burst_time``,
  and ``sfh_burst_width``;
* optional smooth quench term with ``sfh_quench_time``, ``sfh_quench_width``,
  and ``sfh_quench_depth``;
* scalar stellar metallicity ``log10_metallicity`` and scatter
  ``metallicity_scatter``;
* either Salim-style scalar dust or row-resolved two-component COSMOS dust;
* observed-frame broad-band AB photometry through configured filters.

The 10-band COSMOS comparison config uses:

* LSST ``ugrizy`` plus Euclid VIS/Y/J/H in the photometric likelihood;
* Euclid catalog error columns where available;
* SciPIC value-added CSV filters for LSST and Euclid;
* COSMOS two-component attenuation from ``ebv_cosmos_*``,
  ``ext_curve_cosmos_*``, and ``frac_cosmos_*``;
* ``phz_median`` as the redshift estimate and ``phz_min_70``/``phz_max_70``
  as row-level redshift-prior width;
* continuum-only branch-2 targets by default.

Recent Corrections
------------------

The current implementation now makes these science choices explicit:

1. Catalog flux errors are used when configured.

   Euclid survey-like bands can declare ``error_column``. The pipeline converts
   the flux-density error to an AB-magnitude uncertainty and uses it in the
   likelihood and chi-square. Bands without catalog errors still use the
   configured ``sigma_mag`` fallback.

2. Continuum-only DSPS is not scored against emission-line targets by default.

   The default branch-2 target set is ``continuum_internal_dust``. Emission-line
   target sets remain available for diagnostics, but they should not be used as
   the main score until a nebular emission-line model is added.

3. COSMOS dust can be injected into DSPS.

   The 10-band COSMOS config maps ``ebv_cosmos_*``, ``ext_curve_cosmos_*``, and
   ``frac_cosmos_*`` into the DSPS model, so the comparison can use the same
   two-component attenuation family as the COSMOS-template proxy SED.

4. Stellar mass is now a fit amplitude.

   When ``log10_formed_mass_msun`` is present, DSPS normalizes the SFH to that
   formed mass. Catalog ``log_stellar_mass`` is converted from
   ``log10(Msun h^-2)`` to ``log10(Msun)`` only for truth/proxy diagnostics.

5. The photo-z prior uses row-level PHZ intervals.

   ``phz_median`` sets the base redshift and the 70 percent NNPZ interval sets
   ``z_obs_prior_sigma``. This is still a Gaussian approximation, but it is no
   longer a fixed-width redshift prior for every galaxy.

6. The value-added data directory is used as the primary local template source.

   ``galaxy_seds`` and ``galaxy_extincts`` are the SciPIC/COSMOS resources
   documented with the catalog. They replace the external LePhare cache when
   available, while keeping the same pseudo-truth limitation.

Current Scientific Problems
---------------------------

1. SFH still too low-dimensional.

   The default 10-band fit now infers a formed mass plus a simple lognormal SFH
   shape. Burst/quench parameters exist in the code, but they are fixed by
   default to avoid overinterpreting broad-band photometry. This is safer than
   forcing a complex SFH from weak data, but it is still not equivalent to
   PROVABGS NMF SFHs or Prospector non-parametric SFHs.

   Current consequence: the DSPS SED may fit broad-band colors while recovering
   an implausible SFH. Treat SFH parameters as nuisance parameters until the
   model has a mass amplitude and a better SFH prior.

2. Metallicity truth is not stellar metallicity truth.

   ``metallicity_true`` is gas-phase ``12 + log(O/H)``. DSPS uses stellar
   metallicity for SSP weighting. The ``metallicity_true - 10.61`` conversion
   is a plotting proxy only.

3. Dust truth is parameterization-dependent.

   ``dust_ebv_true`` is built from COSMOS template attenuation components.
   It is not a calibrated truth value for DSPS ``dust_av``. The COSMOS dust
   mode reduces mismatch for template diagnostics, but it still inherits the
   template-model assumptions.

4. No emission-line model.

   DSPS currently models continuum+dust only. Branch-2 must not score this
   model against ``*_el_model3_ext*`` targets unless an emission-line module is
   added.

   Current consequence: residuals in bands affected by strong lines can be
   astrophysical model mismatch, not photometric calibration error.

5. Population model incomplete.

   Current population MAP regularizes fitted parameters inside each chunk. It
   is not a learned galaxy population model like pop-cosmos.

6. Photo-z prior is still approximate.

   The current redshift prior compresses the PHZ PDF interval into a Gaussian
   sigma. A full treatment should use the PDF samples or a calibrated mixture
   model when those are available.

7. COSMOS proxy SED is template truth, not physical truth.

   The proxy uses LePhare COSMOS templates, catalog attenuation parameters, and
   Euclid absolute-flux normalization. This is the right diagnostic for
   checking template-level consistency, but it cannot prove that DSPS recovered
   the true stellar population.

Why Burst/Quench SFH Exists
---------------------------

The burst/quench extension is not an invented shape for plot fitting. It is a
small differentiable approximation to two established lessons from the SED
literature:

* `PROVABGS <https://arxiv.org/abs/2202.01809>`__ models DESI BGS galaxies
  with a richer SPS parameterization including NMF SFH coefficients and burst
  terms. Its mock challenge reports that SED-model priors significantly affect
  posteriors and that spectroscopy+photometry constrains galaxy properties
  better than photometry alone.
* `How to Measure Galaxy SFHs II
  <https://arxiv.org/abs/1811.03637>`__ shows that non-parametric SFHs are
  flexible enough to recover broader SFH shapes, but posterior SFR(t) is highly
  prior-dependent even with UV--IR photometry.
* `pop-cosmos <https://arxiv.org/abs/2402.00935>`__ uses a population model
  calibrated to 26-band COSMOS photometry and a state-of-the-art SPS forward
  model, motivating population-level priors and diagnostics rather than only
  independent galaxy fits.

The implemented SFH remains deliberately conservative:

.. math::

   \mathrm{SFR}(t) =
   \left[\mathrm{SFR}_{lognormal}(t) +
   A_{burst}\exp\left(-\frac{(t-t_{burst})^2}{2\sigma_{burst}^2}\right)\right]
   \left[1 - d_q\,\sigma\left(\frac{t-t_q}{w_q}\right)\right].

This keeps the model JAX-vectorized and differentiable while allowing:

* excess recent/intermediate star formation through the burst term;
* smooth suppression after a quench time;
* explicit priors on burst amplitude and quench depth.

The model is still a stepping stone. Best next model: keep formed mass as the
amplitude and replace the lognormal shape with a small non-parametric or NMF
SFH basis with a documented prior.

Priors
------

Implemented fit priors are penalties in the JAX objective. Reported ``chi2``
remains the photometric chi-square.

Current 10-band priors:

* ``z_obs``: truncated-normal centered on row ``phz_median`` with sigma from
  ``phz_min_70``/``phz_max_70``.
* ``log10_formed_mass_msun``: broad normal amplitude prior.
* ``sfh_t_peak`` and ``sfh_tau``: broad smooth-SFH shape priors.
* ``log10_metallicity``: broad normal prior inside configured bounds.

Limitations:

* no mass-metallicity prior, although stellar mass is now available;
* no SFR-mass main-sequence prior;
* no redshift-dependent population prior;
* no dust-mass-SFR relation;
* no explicit selection function.

Required Science Changes
------------------------

1. Replace the lognormal SFH with a basis SFH.

   Candidate paths:

   * PROVABGS-like NMF SFH coefficients plus burst term;
   * Prospector-style non-parametric bins with continuity/stochastic prior;
   * low-rank SFH basis learned from simulations or COSMOS templates.

2. Add calibrated photo-z likelihood.

   Use full PHZ PDF information if available. The current interval-to-Gaussian
   approximation is a useful improvement but not a full photo-z likelihood.

3. Add a nebular emission-line module.

   Emission-line photometry should be tied to SFR, metallicity, ionization
   state, and attenuation. Until this exists, keep emission-line catalog
   columns outside the main likelihood score.

4. Add learned population-level priors.

   Current population MAP is chunk regularization. A pop-cosmos-like model
   would learn a redshift/mass/SFR/dust/metallicity population prior and use it
   during inference rather than only reporting grouped residuals afterward.

5. Expand population-level validation.

   Report metrics versus ``color_kind``, ``z_true_gal``, apparent magnitude, SFR
   proxy, metallicity proxy, template ID pair, and dust curve pair. Single-row
   visual inspection is insufficient. The current workflow now exports these
   grouped diagnostics; next step is to use them to drive model changes.

6. Add posterior checks.

   MAP is useful for speed. Scientific inference needs posterior calibration:
   HMC/NUTS for small subsets, simulation-based calibration or coverage tests
   for larger experiments.

7. Add image-level chromatic PSF prototype.

   Standard GalSim prototype first: profile times SED, bandpass integration,
   chromatic PSF convolution. JAX-GalSim second, after standard GalSim
   validation.

Comparisons And Validity
------------------------

Branch 1: rest-frame SED shape.

* compares COSMOS proxy SED and DSPS attenuated rest SED on common wavelength
  grid;
* least-squares scales DSPS to COSMOS over configured wavelength range;
* reports RMS log residual, median absolute log residual, slope residuals,
  D4000-like residual, and Euclid rest-color residuals.

Valid use: spectral-shape diagnostic.

Invalid use: proof that DSPS recovered true physical SPS parameters.

Branch 2: observed photometry.

* compares DSPS observed-frame model flux to configured catalog target columns;
* default target set is ``continuum_internal_dust``;
* noisy target set uses catalog ``*_error`` columns for chi-square when
  explicitly enabled.

Valid use: survey-like photometry residual diagnostic.

Invalid use: continuum DSPS scored against emission-line targets.

COSMOS proxy SED.

* reconstructed from COSMOS template IDs, extinction curves, E(B-V), and
  component fractions;
* normalized to Euclid ``*_abs`` fluxes;
* useful as template-level pseudo-ground-truth.

Invalid use: exact wavelength-by-wavelength physical truth.

References
----------

* Hahn et al., ``The DESI PRObabilistic Value-Added Bright Galaxy Survey
  (PROVABGS) Mock Challenge``, arXiv:2202.01809.
* Leja et al., ``How to Measure Galaxy Star Formation Histories II:
  Nonparametric Models``, arXiv:1811.03637.
* Alsing et al., ``pop-cosmos: A comprehensive picture of the galaxy population
  from COSMOS data``, arXiv:2402.00935.
