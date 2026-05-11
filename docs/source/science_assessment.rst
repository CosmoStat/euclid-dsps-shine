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

* lognormal SFH baseline with ``log10_sfr``, ``sfh_t_peak``, and ``sfh_tau``;
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
* COSMOS two-component attenuation from ``ebv_cosmos_*``,
  ``ext_curve_cosmos_*``, and ``frac_cosmos_*``;
* continuum-only branch-2 targets by default.

Current Scientific Problems
---------------------------

1. No explicit stellar mass parameter.

   ``log10_sfr`` still acts as luminosity/SFH amplitude. This mixes amplitude,
   recent SFR, and stellar mass. A physical SPS fit needs stellar mass or
   formed mass as a fitted parameter, with SFR derived from the SFH.

2. SFH still too low-dimensional.

   The burst/quench extension is an improvement over a pure lognormal, but it
   is not equivalent to PROVABGS NMF SFHs or Prospector non-parametric SFHs.
   Red/quiescent and post-starburst galaxies remain likely failure modes.

3. Metallicity truth is not stellar metallicity truth.

   ``metallicity_true`` is gas-phase ``12 + log(O/H)``. DSPS uses stellar
   metallicity for SSP weighting. The ``metallicity_true - 10.61`` conversion
   is a plotting proxy only.

4. Dust truth is parameterization-dependent.

   ``dust_ebv_true`` is built from COSMOS template attenuation components.
   It is not a calibrated truth value for DSPS ``dust_av``. The COSMOS dust
   mode reduces mismatch for template diagnostics, but it still inherits the
   template-model assumptions.

5. No emission-line model.

   DSPS currently models continuum+dust only. Branch-2 must not score this
   model against ``*_el_model3_ext*`` targets unless an emission-line module is
   added.

6. LSST filters may still be approximate.

   Euclid passbands are loaded from local ASCII files. LSST bands need exact
   throughput files for precision runs; top-hat/auto filters are acceptable
   only for development diagnostics.

7. Population model incomplete.

   Current population MAP regularizes fitted parameters inside each chunk. It
   is not a learned galaxy population model like pop-cosmos.

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

The model is still a stepping stone. Best next model: explicit stellar mass
plus a small non-parametric or NMF SFH basis with a documented prior.

Priors
------

Implemented fit priors are penalties in the JAX objective. Reported ``chi2``
remains the photometric chi-square.

Current 10-band priors:

* ``z_obs``: truncated-normal centered on row ``z_phz``.
* ``log10_sfr``: broad normal amplitude prior.
* ``sfh_t_peak`` and ``sfh_tau``: broad smooth-SFH priors.
* ``sfh_burst_fraction``: scaled-beta prior favoring no burst but allowing
  burst solutions.
* ``sfh_burst_time``: broad time prior.
* ``sfh_quench_time`` and ``sfh_quench_depth``: broad quench priors, with
  shallow/no quench preferred unless photometry supports stronger quenching.
* ``log10_metallicity``: broad normal prior inside configured bounds.

Limitations:

* no mass-metallicity prior;
* no SFR-mass main-sequence prior;
* no redshift-dependent population prior;
* no dust-mass-SFR relation;
* no explicit selection function.

Required Science Changes
------------------------

1. Add stellar mass or formed mass.

   Fit amplitude as ``log10_stellar_mass`` or ``log10_formed_mass``. Derive
   ``sfr_at_obs`` from the SFH. Keep ``log_sfr_true`` only as external
   diagnostic.

2. Replace burst/quench patch with a basis SFH.

   Candidate paths:

   * PROVABGS-like NMF SFH coefficients plus burst term;
   * Prospector-style non-parametric bins with continuity/stochastic prior;
   * low-rank SFH basis learned from simulations or COSMOS templates.

3. Add calibrated photo-z likelihood.

   Use PHZ uncertainty/PDF columns if available. Current fixed-width
   ``z_phz`` prior is only a placeholder.

4. Add exact LSST passbands.

   Replace any approximate LSST filters before interpreting 10-band residuals
   scientifically.

5. Add emission lines or remove emission-line target sets from science runs.

   Emission-line comparison requires nebular line modelling tied to SFR,
   metallicity, ionization, and dust. Until then, main branch-2 score remains
   continuum-only.

6. Add population-level validation.

   Report metrics versus ``color_kind``, ``z_true``, apparent magnitude, SFR
   proxy, metallicity proxy, template ID pair, and dust curve pair. Single-row
   visual inspection is insufficient.

7. Add posterior checks.

   MAP is useful for speed. Scientific inference needs posterior calibration:
   HMC/NUTS for small subsets, simulation-based calibration or coverage tests
   for larger experiments.

8. Add image-level chromatic PSF prototype.

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
