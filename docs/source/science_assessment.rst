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
* ``phz_median`` as the redshift initializer. ``phz_min/max_70/90/95`` can be
  carried as diagnostic catalog columns, but they are not used as fit priors;
* continuum-only branch-2 targets by default.

Large-Run Fit Modes
-------------------

The million-row path is not full per-galaxy gradient optimization. The default
10-band production config uses ``fit.fast_grid_search``:

* scan a small row-level redshift grid around the initialized value, bounded by
  configured redshift limits rather than PHZ hard intervals by default;
* analytically warm-start ``log10_formed_mass_msun`` from the broadband
  magnitude offset;
* scan small prior-bounded grids for ``log10_metallicity``, ``sfh_t_peak``, and
  ``sfh_tau``;
* compute SFR from the fitted mass-normalized SFH.

This is the intended large-catalog mode because it avoids repeating expensive
DSPS reverse-mode gradients for every galaxy. It is also closer to what the
current local data can constrain: redshift and luminosity amplitude dominate
the likelihood, while SFH shape and stellar metallicity remain weakly
identified by ten broad bands. Fast mode therefore gives these parameters
coarse, prior-regularized fits rather than pretending they are high-resolution
posterior constraints.

``--full-adam`` remains available for validation subsets. It optimizes all
configured free parameters, but should be used to calibrate and audit the fast
mode rather than as the default million-row path.

``fit-population`` follows the same rule. In fast mode it writes empirical
chunk-level population summaries and post-fit linear relation diagnostics. With
``--full-adam`` it runs the true joint population MAP objective and optimizes
population hyperparameters.

Feature Inventory
-----------------

Exists now:

* differentiable DSPS forward model on JAX/GPU;
* Euclid-only and 10-band LSST+Euclid configs;
* SED diagnostic plots/tables with DSPS SED, filters, photometry, and optional
  COSMOS proxy overlay;
* fast large-run fit for redshift, formed mass, stellar metallicity, and
  SFR-through-SFH-shape;
* full Adam MAP for smaller validation subsets;
* HMC/NUTS posterior sampling for small subsets;
* chunk-level performance benchmarks with CPU RSS and GPU memory;
* per-chunk checkpoints for long ``fit-batch`` runs;
* empirical fast-population summaries and full-Adam population MAP.

Does not exist yet:

* full per-galaxy posterior for millions of rows;
* learned population prior trained across the full catalog;
* calibrated main-sequence prior tying mass, SFR, redshift, and dust;
* nebular emission-line model with local compatible SSP assets;
* calibrated non-circular redshift prior or full photo-z PDF likelihood;
* explicit selection-function likelihood;
* image-level chromatic PSF/SED integration.

Should be added next:

* a dedicated GPU ``bench-forward`` command that separates compile time,
  forward time, gradient time, and memory peak;
* fixed-shape padded batches so every chunk reuses the same compiled graph;
* calibrated fast-mode priors for mass-SFR-metallicity-redshift from either
  trusted simulations or a held-out validation subset;
* posterior calibration on stratified subsets before any million-row posterior
  claim;
* a real learned population prior only after fast-mode diagnostics are stable.

Removed From The Active Model
-----------------------------

Several experimental hooks were removed from the active code path to keep the
local workflow interpretable:

* PHZ interval priors were removed. ``phz_median`` initializes redshift, then
  photometry and broad bounds drive the fit.
* binned SFH, burst, and quench parameters were removed. The current model uses
  one compact lognormal SFH shape plus formed-mass normalization.
* Salim scalar dust remains available for generic DSPS runs. The 10-band
  COSMOS diagnostic uses row-injected two-component COSMOS dust instead.
* Emission-line catalog columns are present, but the configured local SSP file
  lacks emission-line luminosity tables. Emission-line photometry must stay out
  of the primary likelihood until compatible assets or a calibrated line model
  exist.

Current Prior Justification
---------------------------

Priors are intentionally broad and stabilizing rather than final astrophysical
population priors. The justification is tied to what the local data can
support:

* ``z_obs`` is initialized from ``phz_median`` but uses a uniform prior in the
  current production configs. PHZ intervals are not used as priors because
  treating central intervals as hard truth is circular for redshift validation.
  Euclid photo-z work emphasizes that cosmology and physical inference need
  calibrated photo-z PDFs/PDZs, not unconstrained
  broad-band redshifts; see `Euclid preparation X
  <https://arxiv.org/abs/2009.12112>`__ and nearest-neighbour photo-z
  methodology such as `Tanaka et al. 2018
  <https://academic.oup.com/pasj/article/doi/10.1093/pasj/psx077/4494086>`__.
* ``log10_formed_mass_msun`` is the luminosity amplitude because SPS maps
  formed stellar mass and SFH to SED normalization. DSPS is explicitly designed
  as a differentiable SPS kernel connecting physical parameters to SEDs; see
  `Hearin et al. 2023 <https://arxiv.org/abs/2112.06830>`__.
* ``sfh_t_peak`` and ``sfh_tau`` use broad priors because compact parametric
  SFHs are stable but restrictive. `Leja et al. 2019
  <https://arxiv.org/abs/1811.03637>`__ shows that flexible/non-parametric SFHs
  are prior-dependent, so broad-band-only fits should avoid overconfident SFH
  freedom. `PROVABGS <https://arxiv.org/abs/2202.01809>`__ motivates richer
  SFH bases and burst terms, but also reinforces that priors and data quality
  matter.
* ``log10_metallicity`` uses a broad stellar-metallicity prior because the
  local truth proxy is gas-phase oxygen abundance, not stellar metallicity.
  The mass-metallicity relation can be measured diagnostically, but the current
  fast mode does not claim a learned physical relation. A proper learned
  population prior is closer to the `pop-cosmos
  <https://arxiv.org/abs/2402.00935>`__ program, where a joint population model
  is calibrated with much richer COSMOS photometry.
* SFR is not an independent amplitude prior in the 10-band fast mode. It is
  derived from fitted mass plus fitted SFH shape. A future main-sequence prior
  should couple mass, SFR, redshift, dust, and selection, rather than adding an
  isolated Gaussian prior on ``log10_sfr``.

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

5. The production redshift prior is not PHZ-based.

   ``phz_median`` sets the base redshift. The 70/90/95 percent NNPZ intervals
   can still be loaded for diagnostics, but ``configs/fs2_phz1_10band.yaml``
   uses ``type: uniform`` for ``z_obs`` and no PHZ interval penalty exists in
   MAP, fast-grid, population, or sampling code.

6. Population MAP can learn a mass-metallicity relation.

   ``fit.population.relations.log10_metallicity`` configures
   ``log10_metallicity ~ log10_formed_mass_msun``. Population MAP optimizes
   intercept, slope, and scatter in JAX along with per-galaxy parameters.
   In fast production mode the relation is instead measured after the fast
   per-galaxy fit and written as ``kind=fast_relation``; it is not used as an
   optimized prior unless ``--full-adam`` is used.

7. The value-added data directory is used as the primary local template source.

   ``galaxy_seds`` and ``galaxy_extincts`` are the SciPIC/COSMOS resources
   documented with the catalog. They replace the external LePhare cache when
   available, while keeping the same pseudo-truth limitation.

Latest Run Diagnostics
----------------------

The latest inspected outputs are:

* ``outputs/runs/10band_max_gpu``: 1,000 population MAP galaxies, 10 bands;
* ``outputs/runs/cosmos_sed_population_dsps_10band_2000_gpu``: 2,000 COSMOS
  pseudo-SED rows, 10-band population DSPS comparison.

Measured results from these runs:

* The broad-band photometry residuals remain large. The 10-band population run
  has median absolute residual ``0.191 mag`` and median reduced chi-square
  ``21.37``. The 2,000-row COSMOS run has median absolute residual ``0.183 mag``
  and median reduced chi-square ``18.44``.
* Redshift recovery is useful but not enough to explain the residuals. The
  2,000-row run has median ``z_obs - z_true = 0.0207``, MAD ``0.0904``, and
  correlation ``0.867``. This is consistent with a PHZ-prior-driven redshift
  estimate, not a fully free spectro-photometric redshift.
* The inferred formed mass is systematically high relative to the catalog mass
  proxy. Median bias is ``+1.31 dex`` in the 1,000-row run and ``+1.40 dex`` in
  the 2,000-row COSMOS run. The correlation is still high, about ``0.8``, so the
  issue is mostly calibration/normalization rather than random failure.
* Several reported ``fit_*`` columns are not inferred distributions. In both
  latest runs, ``sfh_t_peak``, ``sfh_tau``, and ``log10_metallicity`` are exactly
  constant at their initial values. ``dust_av`` is also constant in the COSMOS
  two-component dust run because COSMOS dust columns are injected from the
  catalog instead. These columns should be labeled as fixed or
  prior-dominated, not interpreted as recovered physical distributions.
* The learned mass-metallicity relation did not learn a meaningful slope in the
  1,000-row run. The fitted slopes are ``0.003`` and ``0.002`` in the two
  chunks, while metallicity values remain fixed at ``-2.25``. This is a
  diagnostic failure of the current population relation setup for these data,
  not evidence that the physical relation is flat.
* The strongest band-level photometry bias is in the red/NIR. In the 2,000-row
  run, median residuals are ``-0.273 mag`` in Euclid H, ``-0.178 mag`` in Euclid
  J, and ``-0.096 mag`` in Euclid Y. Median flux ratios are above unity in the
  same bands. This points to continuum-shape / dust / template mismatch.
* UV/blue bands have broad tails. LSST ``u`` has RMS residual ``0.746 mag`` and
  relative-flux residual tails with extreme outliers. Treat UV-driven parameter
  constraints as fragile until outlier handling and target definitions are
  audited.
* Rest-frame COSMOS pseudo-SED matching is imperfect even before observed-frame
  photometry. Branch 1 median RMS log SED residual is ``0.262 dex`` and
  ``95%`` absolute RMS reaches ``0.717 dex``. Early-type/color_kind ``0`` rows
  are worse, with median RMS ``0.327 dex`` versus ``0.207--0.254 dex`` for other
  color kinds.
* Branch 1 residuals depend on metallicity proxy and template/dust family. The
  lowest and highest gas-metallicity proxy bins are worse than the middle bins;
  dust-curve pairs involving code ``0`` are especially poor. These are
  template-proxy diagnostics, not direct stellar-population truth tests.

Current interpretation:

* The weird inferred distributions are mostly caused by under-constrained or
  fixed parameters, not by genuine astrophysical population structure.
* With 10 broad bands, injected COSMOS dust, and one mass
  amplitude, the optimizer mainly uses ``z_obs`` and ``log10_formed_mass_msun``.
  SFH shape and stellar metallicity remain effectively inactive in these runs.
  The fast production mode therefore explicitly treats SFH shape and stellar
  metallicity as fixed/prior-dominated unless they are row-injected.
* The mass offset and red/NIR residuals should be treated as the next primary
  science failure. Before adding more free parameters, first separate
  photometric chi-square, physical-prior penalty, population-prior penalty, and
  normalization diagnostics in the reports.
* The saved run-config JSON files are not sufficient to reproduce or audit the
  fit. They only contain workflow metadata, not the normalized model/fit config.

Current Scientific Problems
---------------------------

1. SFH still too low-dimensional.

   The default fit infers a formed mass plus a compact lognormal SFH shape.
   This is stable for the local broad-band catalog but it is still not
   equivalent to PROVABGS NMF SFHs or Prospector continuity-prior
   non-parametric SFHs. Binned SFH, burst, and quench branches were removed
   from the active model because the local broad-band fit has too few bands to
   constrain those degrees of freedom.

   Current consequence: the DSPS SED may fit broad-band colors while recovering
   an implausible SFH. Treat SFH parameters as nuisance parameters until the
   model has a mass amplitude and a better SFH prior.

   Latest-run consequence: the configured SFH shape parameters did not move from
   their initial values, so the current outputs must not be used as recovered
   SFH distributions.

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

   The local SSP file has only ``ssp_wave``, ``ssp_flux``, ``ssp_lgmet``, and
   ``ssp_lg_age_gyr``. It has no ``ssp_emline_*`` tables, so nebular emission
   cannot be computed from the local assets. Branch-2 must not score this model
   against ``*_el_model3_ext*`` targets.

   Current consequence: residuals in bands affected by strong lines can be
   astrophysical model mismatch, not photometric calibration error.

5. Population model incomplete.

   Current population MAP regularizes fitted parameters inside each chunk and
   can learn a mass-metallicity relation. It is still not a full learned galaxy
   population model like pop-cosmos.

   Latest-run consequence: the configured mass-metallicity relation collapsed to
   an almost-zero slope because the target metallicity values did not move. This
   relation is not yet scientifically usable on the current 10-band run.

6. Photo-z treatment is still approximate.

   The current redshift fit uses broad-band photometry plus simple bounds, not
   full PDF samples or calibrated multimodal mixtures. A full treatment should
   use those objects when available and should be evaluated against non-circular
   validation targets.

7. COSMOS proxy SED is template truth, not physical truth.

   The proxy uses LePhare COSMOS templates, catalog attenuation parameters, and
   Euclid absolute-flux normalization. This is the right diagnostic for
   checking template-level consistency, but it cannot prove that DSPS recovered
   the true stellar population.

Why Complex SFH Is Deferred
---------------------------

Complex SFH terms are scientifically motivated, but they are deferred until the
basic photometric workflow and mass/redshift recovery are stable:

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

Current decision: keep formed mass as the amplitude and use one lognormal SFH
shape. Reintroduce binned or low-rank SFHs only after SED diagnostics show the
simple model fails in a way that cannot be fixed by normalization, dust, or
filter handling.

Implemented Prior Summary
-------------------------

Implemented fit priors are penalties in the JAX objective. Reported ``chi2``
remains the photometric chi-square. Scientific motivation and citations are in
`Current Prior Justification`_ above.

Current 10-band science priors:

* ``z_obs``: uniform inside configured broad bounds. ``phz_median`` initializes
  the fit, while PHZ intervals stay diagnostics only.
* ``log10_formed_mass_msun``: broad normal amplitude prior.
* ``sfh_t_peak`` and ``sfh_tau``: broad smooth-SFH shape priors when free.
* ``log10_metallicity``: broad normal prior in independent MAP; population MAP
  can replace the independent prior with a mass-metallicity relation.

Limitations:

* no SFR-mass main-sequence prior;
* no redshift-dependent population prior;
* no dust-mass-SFR relation;
* no explicit selection function.

Required Science Changes
------------------------

1. Improve the SFH model only when data support it.

   Candidate paths:

   * PROVABGS-like NMF SFH coefficients plus burst term;
   * Prospector-style non-parametric bins with continuity/stochastic prior;
   * low-rank SFH basis learned from simulations or COSMOS templates.

2. Add a nebular emission-line module only with compatible assets.

   The local SSP asset has no line luminosity tables. Emission-line photometry
   should be tied to SFR, metallicity, ionization state, and attenuation only
   after line-enabled SSP data or a calibrated line model are available. Until
   this exists, keep emission-line catalog columns outside the main likelihood
   score.

3. Add broader learned population-level priors.

   Current population MAP is chunk regularization plus configured relations. A
   pop-cosmos-like model would learn a full joint redshift/mass/SFR/dust/metallicity
   population prior and use it during inference.

4. Add posterior checks.

   MAP is useful for speed. Scientific inference needs posterior calibration:
   HMC/NUTS for small subsets, simulation-based calibration or coverage tests
   for larger experiments.

5. Add image-level chromatic PSF prototype.

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
