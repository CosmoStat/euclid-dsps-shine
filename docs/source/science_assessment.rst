Science Assessment
==================

Purpose
-------

This page records the current scientific assumptions behind the Euclid FS2
DSPS prototype and the next modelling steps raised in supervisor discussion.
It is intentionally stricter than the code: the code can be useful for
experiments before every assumption below is fully solved, but reports should
label those experiments accordingly.

Current Model
-------------

The current forward model in ``euclid_dsps.model`` is a compact DSPS model:

* a lognormal star-formation history controlled by ``log10_sfr``,
  ``sfh_t_peak``, and ``sfh_tau``;
* one scalar stellar metallicity parameter, ``log10_metallicity``;
* one scalar metallicity scatter parameter, ``metallicity_scatter``;
* a Salim et al. 2018-style foreground attenuation curve controlled by
  ``dust_av`` and ``dust_slope``;
* observed-frame broad-band AB magnitudes computed through configured filters.

This is appropriate as a differentiable prototype. It is not yet equivalent to
the physical models used in PROVABGS, SEDflow, or pop-cosmos.

Observed Fluxes And Dust
------------------------

The downloaded project parquet contains the selected observed flux columns:
``euclid_vis``, ``euclid_nisp_y``, ``euclid_nisp_j``, and ``euclid_nisp_h``.
It does not contain ``*_abs`` rest-frame flux columns or explicit COSMOS SED
template identifiers.

The local parquet schema has 255896 rows and 48 selected columns. Converting
the Euclid fluxes with the configured ``fnu_cgs`` interpretation gives plausible
AB magnitudes:

.. list-table::
   :header-rows: 1

   * - Band
     - Median AB mag
     - 1--99 percent range
   * - ``euclid_vis``
     - 25.05
     - 20.19--26.93
   * - ``euclid_nisp_y``
     - 24.30
     - 19.40--25.97
   * - ``euclid_nisp_j``
     - 23.96
     - 19.03--25.66
   * - ``euclid_nisp_h``
     - 23.68
     - 18.68--25.61

The important interpretation is:

* The selected ``euclid_*`` columns should be treated as observed, internally
  attenuated broad-band flux densities.
* A DSPS model should therefore compare dust-attenuated model photometry against
  these columns. In that sense, applying attenuation in DSPS is not a
  double-count by itself.
* It becomes scientifically unsafe if the catalog dust truth is forced as an
  exact DSPS ``dust_av`` target. The Flagship attenuation construction is based
  on COSMOS templates, template-specific attenuation curves, and E(B-V)-like
  values. The current DSPS dust parameter is a different Salim-style
  foreground-screen parameterization.
* The current ``dust_ebv_true * 4.05`` mapping is therefore only a diagnostic
  proxy. It should not be interpreted as a calibrated one-to-one truth for
  ``dust_av``.

The same caution applies to ``metallicity_true``. The catalog field is a
gas-phase oxygen abundance, ``12 + log10(O/H)``. The current
``metallicity_true - 10.61`` conversion is a useful proxy for plotting against
DSPS stellar metallicity, but it is not a physical identity between gas-phase
metallicity and the stellar metallicity distribution used by SPS.

Parameter Mapping Risk
----------------------

The supervisor note that "not the same parametrization, so no certainty that
the parameters map" is central. Current comparisons are meaningful as
correlation diagnostics, not as direct parameter recovery, for three reasons:

* ``log_sfr_true`` is an instantaneous or catalog-level SFR label, while the
  DSPS model has a full SFH and reports ``sfr_at_obs`` from the final time bin.
* ``metallicity_true`` is gas-phase oxygen abundance, while DSPS uses stellar
  metallicity for SSP weighting.
* ``dust_ebv_true`` is constructed from COSMOS attenuation components, while
  DSPS uses a different attenuation law and scalar ``A_V``.

Reports should therefore use language such as "catalog proxy", "diagnostic
comparison", or "rank-correlation check" until the project has a validated
mapping.

Physically Motivated Priors
---------------------------

The current default priors are broad truncated normals. They are adequate for
software checks but weak scientifically. A better sequence is:

1. Redshift prior:
   center on ``z_phz`` with a width from a PHZ uncertainty column or PDF summary
   if available. If only ``phz_mode_1`` is present, treat the prior width as a
   sensitivity parameter and compare against ``z_true`` only for validation.

2. Stellar mass prior:
   add a stellar-mass truth/proxy column from CosmoHub if available. Without a
   mass parameter the SPS amplitude is poorly identified; ``log10_sfr`` is
   currently acting as both recent-SFR control and luminosity normalization.

3. SFH prior:
   replace the single lognormal SFH with either a small non-parametric SFH basis
   or a PROVABGS-like NMF basis. Literature on non-parametric SFHs shows that
   the posterior SFH can be prior-dominated, especially for photometry-only
   fits, so this prior must be an explicit scientific choice.

4. Dust prior:
   condition ``dust_av`` or ``E(B-V)`` on color, SFR, mass, and redshift only
   after the parameterization is aligned. Until then, use broad dust priors and
   check how much redshift and SFR posteriors move.

5. Metallicity prior:
   if fitting stellar metallicity, use a broad prior linked to stellar mass and
   redshift. Use gas-phase ``metallicity_true`` only as an external comparison,
   not as the direct target.

Hahn/PROVABGS Direction
-----------------------

The most relevant Hahn line of work is PROVABGS and SEDflow:

* PROVABGS performs Bayesian SED modelling of DESI spectroscopy and Legacy
  Surveys photometry to infer full posteriors for stellar mass, SFR,
  mass-weighted stellar metallicity, and stellar age.
* The PROVABGS model is substantially richer than this prototype: it uses a
  non-parametric SFH, time-varying metallicity history, and a flexible dust
  prescription.
* SEDflow accelerates the PROVABGS-style posterior inference from optical
  photometry using amortized neural posterior estimation.

The direct lesson for this project is not to copy the exact DESI model
immediately. The lesson is to make the generative model and prior explicit and
to validate posterior recovery under the same observable conditions. With only
Euclid VIS+NISP bands, the model is much less constrained than PROVABGS'
photometry+spectroscopy case, so priors will matter more.

pop-cosmos Direction
--------------------

pop-cosmos is directly relevant because it treats redshift and SPS parameters
with an empirical population prior trained on COSMOS2020 photometry.

The practical lessons are:

* joint redshift and physical-parameter inference is possible with photometry,
  but only with a strong population model and careful photometric error model;
* the prior is a scientific object, not a technical nuisance;
* population-level validation is required, not only one-galaxy fits;
* 26-band COSMOS2020 is far more informative than VIS+YJH alone, so this
  project should use LSST bands from the SQL query whenever possible.

Ground-Truth SED Strategy
-------------------------

The current SQL query does not select enough information to reconstruct the
exact Flagship SED template per galaxy. To test whether DSPS can synthesize the
catalog spectrum, the next CosmoHub query should look for and include, if
available:

* COSMOS template or SED identifiers for each component;
* extinction curve identifiers and E(B-V) values for each component;
* intrinsic and attenuated absolute magnitudes or rest-frame fluxes;
* stellar mass and stellar age proxies;
* photometric uncertainties or PHZ PDF summary columns.

If template identifiers are available, a useful validation mode is:

1. reconstruct the catalog template-level SED using the catalog's own template
   and attenuation prescription;
2. integrate it through the same Euclid and LSST bandpasses;
3. compare that photometry to the exported ``euclid_*`` and ``lsst_*`` columns;
4. only then compare DSPS-generated SEDs to the reconstructed template SED.

This separates "can we reproduce the catalog photometry?" from "does the DSPS
physical model recover the catalog's latent labels?" These are different tests.

Chromatic PSF And GalSim
------------------------

For image simulation, the correct conceptual object is a spatial profile with
an SED, drawn through a bandpass and convolved with a possibly chromatic PSF.
In standard GalSim, the separable case is:

.. code-block:: python

   obj = galsim.Sersic(n=2.0, half_light_radius=0.3)
   sed = galsim.SED(wave_flux_table, wave_type="Ang", flux_type="fnu")
   bandpass = galsim.Bandpass(wave_transmission_table, wave_type="Ang")
   chromatic_galaxy = obj * sed
   image = chromatic_galaxy.drawImage(bandpass)

For a bulge+disk model with color gradients, give each component its own SED:

.. code-block:: python

   bulge = galsim.DeVaucouleurs(half_light_radius=bulge_r50) * bulge_sed
   disk = galsim.Exponential(half_light_radius=disk_r50) * disk_sed
   galaxy = bulge + disk

This is the natural bridge from catalog morphology plus SED information to
chromatic image simulation. The JAX-GalSim implementation should be attempted
only after a minimal standard-GalSim prototype is numerically understood. In
the local ``shine`` environment, importing JAX-GalSim currently segfaults, so
this needs environment repair before implementation.

Immediate Science Roadmap
-------------------------

1. Add LSST bands to the default fitting configuration.

   VIS+YJH gives only four photometric points. Fitting redshift, SFR,
   metallicity, and dust from four bands is under-constrained. The SQL already
   downloads LSST ugrizy, so the next science configuration should use all ten
   bands.

2. Add explicit amplitude/mass.

   The current model uses ``log10_sfr`` as the SFH amplitude. A physical SPS
   fit should include stellar mass or formed mass as an explicit parameter and
   derive recent SFR from the SFH shape.

3. Treat dust and metallicity truth columns as proxies.

   Keep reporting them, but rename output labels to make clear they are catalog
   proxies unless a validated mapping is added.

4. Add a "fixed catalog redshift" and "fit redshift" comparison.

   Run the same galaxies with ``z_obs=z_phz``, ``z_obs=z_true``, and sampled
   ``z_obs``. This isolates errors from PHZ, SPS mismatch, and dust/SFH
   degeneracy.

5. Query template-level SED metadata.

   Without SED/template IDs, the project cannot reconstruct catalog
   ground-truth spectra. This is a data-contract issue, not a DSPS issue.

6. Prototype standard GalSim chromatic drawing.

   First draw one Sersic galaxy multiplied by a tabulated DSPS SED through one
   Euclid bandpass. Then add bulge+disk SEDs. Only after that move the same
   calculation into JAX-GalSim.

References
----------

* DSPS paper:
  https://academic.oup.com/mnras/article/521/2/1741/7034352
* PROVABGS mock challenge:
  https://arxiv.org/abs/2202.01809
* SEDflow:
  https://arxiv.org/abs/2203.07391
* SEDflow data model:
  https://changhoonhahn.github.io/SEDflow/current/datamodel/
* PROVABGS code summary:
  https://ascl.net/2407.006
* pop-cosmos 2024 inference paper:
  https://arxiv.org/abs/2406.19437
* pop-cosmos 2025 extended model:
  https://arxiv.org/abs/2506.12122
* pop-cosmos v2 data products:
  https://zenodo.org/records/15623082
* Euclid Flagship 2 public release:
  https://www.euclid-ec.org/public/press-releases/euclid-flagship-simulations/
* Euclid stacked-spectroscopy attenuation discussion:
  https://www.aanda.org/articles/aa/full_html/2022/04/aa42224-21/aa42224-21.html
* GalSim wavelength-dependent profiles:
  https://galsim-developers.github.io/GalSim/_build/html/chromatic.html
* GalSim chromatic objects:
  https://galsim-developers.github.io/GalSim/_build/html/chromaticobject.html
